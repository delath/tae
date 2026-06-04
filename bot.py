"""
Discord Roleplay Bot — single-file, fully async, stateless memory.

Triggers an LLM response (via local llama.cpp) under three conditions:
  1. Every 40th user message in the watched channel  (volume counter)
  2. Any single user message longer than 300 chars   (length detector)
  3. 60+ minutes of channel silence                  (silence breaker)
"""

import asyncio
import datetime
import logging
import os
import re

import discord
import httpx
from discord.ext import tasks

# ──────────────────────────────────────────────
#  CONFIGURATION  — all values read from env.
#  Set them in .env (Docker) or export them
#  manually before running directly.
# ──────────────────────────────────────────────

# Required — no defaults, will raise immediately on startup if missing.
DISCORD_TOKEN        = os.environ["DISCORD_TOKEN"]
TARGET_CHANNEL_ID    = int(os.environ["TARGET_CHANNEL_ID"])

# LLM endpoint. Default points to the tae-llama-server service name
# on the shared Docker bridge; change to any OpenAI-compatible URL.
LLAMACPP_URL         = os.getenv("LLAMACPP_URL", "http://tae-llama-server:8080/v1/chat/completions")

# Optional API key — required for hosted APIs (OpenAI, Together, etc.),
# leave blank or unset when talking to a local llama.cpp server.
LLM_API_KEY          = os.getenv("LLM_API_KEY", "")

# Context window: how many recent messages to include per generation.
HISTORY_LIMIT        = int(os.getenv("HISTORY_LIMIT", "40"))

# Volume trigger: fire after every N user messages.
VOLUME_TRIGGER_N     = int(os.getenv("VOLUME_TRIGGER_N", "40"))

# Length trigger: fire immediately if a single message exceeds N chars.
LENGTH_TRIGGER_CHARS = int(os.getenv("LENGTH_TRIGGER_CHARS", "300"))

# Silence trigger: fire if the channel is quiet for N minutes.
SILENCE_TRIGGER_MINUTES = int(os.getenv("SILENCE_TRIGGER_MINUTES", "60"))

# Model name sent in the API payload.
# llama.cpp ignores this; set to a real name for hosted APIs ("gpt-4o", etc.).
LLM_MODEL            = os.getenv("LLM_MODEL", "local-model")

LLM_MAX_TOKENS       = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_TEMPERATURE      = float(os.getenv("LLM_TEMPERATURE", "0.85"))

# The system persona prepended to every prompt.
# Keep it on a single line in .env; use \n for newlines if needed.
_DEFAULT_SYSTEM_PROMPT = (
    "You are a sarcastic, blunt, and informal bystander lurking in a Discord chat. "
    "You pop in occasionally with short, dry remarks or observations about whatever "
    "is being discussed. Keep replies brief — ideally one or two sentences. "
    "Never break character. Never introduce yourself. Never explain your behaviour."
)
SYSTEM_PROMPT        = os.getenv("SYSTEM_PROMPT", _DEFAULT_SYSTEM_PROMPT)

# ──────────────────────────────────────────────
#  LOGGING
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("roleplay-bot")

# ──────────────────────────────────────────────
#  BOT SETUP
# ──────────────────────────────────────────────

intents = discord.Intents.default()
intents.message_content = True   # required to read message text

bot = discord.Client(intents=intents)

# ──────────────────────────────────────────────
#  MUTABLE STATE  (minimal — only counters/flags)
# ──────────────────────────────────────────────

# Counts user messages since the last volume-trigger reset.
message_counter: int = 0

# UTC timestamp of the last user message seen in TARGET_CHANNEL_ID.
# Initialised to "now" at startup so the silence-breaker doesn't
# fire immediately on a fresh launch.
last_message_time: datetime.datetime = datetime.datetime.now(datetime.timezone.utc)

# Concurrency guard: True while an LLM request is in-flight.
# Prevents pile-up if multiple triggers fire close together.
is_generating: bool = False


# ──────────────────────────────────────────────
#  CORE LLM ROUTINE
# ──────────────────────────────────────────────

async def generate_and_send(
    channel: discord.TextChannel,
    reason: str,
    reply_to: discord.Message | None = None,
) -> None:
    """
    Fetch recent channel history, build an OpenAI-schema prompt,
    POST it to the local llama.cpp server, then send the reply.

    If reply_to is provided, the bot replies directly to that message
    and appends a focused prompt nudge so the LLM addresses it specifically.

    This coroutine is always launched via asyncio.create_task() so it
    never blocks the main event loop / Discord heartbeat.
    """
    global is_generating

    # ── Concurrency guard ──────────────────────────────────────────
    if is_generating:
        log.info("Trigger '%s' skipped — generation already in-flight.", reason)
        return

    is_generating = True
    t_start = datetime.datetime.now(datetime.timezone.utc)
    log.info("[%s] Generation triggered by: %s", t_start.strftime("%H:%M:%S"), reason)

    try:
        # ── Fetch message history (stateless; fresh every call) ────
        # discord.py v2 removed .flatten(); use an async comprehension instead.
        # history() yields newest-first, so reverse for chronological order.
        log.debug("Fetching up to %d messages from channel #%s (%d).",
                  HISTORY_LIMIT, channel.name, channel.id)
        raw_history = [msg async for msg in channel.history(limit=HISTORY_LIMIT)]
        chronological = list(reversed(raw_history))
        log.debug("Fetched %d raw messages from Discord.", len(chronological))

        # ── Build OpenAI-schema messages array ────────────────────
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        skipped_empty = 0
        skipped_self  = 0
        for msg in chronological:
            # Skip empty messages (images-only, stickers, etc.)
            if not msg.content.strip():
                skipped_empty += 1
                continue

            # Skip messages sent by this bot to avoid self-referential loops
            if msg.author.id == bot.user.id:
                skipped_self += 1
                continue

            messages.append({
                "role": "user",
                "content": f"{msg.author.display_name}: {msg.content}",
            })

        user_msgs = len(messages) - 1  # exclude system prompt
        log.info("History built: %d user messages included, %d empty skipped, "
                 "%d self skipped.", user_msgs, skipped_empty, skipped_self)

        # Guard: if there's nothing but the system prompt, bail out
        if len(messages) == 1:
            log.warning("No usable message content found; skipping generation.")
            return

        # When triggered by a long message, append a nudge as the final entry
        # so the LLM focuses its reply on that specific message rather than
        # producing a generic observation about the whole conversation.
        if reply_to is not None:
            nudge = (
                f"[The last message from {reply_to.author.display_name} was "
                f"unusually long. React to it specifically and concisely.]"
            )
            messages.append({"role": "user", "content": nudge})
            log.debug("Appended length-nudge for message from %s (%d chars).",
                      reply_to.author.display_name, len(reply_to.content))

        payload = {
            "model":       LLM_MODEL,
            "messages":    messages,
            "max_tokens":  LLM_MAX_TOKENS,
            "temperature": LLM_TEMPERATURE,
        }

        log.debug("Payload: model=%s, messages=%d, max_tokens=%d, temperature=%.2f",
                  LLM_MODEL, len(messages), LLM_MAX_TOKENS, LLM_TEMPERATURE)
        log.debug("System prompt (%d chars): %.120s%s",
                  len(SYSTEM_PROMPT), SYSTEM_PROMPT,
                  "..." if len(SYSTEM_PROMPT) > 120 else "")

        # ── POST to llama.cpp (fully async — does NOT block heartbeat) ──
        # Include Authorization header only when an API key is configured.
        headers = {"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {}
        log.info("POSTing to %s (timeout=300s, max_tokens=%d) …",
                 LLAMACPP_URL, LLM_MAX_TOKENS)
        t_request = datetime.datetime.now(datetime.timezone.utc)
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(LLAMACPP_URL, json=payload, headers=headers)

        t_response = datetime.datetime.now(datetime.timezone.utc)
        elapsed_req = (t_response - t_request).total_seconds()
        log.info("llama.cpp responded: HTTP %d in %.1fs.",
                 response.status_code, elapsed_req)

        response.raise_for_status()
        data = response.json()

        # Log usage stats when the server returns them
        usage = data.get("usage", {})
        if usage:
            log.info("Token usage — prompt: %s, completion: %s, total: %s.",
                     usage.get("prompt_tokens", "?"),
                     usage.get("completion_tokens", "?"),
                     usage.get("total_tokens", "?"))

        finish_reason = data.get("choices", [{}])[0].get("finish_reason", "unknown")
        log.info("Finish reason: %s.", finish_reason)
        if finish_reason == "length":
            log.warning("Model hit max_tokens (%d) — reply may be truncated. "
                        "Consider raising LLM_MAX_TOKENS.", LLM_MAX_TOKENS)

        # Extract the assistant's reply from the standard OpenAI response shape.
        # Thinking models (e.g. Qwen3) may wrap chain-of-thought in <think>…</think>
        # before the actual response. Strip those blocks before checking for content.
        raw_content: str = data["choices"][0]["message"]["content"] or ""
        log.debug("Raw content from llama.cpp (%d chars): %.200s%s",
                  len(raw_content), raw_content,
                  "..." if len(raw_content) > 200 else "")

        think_blocks = re.findall(r"<think>[\s\S]*?</think>", raw_content)
        if think_blocks:
            total_think_chars = sum(len(b) for b in think_blocks)
            log.info("Stripped %d <think> block(s) totalling %d chars.",
                     len(think_blocks), total_think_chars)

        reply_text: str = re.sub(r"<think>[\s\S]*?</think>", "", raw_content).strip()
        log.debug("Reply after stripping (%d chars): %.200s%s",
                  len(reply_text), reply_text,
                  "..." if len(reply_text) > 200 else "")

        if not reply_text:
            log.warning("llama.cpp returned an empty reply after stripping think "
                        "blocks; raw content was %d chars. Full raw: %r",
                        len(raw_content), raw_content[:500])
            return

        log.info("Sending reply (%d chars) to #%s.", len(reply_text), channel.name)
        if reply_to is not None:
            await reply_to.reply(reply_text)
        else:
            await channel.send(reply_text)

    except httpx.TimeoutException as exc:
        elapsed = (datetime.datetime.now(datetime.timezone.utc) - t_start).total_seconds()
        log.error("llama.cpp request timed out after %.1fs (timeout=300s): %r",
                  elapsed, exc)
    except httpx.HTTPStatusError as exc:
        log.error("llama.cpp HTTP error %s — response body: %s",
                  exc.response.status_code, exc.response.text[:500],
                  exc_info=True)
    except httpx.RequestError as exc:
        log.error("llama.cpp request failed (%s): %r", type(exc).__name__, exc,
                  exc_info=True)
    except (KeyError, IndexError) as exc:
        log.error("Unexpected llama.cpp response shape: %r — data dump: %r",
                  exc, str(data)[:500] if 'data' in dir() else '<no data>',
                  exc_info=True)
    except discord.HTTPException as exc:
        log.error("Discord send failed (status %s): %s", exc.status, exc, exc_info=True)
    finally:
        elapsed_total = (datetime.datetime.now(datetime.timezone.utc) - t_start).total_seconds()
        is_generating = False
        log.info("Generation complete in %.1fs; lock released.", elapsed_total)


# ──────────────────────────────────────────────
#  DISCORD EVENTS
# ──────────────────────────────────────────────

@bot.event
async def on_ready() -> None:
    """Called once the bot has connected and its internal cache is ready."""
    log.info("Logged in as %s (ID: %s)", bot.user, bot.user.id)
    log.info("Watching channel ID: %s", TARGET_CHANNEL_ID)
    log.info("Config — LLM: url=%s model=%s max_tokens=%d temperature=%.2f history=%d",
             LLAMACPP_URL, LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE, HISTORY_LIMIT)
    log.info("Config — triggers: volume=%d length=%d silence=%dmin",
             VOLUME_TRIGGER_N, LENGTH_TRIGGER_CHARS, SILENCE_TRIGGER_MINUTES)
    log.debug("System prompt: %s", SYSTEM_PROMPT)

    # Start the background silence-check loop now that we have a valid session.
    # Guard against accidental double-start (e.g. reconnects).
    # Set SILENCE_TRIGGER_MINUTES=0 in .env to disable this feature entirely.
    if SILENCE_TRIGGER_MINUTES > 0:
        if not silence_check.is_running():
            silence_check.start()
            log.info("Silence-breaker loop started (polls every 5 minutes).")
    else:
        log.info("Silence-breaker disabled (SILENCE_TRIGGER_MINUTES=0).")


@bot.event
async def on_message(message: discord.Message) -> None:
    """
    Primary event handler. Evaluated on every message the bot can see.
    Triggers are checked in priority order; only one fires per message.
    """
    global message_counter, last_message_time

    # ── Ignore anything outside the target channel ─────────────────
    if message.channel.id != TARGET_CHANNEL_ID:
        return

    # ── Ignore the bot's own messages ──────────────────────────────
    if message.author.id == bot.user.id:
        return

    # ── Ignore bots (webhooks, other bots) ─────────────────────────
    if message.author.bot:
        return

    # ── Update the silence-breaker's timestamp ─────────────────────
    last_message_time = datetime.datetime.now(datetime.timezone.utc)

    # ── Increment the volume counter ───────────────────────────────
    message_counter += 1
    log.debug("Message counter: %d | length: %d", message_counter, len(message.content))

    # ── TRIGGER 2: Length detector (highest priority) ───────────────
    # Checked before the volume counter so a long message resets the
    # counter without accidentally double-firing both triggers.
    if len(message.content) > LENGTH_TRIGGER_CHARS:
        log.info("Length trigger fired (%d chars).", len(message.content))
        message_counter = 0   # reset counter on any non-volume trigger
        asyncio.create_task(
            generate_and_send(message.channel, reason="length-detector", reply_to=message)
        )
        return  # skip volume check this cycle

    # ── TRIGGER 1: Volume counter ───────────────────────────────────
    if message_counter >= VOLUME_TRIGGER_N:
        log.info("Volume trigger fired at message #%d.", message_counter)
        message_counter = 0   # reset immediately — before the async task runs
        asyncio.create_task(
            generate_and_send(message.channel, reason="volume-counter")
        )

    # NOTE: allow_commands is intentionally omitted — this bot has no prefix
    # commands.  If you add them later, call: await bot.process_commands(message)


# ──────────────────────────────────────────────
#  BACKGROUND TASK — SILENCE BREAKER
# ──────────────────────────────────────────────

@tasks.loop(minutes=5)
async def silence_check() -> None:
    """
    Runs every 5 minutes. If the watched channel has been silent for
    SILENCE_TRIGGER_MINUTES or more, fire the LLM routine.

    Polling every 5 min rather than every 1 min keeps CPU/network
    overhead negligible while still catching the 60-min threshold
    within a 5-minute window of accuracy.
    """
    global last_message_time

    now = datetime.datetime.now(datetime.timezone.utc)
    silence_duration = now - last_message_time
    silence_minutes  = silence_duration.total_seconds() / 60

    log.debug("Silence check: %.1f min since last message.", silence_minutes)

    if silence_minutes >= SILENCE_TRIGGER_MINUTES:
        log.info("Silence trigger fired (%.1f min of quiet).", silence_minutes)

        # Resolve the channel object from the cache.
        channel = bot.get_channel(TARGET_CHANNEL_ID)
        if channel is None:
            log.warning("Silence trigger: target channel not in cache; skipping.")
            return

        # Reset the silence clock immediately so the trigger doesn't
        # fire repeatedly on every 5-min poll after a long silence.
        # (It will reset again properly on the next real user message.)
        last_message_time = now

        asyncio.create_task(
            generate_and_send(channel, reason="silence-breaker")
        )


@silence_check.error
async def silence_check_error(error: Exception) -> None:
    """Log errors from the silence-check loop so it doesn't die silently."""
    log.error("silence_check task raised an exception: %s", error, exc_info=error)


@silence_check.before_loop
async def before_silence_check() -> None:
    """Ensures the bot is fully connected before the loop begins polling."""
    await bot.wait_until_ready()


# ──────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN, log_handler=None)  # log_handler=None uses our own config
