# tae

A non-invasive Discord roleplay bot. Silently watches a channel and drops an LLM-generated message when specific conditions are met. No slash commands, no mention listeners.

---

## Docker (recommended)

```bash
cp .env.example .env
# edit .env — at minimum set DISCORD_TOKEN, TARGET_CHANNEL_ID, LLAMA_MODEL
```

Put your `.gguf` model file in `./models/`, then:

```bash
docker compose up -d
```

The bot and llama-server share an internal `tae-ai-network` bridge. The bot connects to llama-server by service name — `LLAMACPP_URL` defaults to `http://tae-llama-server:8080/v1/chat/completions` so you don't need to change it for local use.

> **Note:** llama-server can take 30–90 seconds to load the model. The bot will log connection errors during that window and retry on the next trigger.

---

## Running without Docker

```bash
pip install -r requirements.txt
export DISCORD_TOKEN=...
export TARGET_CHANNEL_ID=...
# set LLAMACPP_URL to http://127.0.0.1:8080/v1/chat/completions for a local server
python bot.py
```

---

## Configuration

All configuration is driven by environment variables. Set them in `.env` (copy from `.env.example`) for Docker, or `export` them when running directly.

| Variable | Default | Description |
|---|---|---|
| `DISCORD_TOKEN` | *(required)* | Your bot token from the [Discord Developer Portal](https://discord.com/developers/applications) |
| `TARGET_CHANNEL_ID` | *(required)* | Integer ID of the channel to watch. Right-click the channel → *Copy Channel ID* (requires Developer Mode) |
| `LLAMA_MODEL` | — | Filename of the `.gguf` inside `./models/` — used by the llama-server container |
| `LLAMACPP_URL` | `http://tae-llama-server:8080/v1/chat/completions` | Any OpenAI-compatible `/v1/chat/completions` endpoint |
| `LLM_API_KEY` | *(empty)* | API key — only needed for hosted APIs (OpenAI, Together, etc.) |
| `HISTORY_LIMIT` | `40` | How many recent messages to pull as context per generation |
| `VOLUME_TRIGGER_N` | `40` | Fire after every N messages. **Set to `200` here to change the threshold** |
| `LENGTH_TRIGGER_CHARS` | `300` | Fire immediately if a single message exceeds this character count |
| `SILENCE_TRIGGER_MINUTES` | `60` | Fire if the channel has been quiet for this many minutes |
| `LLM_MODEL` | `local-model` | Model name in the API payload. Ignored by llama.cpp; set to `gpt-4o` etc. for hosted APIs |
| `LLM_MAX_TOKENS` | `256` | Max tokens the LLM generates per response |
| `LLM_TEMPERATURE` | `0.85` | Sampling temperature |
| `SYSTEM_PROMPT` | *(sarcastic bystander)* | The persona. Single line. This is the only thing that defines the bot's character |

### Using OpenAI or another hosted API

Set `LLAMACPP_URL` to the provider's endpoint and provide an `LLM_API_KEY`:

```env
LLAMACPP_URL=https://api.openai.com/v1/chat/completions
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o
```

### Changing the persona

Edit `SYSTEM_PROMPT` in `.env`. Example:

```env
SYSTEM_PROMPT=You are a medieval scholar who has somehow ended up in a modern Discord server. React to everything with Shakespearean confusion. One sentence only.
```

---

## How triggers work

Three independent conditions can fire a generation. All three share one concurrency lock — if a generation is already in-flight, subsequent triggers are silently dropped.

**Volume counter** — increments on every user message, fires at `VOLUME_TRIGGER_N`, then resets to 0.

**Length detector** — fires immediately when a single message exceeds `LENGTH_TRIGGER_CHARS`. Resets the volume counter.

**Silence breaker** — a background task polls every 5 minutes. If `SILENCE_TRIGGER_MINUTES` have passed since the last message, it fires once and re-arms itself.

---

The bot requires the `message_content` privileged intent. Enable it in the Developer Portal under your application → *Bot* → *Privileged Gateway Intents*.
