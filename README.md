# MM-Bridge

MM-Bridge is a FastAPI proxy that connects a text-only reasoning model to a separate multimodal analyzer.

When a request contains no media, the bridge forwards it directly to the final text-model server. When a request contains an image, audio clip, or video, the bridge first asks the vision server to extract visual evidence, then injects that evidence into the original request before forwarding it to the text model.

The bridge is designed for setups such as:

- **Final reasoning model:** a text-only model served by vLLM
- **Vision analyzer:** a multimodal model served by vLLM or another compatible server
- **Clients:** OpenWebUI, Cline, Claude Code, or any compatible API client

> The existing `LLAMA_*` environment-variable names are kept for compatibility. `LLAMA_ROOT_URL` may point to a vision-capable vLLM server; llama.cpp is not required.

## Documentation

- [Environment configuration](docs/ENVIRONMENT.md)
- [Multimodal module internals](modules/multimodal/README.md)

## What the bridge does

```text
Client
  │
  ▼
MM-Bridge
  ├─ No media ───────────────────────────────► Final text model
  │
  └─ Media present
       ├─ Send media + exact user request ───► Vision analyzer
       ├─ Receive OCR and visual evidence
       ├─ Inject the evidence into the original request
       └─────────────────────────────────────► Final text model
```

The bridge preserves the incoming API protocol and endpoint. It does not translate OpenAI requests into Anthropic requests or vice versa.

```text
/v1/chat/completions → vision /v1/chat/completions → text /v1/chat/completions
/v1/messages         → vision /v1/messages         → text /v1/messages
/v1/responses        → vision /v1/responses        → text /v1/responses
```

## Main features

- Direct pass-through for requests without media
- Question-aware visual analysis for requests with media
- Verbatim OCR, speech, and lyrics blocks separated from visual and audio analysis
- Question-aware analysis cache
- OpenAI- and Anthropic-compatible request handling
- Optional media replacement for text-only final models
- SSE streaming with optional heartbeat messages
- Automatic model discovery through `/v1/models`

## Multimodal evidence blocks

The analyzer returns one `media` array without a top-level summary or candidate answer.

```text
text_blocks
└─ Verbatim OCR, speech, and lyrics with spatial or temporal locations

visual_blocks
└─ Image/video UI states, document structure, tables, charts, diagrams, and spatial relationships

audio_blocks
└─ Audio/video sound events and supported musical analysis
```

For an exact transcription request, the final text model is instructed to use `text_blocks[].text` rather than replacing it with a translation, correction, summary, or inferred meaning. `visual_blocks[].basis` distinguishes directly observed evidence from inferred conclusions.

The user's exact request guides inspection depth without allowing relevant text, UI state, document structure, chart relationships, or audio events to be omitted.

> This separation is defined by prompting and validated field conventions. Validation rejects malformed, truncated, or unsupported evidence output, but it cannot guarantee perfect OCR.

## Requirements

- Python 3.10 or newer
- A text-model server compatible with the endpoints used by your client
- A multimodal vision-model server compatible with the same endpoints
- The Python packages listed in `requirements.txt`

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### 2. Create and activate a virtual environment

#### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Windows Command Prompt

```bat
py -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Install dependencies from `requirements.txt`

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Create the local `.env` file

#### Linux or macOS

```bash
cp .env.example .env
```

#### Windows PowerShell

```powershell
Copy-Item .env.example .env
```

Generate a new bridge token:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Copy the generated value into `MM_BRIDGE_TOKEN` in your local `.env` file.

### 5. Configure the two upstream servers

Use root URLs without a trailing `/v1`:

```env
VLLM_ROOT_URL=http://127.0.0.1:8000
LLAMA_ROOT_URL=http://127.0.0.1:8001
```

- `VLLM_ROOT_URL` is the final text-model server.
- `LLAMA_ROOT_URL` is the vision-analyzer server.
- Either server may be vLLM as long as it provides the required compatible API.

For text-only final models, use:

```env
MM_VLLM_MEDIA_POLICY=replace
```

For complex OCR, game screens, terminal screenshots, and multi-image requests, this project recommends at least:

```env
MM_ANALYZER_MAX_TOKENS=16384
```

The configured model context length and server limits must support the selected value.

See [Environment configuration](docs/ENVIRONMENT.md) for every available variable.


## Basic checks

### Health check

```bash
curl http://127.0.0.1:18000/health
```

> The current health response may include upstream configuration details. Do not expose `/health` publicly without access control or a reverse-proxy rule.

### Model list

```bash
curl \
  -H "Authorization: Bearer YOUR_LOCAL_BRIDGE_TOKEN" \
  http://127.0.0.1:18000/v1/models
```

### Text request

```bash
curl http://127.0.0.1:18000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_LOCAL_BRIDGE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-mm-bridge",
    "messages": [
      {"role": "user", "content": "Reply with one short sentence."}
    ],
    "stream": false
  }'
```

## Repository layout

```text
.
├─ main.py
├─ requirements.txt
├─ .env.example
├─ .gitignore
├─ README.md
├─ docs/
│  └─ ENVIRONMENT.md
├─ modules/
│  ├─ __init__.py
│  └─ multimodal/
│     ├─ __init__.py
│     ├─ README.md
│     ├─ app.py
│     ├─ cache.py
│     ├─ config.py
│     ├─ context.py
│     ├─ media.py
│     ├─ prompting.py
│     ├─ security.py
│     └─ upstream.py
└─ scripts/
   ├─ run_mm_bridge.bat
   └─ run_mm_bridge.sh
```

## Important limitations

- The final text model does not receive visual tokens directly.
- Visual information omitted by the analyzer cannot be reconstructed by the text model.
- OCR accuracy depends on image quality, text size, compression, and the vision model.
- Evidence validation checks the supported schema and truncation signals, but it does not prove that the analyzer observed every real detail correctly.
- Small HUD values and dense screenshots may require higher-resolution inputs or future crop/retry logic.
