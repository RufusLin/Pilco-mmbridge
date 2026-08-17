# `modules.multimodal`

This package contains the complete MM-Bridge request pipeline: configuration loading, authentication, media detection, question-aware visual analysis, OCR evidence injection, caching, upstream forwarding, and SSE streaming.

## Files

| File | Responsibility |
|---|---|
| `app.py` | FastAPI application, routing, request pipeline, model discovery, debug dumps, analyzer calls, and final upstream forwarding. |
| `config.py` | Reads environment variables into the immutable `Settings` dataclass and defines supported policy types. |
| `context.py` | Separates the active real-user request from the current direct-user or tool-result event so historical media is not reanalyzed. |
| `media.py` | Finds image, audio, and video blocks; estimates media size; computes hashes; and keeps, replaces, or removes media in the final request. |
| `prompting.py` | Extracts the exact user request, builds analyzer requests, defines OCR-safe prompts, injects analysis into the original protocol, rewrites model IDs, and extracts analyzer text. |
| `cache.py` | Stores analyzer responses in a file cache keyed by prompt version, endpoint, analyzer model, media hashes, and the exact user request. |
| `security.py` | Validates client authentication and request-body size. |
| `upstream.py` | Builds upstream URLs and headers, sends HTTP requests, forwards responses, and handles SSE streaming and heartbeat messages. |
| `__init__.py` | Marks the directory as a Python package. |

## Request flow without media

```text
Client request
  → check_client_auth()
  → assert_body_size()
  → parse JSON when possible
  → extract_current_request_context() returns no current-event media
  → discover or use the final text-model ID
  → rewrite_model()
  → forward to VLLM_ROOT_URL using the original endpoint
```

The vision analyzer is not called.

## Request flow with media

```text
Client request + media
  → authenticate and validate body size
  → identify the current direct-user or tool-result event
  → detect only media added by that event
  → enforce item-count and per-item size limits
  → verify that the endpoint supports analyzer injection
  → retain the latest actual user request that started the active tool chain
  → discover or use the vision-model ID
  → compute a question-aware cache key
     ├─ cache hit: reuse the analysis
     └─ cache miss:
          → build_analyzer_body()
          → send media + exact user request to LLAMA_ROOT_URL
          → extract and validate analyzer evidence JSON
          → store the result in the cache
  → inject_analysis()
  → keep, replace, or strip original media
  → rewrite the bridge model alias
  → forward to VLLM_ROOT_URL using the original endpoint
  → return the final response or SSE stream
```

## Protocol preservation

The bridge builds the analyzer request in the same protocol as the incoming endpoint.

| Incoming endpoint | Analyzer endpoint | Final text endpoint | Injection location |
|---|---|---|---|
| `/v1/chat/completions` | `/v1/chat/completions` | `/v1/chat/completions` | A new `system` message |
| `/v1/messages` | `/v1/messages` | `/v1/messages` | Top-level `system` |
| `/v1/responses` | `/v1/responses` | `/v1/responses` | `instructions` |

No OpenAI-to-Anthropic or Anthropic-to-OpenAI translation is performed.

## Current request context

`extract_current_request_context()` separates two values:

- The latest actual user request that started the active request or tool chain
- Media added by the current input event only

A direct user image uses text and media from that same user event. A trailing tool result uses only media from the latest tool-result batch while retaining the earlier actual user request. Tool-result wrappers represented as `role: "user"` are not treated as new user intent when protocol structure identifies them as part of the active tool call.

Historical media is never collected merely because it remains in the request body. The original client request is not replaced; its text is copied only into the analyzer request.

## Analyzer request

`build_analyzer_body()` sends the vision model:

```text
1. ANALYZER_SYSTEM_PROMPT
2. The user's exact current request
3. A media summary containing item metadata
4. Media blocks normalized for the incoming analyzer protocol
```

Analyzer requests always use `stream: false` because the bridge must receive the complete analysis before it can build the final text-model request.

Token fields are selected by protocol:

```text
/v1/chat/completions → max_tokens
/v1/messages         → max_tokens
/v1/responses        → max_output_tokens
```

The value comes from `MM_ANALYZER_MAX_TOKENS`. The project recommends at least `16384` for complex OCR, game screens, source-code screenshots, terminal output, and multi-image requests.

## Multimodal evidence output

The analyzer returns a single top-level `media` array. It does not return a top-level summary, candidate answer, or recommendation.

### Text blocks

`text_blocks` contain verbatim OCR, speech, or lyrics. `source` identifies which kind of text was extracted, while `location` records an image position, audio time range, or video time and position.

```json
{
  "source": "ocr",
  "text": "Connection refused",
  "location": "center dialog body",
  "confidence": "high",
  "uncertainty": ""
}
```

The analyzer preserves meaningful line breaks, casing, punctuation, commands, paths, URLs, error codes, timestamps, labels, and numbers. It does not correct, translate, paraphrase, summarize, or complete missing text. Unclear spans use `[unreadable]` or `[uncertain: candidate1|candidate2]`.

### Visual blocks

Images and videos with visual content may include `visual_blocks` for UI states, document structure, tables, charts, diagrams, objects, and spatial relationships. The bridge does not assign or require a semantic `kind`; it passes the analyzer's evidence to the final model without classifying it again.

```json
{
  "subject": "Save button",
  "description": "The button is grey and appears disabled.",
  "location": "form bottom-right",
  "relationships": [
    "The same form shows a required-email validation error."
  ],
  "basis": "observed",
  "confidence": "high",
  "uncertainty": ""
}
```

`basis="observed"` marks directly visible evidence. `basis="inferred"` marks a supported conclusion and must identify its visible basis and uncertainty.

### Audio blocks

Audio and videos with an audio track may include `audio_blocks`. Spoken words and lyrics remain in `text_blocks`; `audio_blocks` contain non-verbal sound events and supported music analysis.

```json
{
  "description": "A slow minor-key instrumental passage.",
  "location": "00:10.000-00:28.500",
  "musical_features": {
    "style_or_character": "somber and sustained",
    "tonal_center": "A minor",
    "harmony": "repeating Am-F-C-G progression",
    "melody": "descending phrase repeated by strings",
    "rhythm_meter_tempo": "4/4, approximately 70 BPM",
    "form": "repeating eight-bar section",
    "instrumentation": "strings and soft piano"
  },
  "confidence": "medium",
  "uncertainty": "The tonal center may shift briefly near the end."
}
```

Musical details must be omitted or marked uncertain when they cannot be heard reliably.

### Complete image example

```json
{
  "media": [
    {
      "index": 1,
      "type": "image",
      "text_blocks": [
        {
          "source": "ocr",
          "text": "Email is required",
          "location": "below the email input",
          "confidence": "high",
          "uncertainty": ""
        }
      ],
      "visual_blocks": [
        {
          "subject": "Email input",
          "description": "The input is empty and has a red error border.",
          "location": "form center",
          "relationships": [
            "The required-email message is immediately below it."
          ],
          "basis": "observed",
          "confidence": "high",
          "uncertainty": ""
        }
      ]
    }
  ]
}
```

## Final context injection

`inject_analysis()` deep-copies the original request before modifying it.

The injected context tells the final text model:

- `text_blocks[].text` is the canonical verbatim OCR, speech, and lyrics evidence.
- `text_blocks[].source` distinguishes OCR, speech, and lyrics.
- Exact transcription requests must use the text blocks.
- Original text must be shown separately before translation or explanation when both are requested.
- `visual_blocks[].basis` distinguishes direct observations from fallible inferences.
- `audio_blocks` contain non-verbal sound events or fallible musical analysis.
- Uncertainty markers must be preserved.

## Question-aware cache

`make_cache_key()` uses:

```text
ANALYZER_PROMPT_VERSION
endpoint
vision-model ID
media hashes
exact user request
```

Therefore, the same image with different questions creates different cache entries.

```text
"What is the health value?"
"Where is the exit?"
"Copy the warning text exactly."
```

The cache files are stored under `MM_ANALYZER_CACHE_DIR` and may contain analyzer output. Do not commit the cache directory.

## Media detection and replacement

`find_media_items()` recursively scans the selected current event for supported media forms, including common OpenAI, Anthropic, and MCP content blocks.

Representative supported forms include:

```text
image_url
input_image
input_audio
input_video
Anthropic image/audio/video source blocks
MCP image/audio content using data and mimeType
data:...;base64,... URLs
recognized HTTP, HTTPS, and file media URLs
```

`MM_VLLM_MEDIA_POLICY` controls the final request:

- `keep`: retain original media
- `replace`: replace media with a text placeholder
- `strip`: remove media

Use `replace` for a text-only final model.

## Model discovery

When `VLLM_MODEL` or `LLAMA_MODEL` is empty, the bridge calls `GET /v1/models` on the corresponding server and uses the first returned model ID. The discovered ID is cached in the running HTTP client object.

## Streaming

Only the final upstream response is streamed. Analyzer responses are collected completely first.

For SSE responses, `upstream.py`:

- Preserves complete SSE event boundaries
- Requests identity encoding when possible
- Handles compressed upstream streams safely
- Sends comment-style heartbeat events during idle periods
- Adds `x-mm-bridge-stage` and `x-mm-bridge-request-id` response headers

The heartbeat interval comes from `MM_STREAM_HEARTBEAT_SECONDS`.

## Authentication and limits

`security.py` accepts either:

```http
Authorization: Bearer <MM_BRIDGE_TOKEN>
```

or:

```http
x-api-key: <MM_BRIDGE_TOKEN>
```

If `MM_BRIDGE_TOKEN` is empty, authentication is disabled.

Request and media limits are controlled by:

```text
MM_MAX_BODY_BYTES
MM_MAX_MEDIA_BYTES
MM_MAX_MEDIA_ITEMS
```

## Debug files

When `MM_DEBUG_DUMP=true`, `app.py` may write files such as:

```text
incoming.json
incoming_direct.json
llama_request.json
llama_response.body
vllm_request.json
```

These files may contain user prompts, base64 media, internal URLs, analyzer output, or final model requests. Keep `.debug/` out of Git and delete debug data after diagnosis.

## Current limitations

- Field-level validation checks the supported evidence contract and truncation signals, but semantic OCR and visual-analysis accuracy still depends on the analyzer.
- The current implementation performs one analyzer call per cache miss.
- There is no automatic crop, zoom, OCR retry, or second-pass inspection.
- The vision model may still misread small or compressed text.
- The final text model cannot recover visual information omitted by the analyzer.
