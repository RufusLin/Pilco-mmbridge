# Environment Configuration

Copy `.env.example` to `.env`, then edit the local `.env` file for your servers.

```bash
cp .env.example .env
```

Do not commit the real `.env` file. It may contain credentials, private IP addresses, internal hostnames, or service URLs.

## Recommended production baseline

```env
MM_BRIDGE_HOST=127.0.0.1
MM_BRIDGE_PORT=18000
MM_BRIDGE_TOKEN=REPLACE_WITH_A_LONG_RANDOM_TOKEN
MM_BRIDGE_MODEL_ID=deepseek-v4-mm-bridge
MM_BRIDGE_MODE=1
MM_LOG_LEVEL=INFO

VLLM_ROOT_URL=http://127.0.0.1:8000
VLLM_API_KEY=EMPTY
VLLM_AUTH_STYLE=auto
VLLM_MODEL=

LLAMA_ROOT_URL=http://127.0.0.1:8001
LLAMA_API_KEY=EMPTY
LLAMA_AUTH_STYLE=auto
LLAMA_MODEL=

MM_ANALYZER_ENDPOINTS=/v1/chat/completions,/v1/messages,/v1/responses
MM_ANALYZER_FORCE_STREAM_FALSE=true
MM_ANALYZER_RESPONSE_FORMAT_JSON=true
MM_ANALYZER_MAX_TOKENS=16384
MM_ANALYZER_TEMPERATURE=0
MM_ANALYZER_CACHE=true
MM_ANALYZER_CACHE_DIR=.cache/mm_bridge
MM_FAIL_ON_ANALYZER_ERROR=true

MM_VLLM_MEDIA_POLICY=replace
MM_UNSUPPORTED_MEDIA_ENDPOINT_POLICY=error
MM_MODELS_POLICY=alias_only
MM_REWRITE_BRIDGE_MODEL_ONLY=true

MM_MAX_BODY_BYTES=104857600
MM_MAX_MEDIA_BYTES=52428800
MM_MAX_MEDIA_ITEMS=8
MM_HTTP_TIMEOUT_SECONDS=600
MM_STREAM_HEARTBEAT_SECONDS=15

MM_DEBUG_DUMP=false
MM_DEBUG_DUMP_DIR=.debug/mm_bridge
MM_WRAP_UPSTREAM_ERRORS=false
```

## Bridge server

| Variable | Code default | Recommended value | Description |
|---|---:|---:|---|
| `MM_BRIDGE_HOST` | `0.0.0.0` | `127.0.0.1` behind a local reverse proxy | Address used by Uvicorn. `0.0.0.0` listens on all interfaces. |
| `MM_BRIDGE_PORT` | `18000` | `18000` | Bridge HTTP port. |
| `MM_BRIDGE_TOKEN` | empty | A long random token | Client authentication token. An empty value disables bridge authentication. |
| `MM_BRIDGE_MODEL_ID` | `deepseek-v4-mm-bridge` | Any stable client-facing alias | Model ID exposed by the bridge. |
| `MM_BRIDGE_MODE` | `1` | `1` for a separate analyzer and text model | `1` analyzes current-event media before the final request. `2` skips analyzer enrichment and forwards to a multimodal main model. |
| `MM_LOG_LEVEL` | `INFO` | `INFO` | Uvicorn log level. Use `DEBUG` only while diagnosing problems. |

Generate a token locally:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Clients may authenticate with either header:

```http
Authorization: Bearer <MM_BRIDGE_TOKEN>
```

```http
x-api-key: <MM_BRIDGE_TOKEN>
```

## Final text-model upstream

| Variable | Code default | Recommended value | Description |
|---|---:|---:|---|
| `VLLM_ROOT_URL` | `http://127.0.0.1:8000` | Your text-model root URL | Do not append `/v1`. The bridge appends the incoming endpoint path. |
| `VLLM_API_KEY` | `EMPTY` | Match the upstream server | Authentication value sent to the final text-model server. `EMPTY` is only a placeholder. |
| `VLLM_AUTH_STYLE` | `auto` | `auto` or the exact server requirement | Controls which upstream authentication header is sent. |
| `VLLM_MODEL` | empty | Empty for one-model servers, explicit ID for multi-model servers | When empty, the bridge discovers the first model returned by `GET /v1/models`. |

`VLLM_ROOT_URL` falls back to `VLLM_BASE_URL` when the preferred variable is absent.

### Authentication styles

| Value | Behavior |
|---|---|
| `auto` | Uses `x-api-key` for `/v1/messages` and `/v1/messages/count_tokens`; otherwise uses Bearer auth. |
| `bearer` | Sends `Authorization: Bearer <key>`. |
| `x-api-key` | Sends `x-api-key: <key>`. |
| `both` | Sends both headers. |
| `none` | Sends no upstream authentication header. |

## Multimodal-analyzer upstream

The `LLAMA_*` names are retained for compatibility. They may point to llama.cpp or to a multimodal vLLM server.

| Variable | Code default | Recommended value | Description |
|---|---:|---:|---|
| `LLAMA_ROOT_URL` | `http://127.0.0.1:8080` | Your analyzer-server root URL | Do not append `/v1`. |
| `LLAMA_API_KEY` | `EMPTY` | Match the analyzer server | Authentication value sent to the multimodal analyzer. |
| `LLAMA_AUTH_STYLE` | `auto` | `auto` or the exact server requirement | Uses the same allowed values as `VLLM_AUTH_STYLE`. |
| `LLAMA_MODEL` | empty | Empty for one-model servers, explicit ID for multi-model servers | When empty, the bridge discovers the first model returned by `GET /v1/models`. |

Compatibility fallbacks currently supported by the code:

```text
LLAMA_ROOT_URL → JETSON_LLAMA_BASE_URL
LLAMA_API_KEY  → JETSON_LLAMA_API_KEY
LLAMA_MODEL    → JETSON_LLAMA_MODEL
```

## Analyzer and injection

| Variable | Code default | Project recommendation | Description |
|---|---:|---:|---|
| `MM_ANALYZER_ENDPOINTS` | `/v1/chat/completions,/v1/messages,/v1/responses` | Keep the default list | Endpoints on which media analysis is generated and injected. |
| `MM_ANALYZER_FORCE_STREAM_FALSE` | `true` | `true` | Loaded by `Settings`. Analyzer requests are currently constructed with `stream: false`. |
| `MM_ANALYZER_RESPONSE_FORMAT_JSON` | `true` | `true` | Adds `response_format={"type":"json_object"}` for `/v1/chat/completions`. The other analyzer protocols rely on the prompt to request JSON. |
| `MM_ANALYZER_MAX_TOKENS` | `1200` | **At least `16384`** | Maximum analyzer output tokens. Increase for dense OCR, game screens, code, terminal screenshots, and multiple images. |
| `MM_ANALYZER_TEMPERATURE` | `0` | `0` | Low randomness is preferred for OCR and factual visual extraction. |
| `MM_ANALYZER_CACHE` | `true` | `true` in production | Enables file-based analysis caching. |
| `MM_ANALYZER_CACHE_DIR` | `.cache/mm_bridge` | `.cache/mm_bridge` | Cache directory. Falls back to `MM_MEDIA_CACHE_DIR`. |
| `MM_FAIL_ON_ANALYZER_ERROR` | `true` | `true` | Stops the request when visual analysis fails instead of allowing the text model to answer without visual evidence. |

### Token recommendation

Use at least:

```env
MM_ANALYZER_MAX_TOKENS=16384
```

Consider a higher value for long terminal output, source code, documents, or multi-image requests:

```env
MM_ANALYZER_MAX_TOKENS=32768
```

This variable sets a maximum, not a required output length. The analyzer model, its context window, and the serving configuration must support the selected value. A high token limit does not improve unreadable source media by itself.

## Final-model media policy

| Variable | Code default | Recommended value for a text-only final model | Description |
|---|---:|---:|---|
| `MM_VLLM_MEDIA_POLICY` | `keep` | `replace` | Controls whether original media remains in the final text-model request. |

Allowed values:

| Value | Behavior |
|---|---|
| `keep` | Keeps the original media blocks after injecting the analysis. Use only when the final model and server accept media. |
| `replace` | Replaces media blocks with text placeholders and sends the analysis as context. Recommended for text-only final models. |
| `strip` | Removes media blocks completely and sends only the injected analysis. |

If analyzer processing fails while `MM_FAIL_ON_ANALYZER_ERROR=false`, the Bridge does not apply `replace` or `strip` to an empty analysis. It forwards the original raw media to the final upstream and marks the response with `x-mm-bridge-analyzer-source: fail_open`. Only enable this when the final upstream accepts the incoming media format.

## Unsupported media endpoints

| Variable | Code default | Recommended value for strict text-only operation | Description |
|---|---:|---:|---|
| `MM_UNSUPPORTED_MEDIA_ENDPOINT_POLICY` | `passthrough` | `error` | Controls requests that contain media but use an endpoint outside `MM_ANALYZER_ENDPOINTS`. |

Allowed values:

| Value | Behavior |
|---|---|
| `passthrough` | Forwards the request without analyzer enrichment. This may send raw media to the final upstream. |
| `error` | Returns HTTP 501 at the bridge protocol gate. |

## Model list and alias rewrite

| Variable | Code default | Recommended value | Description |
|---|---:|---:|---|
| `MM_MODELS_POLICY` | `alias_plus_upstream` | `alias_only` for ordinary clients | Controls the `/v1/models` response. |
| `MM_REWRITE_BRIDGE_MODEL_ONLY` | `true` | `true` | Rewrites the requested model only when it matches the bridge alias or is absent. |

`MM_MODELS_POLICY` values:

| Value | Behavior |
|---|---|
| `alias_plus_upstream` | Shows the bridge alias and upstream text models. |
| `alias_only` | Shows only the bridge alias. |
| `upstream_only` | Shows only upstream text models. |
| `passthrough` | Proxies `/v1/models` directly to the text-model upstream. |

## Limits and timeouts

| Variable | Code default | Description |
|---|---:|---|
| `MM_MAX_BODY_BYTES` | `104857600` | Maximum complete request-body size in bytes. |
| `MM_MAX_MEDIA_BYTES` | `52428800` | Maximum approximate size of one media item in bytes. |
| `MM_MAX_MEDIA_ITEMS` | `8` | Maximum number of media items per request. |
| `MM_HTTP_TIMEOUT_SECONDS` | `600` | Shared HTTP timeout used for analyzer, final-model, and model-discovery upstream requests. A non-stream timeout is returned as a structured HTTP 504 with its stage and elapsed time unless analyzer fail-open is enabled. |
| `MM_STREAM_HEARTBEAT_SECONDS` | `15` | SSE heartbeat interval. Set to `0` or less to disable heartbeat messages. |

Remember that base64 media is larger than the original binary file. Reverse-proxy limits must also allow the request size and duration.

The timeout remains one shared setting. `x-mm-bridge-stage` distinguishes `llama_analyzer`, `vllm_final`, and model-discovery failures; `x-mm-bridge-analyzer-source` distinguishes a fresh analyzer call from a cache hit or fail-open route. With `MM_FAIL_ON_ANALYZER_ERROR=false`, analyzer and analyzer-model-discovery timeouts/transport failures are absorbed into the raw-media `fail_open` route instead of returning 504/502. Best-effort alias-plus-upstream model listing may also absorb model-list failures.

For long-running final-model generations, raise the Bridge timeout as needed and set the client timeout to the same or a larger value. This timeout controls waiting, not generation length; configure the output-token ceiling separately in the calling client.

## PDF preprocessing

PDF handling is disabled by default. When enabled, MM-Bridge extracts native
text locally, page by page; only pages below `MM_PDF_MIN_NATIVE_TEXT_CHARS` are
rasterized. The OCR provider boundary accepts rendered page images only, never
raw PDF bytes. Evidence preserves `page_number` and keeps `native_text`,
`ocr_text`, and `interpretation` in separate fields.

The deployment must set and test every limit before enabling PDFs:

```env
MM_PDF_ENABLED=false
MM_PDF_MAX_BYTES=25000000
MM_PDF_MAX_PAGES=100
MM_PDF_MAX_RENDERED_PIXELS=100000000
MM_PDF_MIN_NATIVE_TEXT_CHARS=40
MM_PDF_RENDER_DPI=144
MM_PDF_TIMEOUT_SECONDS=120
MM_PDF_OCR_TIMEOUT_SECONDS=60
MM_PDF_MAX_CONCURRENCY=1
MM_PDF_MAX_QUEUE=2
MM_PDF_OCR_PROVIDER=disabled
MM_PDF_OCR_URL=
MM_PDF_OCR_API_KEY=
MM_PDF_OCR_MODEL=
```

`qwen_vision` targets an OpenAI-compatible vision server. `http` targets a
backend-neutral OCR service such as bounded Tika/Tesseract. Provider hosts and
deployment limits are configuration, not source constants.

Limits are checked before OCR: decoded size, page count, cumulative rendered
pixels, bounded queue/concurrency, then provider timeout. Provider payloads
contain only the page number, MIME type, and base64 rendered page image.

## Debugging

| Variable | Code default | Production recommendation | Description |
|---|---:|---:|---|
| `MM_DEBUG_DUMP` | `false` | `false` | Writes intermediate requests and responses to disk. These files may contain prompts, base64 media, generated analysis, and internal URLs. |
| `MM_DEBUG_DUMP_DIR` | `.debug/mm_bridge` | `.debug/mm_bridge` | Debug-output directory. |
| `MM_WRAP_UPSTREAM_ERRORS` | `false` | `false` | When enabled, error responses may include the upstream URL and response body. |

Temporary debugging configuration:

```env
MM_DEBUG_DUMP=true
MM_ANALYZER_CACHE=false
MM_WRAP_UPSTREAM_ERRORS=true
```

Restore production-safe values after testing:

```env
MM_DEBUG_DUMP=false
MM_ANALYZER_CACHE=true
MM_WRAP_UPSTREAM_ERRORS=false
```

## Sensitive values that must not be committed

Never publish real values for:

```text
MM_BRIDGE_TOKEN
VLLM_API_KEY
LLAMA_API_KEY
VLLM_ROOT_URL when it contains a private hostname or private IP
LLAMA_ROOT_URL when it contains a private hostname or private IP
```

Also keep `.cache/` and `.debug/` out of Git because they may contain private user data and model traffic.
