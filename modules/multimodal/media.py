from __future__ import annotations

import base64
import copy
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

DATA_URI_RE = re.compile(r"^data:(image|audio|video)/[a-zA-Z0-9.+-]+;base64,", re.IGNORECASE)
REMOTE_MEDIA_RE = re.compile(r"^https?://.*\.(png|jpe?g|webp|gif|bmp|mp3|wav|flac|mp4|mov|mkv|webm)(\?.*)?$", re.IGNORECASE)
LOCAL_MEDIA_RE = re.compile(r"^file://.*\.(png|jpe?g|webp|gif|bmp|mp3|wav|flac|mp4|mov|mkv|webm)$", re.IGNORECASE)
RAW_B64_RE = re.compile(r"^[A-Za-z0-9+/=\r\n]{200,}$")


@dataclass(frozen=True)
class MediaItem:
    index: int
    kind: str
    path: str
    block: Any
    approx_bytes: int
    hash: str


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_text(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _estimate_b64_size(data: str) -> int:
    comma = data.find(",")
    if comma >= 0 and data[:comma].lower().startswith("data:"):
        data = data[comma + 1 :]
    stripped = re.sub(r"\s+", "", data)
    if not stripped:
        return 0
    padding = stripped.count("=")
    return max(0, (len(stripped) * 3 // 4) - padding)


def _size_of_url_or_b64(value: str) -> int:
    if DATA_URI_RE.match(value):
        return _estimate_b64_size(value)
    if RAW_B64_RE.match(value) and not value.startswith(("http://", "https://", "file://")):
        # Only an estimate. We do not decode here.
        return _estimate_b64_size(value)
    return len(value.encode("utf-8"))


def _detect_kind_from_data_uri(value: str) -> str | None:
    m = DATA_URI_RE.match(value)
    if m:
        return m.group(1).lower()
    return None


def _detect_kind_from_url(value: str) -> str | None:
    lower = value.lower()
    if lower.startswith("data:"):
        return _detect_kind_from_data_uri(value)
    if lower.startswith(("http://", "https://", "file://")):
        if re.search(r"\.(png|jpe?g|webp|gif|bmp)(\?.*)?$", lower):
            return "image"
        if re.search(r"\.(mp3|wav|flac)(\?.*)?$", lower):
            return "audio"
        if re.search(r"\.(mp4|mov|mkv|webm)(\?.*)?$", lower):
            return "video"
    return None


def _is_openai_media_block(obj: dict[str, Any]) -> tuple[bool, str | None]:
    t = obj.get("type")
    if not isinstance(t, str):
        return False, ""
    if t == "image_url" and isinstance(obj.get("image_url"), dict):
        return True, "image"
    if t == "input_audio" and isinstance(obj.get("input_audio"), dict):
        return True, "audio"
    if t == "input_video" and isinstance(obj.get("input_video"), dict):
        return True, "video"
    # Responses API common naming.
    if t == "input_image":
        return True, "image"
    if t == "input_audio":
        return True, "audio"
    if t == "input_video":
        return True, "video"
    return False, None


def _is_anthropic_media_block(obj: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, ""

    t = obj.get("type")

    # Claude Code 요청 안에는 type이 dict/list인 구조도 올 수 있음.
    # 미디어 block type은 반드시 문자열이어야 함.
    if not isinstance(t, str):
        return False, ""

    if t in {"image", "audio", "video"} and isinstance(obj.get("source"), dict):
        return True, t

    return False, ""

def _approx_block_bytes(block: Any) -> int:
    if isinstance(block, dict):
        # Official OpenAI Chat multimedia fields.
        if block.get("type") == "image_url" and isinstance(block.get("image_url"), dict):
            url = block["image_url"].get("url")
            if isinstance(url, str):
                return _size_of_url_or_b64(url)
        if block.get("type") == "input_audio" and isinstance(block.get("input_audio"), dict):
            src = block["input_audio"].get("data") or block["input_audio"].get("url")
            if isinstance(src, str):
                return _size_of_url_or_b64(src)
        if block.get("type") == "input_video" and isinstance(block.get("input_video"), dict):
            src = block["input_video"].get("data") or block["input_video"].get("url")
            if isinstance(src, str):
                return _size_of_url_or_b64(src)
        # Anthropic source block.
        source = block.get("source")
        if isinstance(source, dict):
            src = source.get("data") or source.get("url")
            if isinstance(src, str):
                return _size_of_url_or_b64(src)
        # Responses-style blocks.
        for key in ("image_url", "audio_url", "video_url", "data", "url"):
            val = block.get(key)
            if isinstance(val, str):
                kind = _detect_kind_from_url(val)
                if kind:
                    return _size_of_url_or_b64(val)
    if isinstance(block, str):
        if _detect_kind_from_url(block):
            return _size_of_url_or_b64(block)
    return len(canonical_json(block))


def find_media_items(value: Any) -> list[MediaItem]:
    items: list[MediaItem] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            is_mm, kind = _is_openai_media_block(node)
            if not is_mm:
                is_mm, kind = _is_anthropic_media_block(node)
            if is_mm:
                block = copy.deepcopy(node)
                data = canonical_json(block)
                items.append(
                    MediaItem(
                        index=len(items) + 1,
                        kind=kind or "media",
                        path=path,
                        block=block,
                        approx_bytes=_approx_block_bytes(block),
                        hash=sha256_text(data),
                    )
                )
                return
            for key, child in node.items():
                walk(child, f"{path}.{key}" if path else str(key))
        elif isinstance(node, list):
            for i, child in enumerate(node):
                walk(child, f"{path}[{i}]")
        elif isinstance(node, str):
            kind = _detect_kind_from_url(node)
            if kind:
                data = node.encode("utf-8")
                items.append(
                    MediaItem(
                        index=len(items) + 1,
                        kind=kind,
                        path=path,
                        block=node,
                        approx_bytes=_size_of_url_or_b64(node),
                        hash=sha256_text(data),
                    )
                )

    walk(value, "")
    return items


def media_summary(items: list[MediaItem]) -> str:
    lines = []
    for item in items:
        lines.append(
            f"#{item.index}: kind={item.kind}, path={item.path or '$'}, approx_bytes={item.approx_bytes}, sha256={item.hash[:16]}"
        )
    return "\n".join(lines)


def replace_or_strip_media_blocks(value: Any, policy: str, placeholder_prefix: str = "attached media") -> Any:
    if policy == "keep":
        return value
    counter = {"n": 0}

    def transform(node: Any) -> Any:
        if isinstance(node, dict):
            is_mm, kind = _is_openai_media_block(node)
            if not is_mm:
                is_mm, kind = _is_anthropic_media_block(node)
            if is_mm:
                counter["n"] += 1
                label = f"[{placeholder_prefix} #{counter['n']} ({kind or 'media'}): analysis is included in the system/instructions context]"
                if policy == "strip":
                    return None
                # Use official text blocks for the dominant protocol shapes.
                block_type = node.get("type")
                if block_type in {"image_url", "input_audio", "input_video"}:
                    return {"type": "text", "text": label}
                if block_type in {"image", "audio", "video"}:
                    return {"type": "text", "text": label}
                if block_type and str(block_type).startswith("input_"):
                    return {"type": "input_text", "text": label}
                return label
            out: dict[str, Any] = {}
            for k, v in node.items():
                tv = transform(v)
                if tv is not None:
                    out[k] = tv
            return out
        if isinstance(node, list):
            out_list = []
            for v in node:
                tv = transform(v)
                if tv is not None:
                    out_list.append(tv)
            return out_list
        return node

    return transform(value)
