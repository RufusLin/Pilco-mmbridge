from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from .media import MediaItem


SUPPORTED_MEDIA_TYPES = {"image", "video", "audio"}
SOURCE_VALUES = {"", "ocr", "speech", "lyrics"}
BASIS_VALUES = {"", "observed", "inferred"}
CONFIDENCE_VALUES = {"", "high", "medium", "low"}

TEXT_BLOCK_FIELDS = {"source", "text", "location", "confidence", "uncertainty"}
VISUAL_BLOCK_FIELDS = {
    "subject",
    "description",
    "location",
    "relationships",
    "basis",
    "confidence",
    "uncertainty",
}
AUDIO_BLOCK_FIELDS = {
    "description",
    "location",
    "confidence",
    "uncertainty",
}
MUSICAL_FEATURE_FIELDS = {
    "style_or_character",
    "tonal_center",
    "harmony",
    "melody",
    "rhythm_meter_tempo",
    "form",
    "instrumentation",
}


@dataclass(frozen=True)
class AnalysisResult:
    analysis_text: str
    payload: dict[str, Any]
    parsed_payload: dict[str, Any]
    repairs: list[dict[str, Any]]


class _InvalidBlock(ValueError):
    pass


def process_analysis_text(
    analysis_text: str,
    expected_media: list[MediaItem],
) -> AnalysisResult:
    _validate_expected_media(expected_media)
    try:
        parsed = json.loads(analysis_text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("analyzer output is not valid JSON") from exc

    if not isinstance(parsed, dict):
        raise ValueError("analyzer output must be a JSON object")
    media = parsed.get("media")
    if not isinstance(media, list):
        raise ValueError('analyzer output "media" must be an array')
    single_attachment_split = len(expected_media) == 1 and len(media) > 1
    if len(media) != len(expected_media) and not single_attachment_split:
        raise ValueError(
            f"analyzer output media count {len(media)} "
            f"does not match input count {len(expected_media)}"
        )

    repairs: list[dict[str, Any]] = []
    _record_unknown_fields(parsed, {"media"}, "$", repairs)

    if len(expected_media) == 1:
        expected = expected_media[0]
        normalized_entries = [
            _normalize_media_item(raw_item, expected, position, repairs)
            for position, raw_item in enumerate(media)
        ]
        if len(media) != 1:
            _add_repair(
                repairs,
                "media",
                "merged_single_attachment_entries",
                analyzer_entry_count=len(media),
                input_attachment_count=1,
            )
        normalized_media = [
            _merge_single_attachment_entries(normalized_entries, expected)
        ]
    else:
        normalized_media = [
            _normalize_media_item(raw_item, expected, position, repairs)
            for position, (raw_item, expected) in enumerate(
                zip(media, expected_media, strict=True)
            )
        ]

    payload = {"media": normalized_media}
    validate_normalized_payload(payload, expected_media)
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return AnalysisResult(
        analysis_text=canonical,
        payload=payload,
        parsed_payload=parsed,
        repairs=repairs,
    )


def _merge_single_attachment_entries(
    entries: list[dict[str, Any]],
    expected: MediaItem,
) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "index": expected.index,
        "type": expected.kind,
        "text_blocks": [],
    }
    text_blocks: list[dict[str, Any]] = []
    optional_names = {
        "image": ("visual_blocks",),
        "audio": ("audio_blocks",),
        "video": ("visual_blocks", "audio_blocks"),
    }[expected.kind]
    optional_blocks: dict[str, list[dict[str, Any]]] = {
        name: [] for name in optional_names
    }
    optional_present = {name: False for name in optional_names}

    for entry in entries:
        text_blocks.extend(entry["text_blocks"])
        for optional_name in optional_names:
            if optional_name in entry:
                optional_present[optional_name] = True
                optional_blocks[optional_name].extend(entry[optional_name])

    merged["text_blocks"] = _deduplicate_blocks(text_blocks)
    for optional_name in optional_names:
        if optional_present[optional_name]:
            merged[optional_name] = _deduplicate_blocks(
                optional_blocks[optional_name]
            )
    return merged


def _deduplicate_blocks(
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for block in blocks:
        key = json.dumps(
            block,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(block)
    return unique


def _validate_expected_media(expected_media: list[MediaItem]) -> None:
    for position, expected in enumerate(expected_media):
        if expected.kind not in SUPPORTED_MEDIA_TYPES:
            raise ValueError(
                f"input media[{position}] type {expected.kind!r} is not supported"
            )


def _normalize_media_item(
    raw_item: Any,
    expected: MediaItem,
    position: int,
    repairs: list[dict[str, Any]],
) -> dict[str, Any]:
    path = f"media[{position}]"
    if not isinstance(raw_item, dict):
        raise ValueError(f"{path} must be an object")

    allowed_item_fields = {
        "index",
        "type",
        "text_blocks",
        "visual_blocks",
        "audio_blocks",
    }
    _record_unknown_fields(raw_item, allowed_item_fields, path, repairs)

    if raw_item.get("index") != expected.index:
        _add_repair(
            repairs,
            f"{path}.index",
            "overrode_index_from_input",
            old=raw_item.get("index"),
            new=expected.index,
        )
    if raw_item.get("type") != expected.kind:
        _add_repair(
            repairs,
            f"{path}.type",
            "overrode_type_from_input",
            old=raw_item.get("type"),
            new=expected.kind,
        )

    normalized: dict[str, Any] = {
        "index": expected.index,
        "type": expected.kind,
        "text_blocks": _normalize_block_container(
            raw_item.get("text_blocks"),
            present="text_blocks" in raw_item,
            path=f"{path}.text_blocks",
            repairs=repairs,
            normalizer=_normalize_text_block,
            required=True,
        ),
    }

    if expected.kind in {"image", "video"}:
        visual_blocks = _normalize_block_container(
            raw_item.get("visual_blocks"),
            present="visual_blocks" in raw_item,
            path=f"{path}.visual_blocks",
            repairs=repairs,
            normalizer=_normalize_visual_block,
            required=False,
        )
        if visual_blocks is not None:
            normalized["visual_blocks"] = visual_blocks
        if expected.kind == "image" and "audio_blocks" in raw_item:
            _add_repair(
                repairs,
                f"{path}.audio_blocks",
                "removed_disallowed_container",
                input_media_type=expected.kind,
            )
    if expected.kind in {"audio", "video"}:
        audio_blocks = _normalize_block_container(
            raw_item.get("audio_blocks"),
            present="audio_blocks" in raw_item,
            path=f"{path}.audio_blocks",
            repairs=repairs,
            normalizer=_normalize_audio_block,
            required=False,
        )
        if audio_blocks is not None:
            normalized["audio_blocks"] = audio_blocks
        if expected.kind == "audio" and "visual_blocks" in raw_item:
            _add_repair(
                repairs,
                f"{path}.visual_blocks",
                "removed_disallowed_container",
                input_media_type=expected.kind,
            )

    return normalized


BlockNormalizer = Callable[
    [Any, str, list[dict[str, Any]]],
    dict[str, Any] | None,
]


def _normalize_block_container(
    value: Any,
    *,
    present: bool,
    path: str,
    repairs: list[dict[str, Any]],
    normalizer: BlockNormalizer,
    required: bool,
) -> list[dict[str, Any]] | None:
    if not present or value is None:
        if required:
            _add_repair(repairs, path, "inserted_missing_array")
            return []
        if present:
            _add_repair(repairs, path, "removed_null_optional_container")
        return None
    if not isinstance(value, list):
        _add_repair(repairs, path, "replaced_invalid_array")
        return [] if required else None

    blocks: list[dict[str, Any]] = []
    for block_index, block in enumerate(value):
        block_path = f"{path}[{block_index}]"
        normalized = normalizer(block, block_path, repairs)
        if normalized is not None:
            blocks.append(normalized)
    return blocks


def _normalize_text_block(
    block: Any,
    path: str,
    repairs: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not isinstance(block, dict):
        _discard_block(repairs, path, "block is not an object")
        return None
    _record_unknown_fields(block, TEXT_BLOCK_FIELDS, path, repairs)
    try:
        return {
            "source": _normalize_string(
                block, "source", path, repairs, enum_values=SOURCE_VALUES
            ),
            "text": _normalize_string(block, "text", path, repairs),
            "location": _normalize_string(block, "location", path, repairs),
            "confidence": _normalize_string(
                block,
                "confidence",
                path,
                repairs,
                enum_values=CONFIDENCE_VALUES,
            ),
            "uncertainty": _normalize_string(
                block, "uncertainty", path, repairs
            ),
        }
    except _InvalidBlock as exc:
        _discard_block(repairs, path, str(exc))
        return None


def _normalize_visual_block(
    block: Any,
    path: str,
    repairs: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not isinstance(block, dict):
        _discard_block(repairs, path, "block is not an object")
        return None
    _record_unknown_fields(block, VISUAL_BLOCK_FIELDS, path, repairs)
    try:
        return {
            "subject": _normalize_string(block, "subject", path, repairs),
            "description": _normalize_string(
                block, "description", path, repairs
            ),
            "location": _normalize_string(block, "location", path, repairs),
            "relationships": _normalize_relationships(
                block, "relationships", path, repairs
            ),
            "basis": _normalize_string(
                block, "basis", path, repairs, enum_values=BASIS_VALUES
            ),
            "confidence": _normalize_string(
                block,
                "confidence",
                path,
                repairs,
                enum_values=CONFIDENCE_VALUES,
            ),
            "uncertainty": _normalize_string(
                block, "uncertainty", path, repairs
            ),
        }
    except _InvalidBlock as exc:
        _discard_block(repairs, path, str(exc))
        return None


def _normalize_audio_block(
    block: Any,
    path: str,
    repairs: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not isinstance(block, dict):
        _discard_block(repairs, path, "block is not an object")
        return None
    allowed = AUDIO_BLOCK_FIELDS | {"musical_features"}
    _record_unknown_fields(block, allowed, path, repairs)
    try:
        normalized: dict[str, Any] = {
            "description": _normalize_string(
                block, "description", path, repairs
            ),
            "location": _normalize_string(block, "location", path, repairs),
            "confidence": _normalize_string(
                block,
                "confidence",
                path,
                repairs,
                enum_values=CONFIDENCE_VALUES,
            ),
            "uncertainty": _normalize_string(
                block, "uncertainty", path, repairs
            ),
        }
    except _InvalidBlock as exc:
        _discard_block(repairs, path, str(exc))
        return None

    if "musical_features" in block:
        features = block.get("musical_features")
        if isinstance(features, dict):
            _record_unknown_fields(
                features,
                MUSICAL_FEATURE_FIELDS,
                f"{path}.musical_features",
                repairs,
            )
            try:
                normalized["musical_features"] = {
                    field: _normalize_string(
                        features,
                        field,
                        f"{path}.musical_features",
                        repairs,
                    )
                    for field in sorted(MUSICAL_FEATURE_FIELDS)
                }
            except _InvalidBlock:
                _add_repair(
                    repairs,
                    f"{path}.musical_features",
                    "removed_invalid_optional_container",
                )
        else:
            _add_repair(
                repairs,
                f"{path}.musical_features",
                "removed_invalid_optional_container",
            )
    return normalized


def _normalize_string(
    block: dict[str, Any],
    field: str,
    path: str,
    repairs: list[dict[str, Any]],
    *,
    enum_values: set[str] | None = None,
) -> str:
    value = block.get(field)
    field_path = f"{path}.{field}"
    if field not in block or value is None:
        _add_repair(repairs, field_path, "inserted_missing_string")
        return ""
    if not isinstance(value, str):
        raise _InvalidBlock(f"{field_path} must be a string")
    if enum_values is not None and value not in enum_values:
        _add_repair(
            repairs,
            field_path,
            "replaced_unsupported_enum",
            old=value,
            new="",
        )
        return ""
    return value


def _normalize_relationships(
    block: dict[str, Any],
    field: str,
    path: str,
    repairs: list[dict[str, Any]],
) -> list[str]:
    value = block.get(field)
    field_path = f"{path}.{field}"
    if field not in block or value is None:
        _add_repair(repairs, field_path, "inserted_missing_array")
        return []
    if isinstance(value, str):
        _add_repair(repairs, field_path, "wrapped_string_in_array")
        return [value]
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise _InvalidBlock(f"{field_path} must be an array of strings")
    return list(value)


def _record_unknown_fields(
    value: dict[str, Any],
    allowed: set[str],
    path: str,
    repairs: list[dict[str, Any]],
) -> None:
    for field in sorted(set(value) - allowed):
        field_path = f"{path}.{field}" if path != "$" else f"$.{field}"
        _add_repair(
            repairs,
            field_path,
            "removed_unknown_field",
            field=field,
        )


def _discard_block(
    repairs: list[dict[str, Any]],
    path: str,
    reason: str,
) -> None:
    _add_repair(
        repairs,
        path,
        "discarded_invalid_block",
        reason=reason,
    )


def _add_repair(
    repairs: list[dict[str, Any]],
    path: str,
    action: str,
    **details: Any,
) -> None:
    repair = {"path": path, "action": action}
    repair.update(details)
    repairs.append(repair)


def validate_normalized_payload(
    payload: dict[str, Any],
    expected_media: list[MediaItem],
) -> None:
    _validate_expected_media(expected_media)
    if not isinstance(payload, dict) or set(payload) != {"media"}:
        raise ValueError('analyzer output must contain only the top-level "media" field')
    media = payload.get("media")
    if not isinstance(media, list):
        raise ValueError('analyzer output "media" must be an array')
    if len(media) != len(expected_media):
        raise ValueError(
            f"analyzer output media count {len(media)} "
            f"does not match input count {len(expected_media)}"
        )

    for position, (item, expected) in enumerate(
        zip(media, expected_media, strict=True)
    ):
        _validate_media_item(item, expected, position)


def _validate_media_item(item: Any, expected: MediaItem, position: int) -> None:
    path = f"media[{position}]"
    if not isinstance(item, dict):
        raise ValueError(f"{path} must be an object")
    allowed = {"index", "type", "text_blocks"}
    if expected.kind in {"image", "video"}:
        allowed.add("visual_blocks")
    if expected.kind in {"audio", "video"}:
        allowed.add("audio_blocks")
    if set(item) - allowed:
        raise ValueError(f"{path} has unsupported fields")
    if item.get("index") != expected.index:
        raise ValueError(f"{path}.index must be {expected.index}")
    if item.get("type") != expected.kind:
        raise ValueError(f"{path}.type must be {expected.kind}")

    text_blocks = item.get("text_blocks")
    if not isinstance(text_blocks, list):
        raise ValueError(f"{path}.text_blocks must be an array")
    for block_index, block in enumerate(text_blocks):
        _validate_text_block(block, f"{path}.text_blocks[{block_index}]")

    visual_blocks = item.get("visual_blocks")
    if visual_blocks is not None:
        if not isinstance(visual_blocks, list):
            raise ValueError(f"{path}.visual_blocks must be an array")
        for block_index, block in enumerate(visual_blocks):
            _validate_visual_block(block, f"{path}.visual_blocks[{block_index}]")

    audio_blocks = item.get("audio_blocks")
    if audio_blocks is not None:
        if not isinstance(audio_blocks, list):
            raise ValueError(f"{path}.audio_blocks must be an array")
        for block_index, block in enumerate(audio_blocks):
            _validate_audio_block(block, f"{path}.audio_blocks[{block_index}]")


def _validate_text_block(block: Any, path: str) -> None:
    _validate_exact_object(block, TEXT_BLOCK_FIELDS, path)
    _validate_string_fields(block, TEXT_BLOCK_FIELDS, path)
    if block["source"] not in SOURCE_VALUES:
        raise ValueError(f"{path}.source is invalid")
    if block["confidence"] not in CONFIDENCE_VALUES:
        raise ValueError(f"{path}.confidence is invalid")


def _validate_visual_block(block: Any, path: str) -> None:
    _validate_exact_object(block, VISUAL_BLOCK_FIELDS, path)
    _validate_string_fields(block, VISUAL_BLOCK_FIELDS - {"relationships"}, path)
    relationships = block["relationships"]
    if not isinstance(relationships, list) or not all(
        isinstance(item, str) for item in relationships
    ):
        raise ValueError(f"{path}.relationships must be an array of strings")
    if block["basis"] not in BASIS_VALUES:
        raise ValueError(f"{path}.basis is invalid")
    if block["confidence"] not in CONFIDENCE_VALUES:
        raise ValueError(f"{path}.confidence is invalid")


def _validate_audio_block(block: Any, path: str) -> None:
    if not isinstance(block, dict):
        raise ValueError(f"{path} must be an object")
    allowed = AUDIO_BLOCK_FIELDS | {"musical_features"}
    if not AUDIO_BLOCK_FIELDS.issubset(block) or set(block) - allowed:
        raise ValueError(f"{path} has invalid fields")
    _validate_string_fields(block, AUDIO_BLOCK_FIELDS, path)
    if block["confidence"] not in CONFIDENCE_VALUES:
        raise ValueError(f"{path}.confidence is invalid")
    if "musical_features" in block:
        features = block["musical_features"]
        _validate_exact_object(
            features,
            MUSICAL_FEATURE_FIELDS,
            f"{path}.musical_features",
        )
        _validate_string_fields(
            features,
            MUSICAL_FEATURE_FIELDS,
            f"{path}.musical_features",
        )


def _validate_exact_object(value: Any, fields: set[str], path: str) -> None:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{path} has invalid fields")


def _validate_string_fields(
    value: dict[str, Any],
    fields: set[str],
    path: str,
) -> None:
    for field in fields:
        if not isinstance(value.get(field), str):
            raise ValueError(f"{path}.{field} must be a string")
