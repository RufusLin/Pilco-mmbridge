from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .media import MediaItem, sha256_text
from .prompting import ANALYZER_PROMPT_VERSION


def make_cache_key(endpoint: str, model: str, items: list[MediaItem], user_request: str = "") -> str:
    data = {
        "version": ANALYZER_PROMPT_VERSION,
        "endpoint": endpoint,
        "model": model,
        "media_hashes": [item.hash for item in items],
        "user_request": user_request,
    }
    return sha256_text(json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8"))


class AnalysisCache:
    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> str | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            text = data.get("analysis")
            return text if isinstance(text, str) else None
        except Exception:
            return None

    def set(self, key: str, analysis: str, metadata: dict[str, Any] | None = None) -> None:
        path = self._path(key)
        tmp = path.with_suffix(".tmp")
        data = {"analysis": analysis, "metadata": metadata or {}}
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
