"""Content-addressed on-disk cache for synthesized audio.

Key = sha256(model | voice | format | settings | normalized text), so the same
question spoken with the same voice is generated once and replayed for free.
Writes are atomic (temp file + os.replace) so concurrent prefetch workers never
expose a half-written clip, and the cache is trimmed LRU-style by access time.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path


class AudioCache:
    def __init__(self, root: Path, max_mb: int = 200) -> None:
        self.root = Path(root)
        self.max_bytes = max(1, max_mb) * 1024 * 1024
        self._lock = threading.Lock()

    @staticmethod
    def key(text: str, *, model: str, voice_id: str, output_format: str, settings: dict) -> str:
        material = json.dumps(
            {
                "t": " ".join(text.split()),
                "m": model,
                "v": voice_id,
                "f": output_format,
                "s": settings,
            },
            sort_keys=True,
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        # Two-level fan-out keeps directories small.
        return self.root / key[:2] / f"{key}.mp3"

    def get(self, key: str) -> bytes | None:
        path = self._path(key)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return None
        try:
            os.utime(path)  # bump access time for LRU eviction
        except OSError:
            pass
        return data or None

    def put(self, key: str, data: bytes) -> None:
        if not data:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".part")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        with self._lock:
            files = [p for p in self.root.rglob("*.mp3") if p.is_file()]
            total = sum(p.stat().st_size for p in files)
            if total <= self.max_bytes:
                return
            for p in sorted(files, key=lambda f: f.stat().st_mtime):
                try:
                    size = p.stat().st_size
                    p.unlink()
                    total -= size
                except OSError:
                    continue
                if total <= self.max_bytes * 0.9:
                    break
