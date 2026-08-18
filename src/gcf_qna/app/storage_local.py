"""Local-disk storage client for chainlit's data layer.

Persisted elements (our highlighted evidence pages) are copied under
public/app_files/, which chainlit serves natively at /public — so resumed
threads render their images across restarts with no extra server code.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Union

from chainlit.data.storage_clients.base import BaseStorageClient

from gcf_qna import config

_SAFE = re.compile(r"[^A-Za-z0-9._@-]+")


def _safe_key(object_key: str) -> str:
    """Path-traversal-proof relative key: sanitize each segment, drop empties."""
    parts = [_SAFE.sub("_", p) for p in object_key.split("/") if p not in ("", ".", "..")]
    return "/".join(parts) or "unnamed"


class LocalStorageClient(BaseStorageClient):
    def __init__(self, base_dir: Path | None = None, url_prefix: str = "/public/app_files"):
        self.base_dir = Path(base_dir or config.PUBLIC_DIR / "app_files")
        self.url_prefix = url_prefix.rstrip("/")

    def _path(self, object_key: str) -> Path:
        return self.base_dir / _safe_key(object_key)

    async def upload_file(
        self,
        object_key: str,
        data: Union[bytes, str],
        mime: str = "application/octet-stream",
        overwrite: bool = True,
        content_disposition: str | None = None,
    ) -> Dict[str, Any]:
        path = self._path(object_key)
        if path.exists() and not overwrite:
            return {"object_key": _safe_key(object_key),
                    "url": f"{self.url_prefix}/{_safe_key(object_key)}"}
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = data.encode("utf-8") if isinstance(data, str) else data
        path.write_bytes(payload)
        key = _safe_key(object_key)
        return {"object_key": key, "url": f"{self.url_prefix}/{key}"}

    async def delete_file(self, object_key: str) -> bool:
        path = self._path(object_key)
        try:
            path.unlink(missing_ok=True)
            return True
        except OSError:
            return False

    async def get_read_url(self, object_key: str) -> str:
        return f"{self.url_prefix}/{_safe_key(object_key)}"

    async def close(self) -> None:
        return None
