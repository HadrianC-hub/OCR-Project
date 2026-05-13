"""
apps/search/service.py — Singleton del VectorStore.

El store vive en MEDIA_ROOT/search_index/store.* y se carga la primera
vez que alguien llama get_store(). Es thread-safe.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

from apps.search.store import VectorStore
from apps.search import encoder

logger = logging.getLogger(__name__)


_store: Optional[VectorStore] = None
_lock = threading.Lock()


def _get_store_path() -> Path:
    """Ruta base del store. Configurable vía settings.SEARCH_STORE_PATH."""
    try:
        from django.conf import settings  # noqa: PLC0415
        override = getattr(settings, "SEARCH_STORE_PATH", None)
        if override:
            return Path(override)
        return Path(settings.MEDIA_ROOT) / "search_index" / "store"
    except Exception:
        return Path("media/search_index/store")


def get_store() -> VectorStore:
    """
    Devuelve el VectorStore singleton, cargándolo del disco la primera
    vez. Subsiguientes llamadas reciben la misma instancia.
    """
    global _store
    if _store is not None:
        return _store
    with _lock:
        if _store is not None:
            return _store
        path = _get_store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        _store = VectorStore(path, model_signature=encoder.model_signature())
        try:
            _store.load()
        except Exception as exc:
            logger.warning("No se pudo cargar el store: %s", exc)
        return _store


def reset_store_singleton() -> None:
    """Para tests / reindex desde cero."""
    global _store
    with _lock:
        _store = None
