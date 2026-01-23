# dreamsApp/analytics/persistence.py

import json
import os
import tempfile
from pathlib import Path
from typing import Optional, Callable, TypeVar, Dict, Any

from .serialization import SerializedPayload


__all__ = [
    'ContentAddressedStore',
    'StructuralCache',
]


T = TypeVar('T')


class ContentAddressedStore:
    
    def __init__(self, base_path: Path) -> None:
        if not isinstance(base_path, Path):
            raise TypeError(f"base_path must be a Path, got {type(base_path).__name__}")
        self._base_path = base_path
        self._base_path.mkdir(parents=True, exist_ok=True)
    
    def _key_to_path(self, fingerprint: str) -> Path:
        return self._base_path / f"{fingerprint}.json"
    
    def store(self, payload: SerializedPayload) -> str:
        if not isinstance(payload, SerializedPayload):
            raise TypeError(f"payload must be SerializedPayload, got {type(payload).__name__}")
        
        target_path = self._key_to_path(payload.fingerprint)
        json_data = payload.to_json()
        
        fd, temp_path = tempfile.mkstemp(
            suffix='.tmp',
            prefix='cas_',
            dir=str(self._base_path)
        )
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(json_data)
            os.replace(temp_path, target_path)
        except Exception:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
        
        return payload.fingerprint
    
    def load(self, fingerprint: str) -> Optional[SerializedPayload]:
        if not isinstance(fingerprint, str):
            raise TypeError(f"fingerprint must be a string, got {type(fingerprint).__name__}")
        
        target_path = self._key_to_path(fingerprint)
        if not target_path.exists():
            return None
        
        with open(target_path, 'r', encoding='utf-8') as f:
            json_str = f.read()
        
        return SerializedPayload.from_json(json_str)
    
    def exists(self, fingerprint: str) -> bool:
        if not isinstance(fingerprint, str):
            raise TypeError(f"fingerprint must be a string, got {type(fingerprint).__name__}")
        return self._key_to_path(fingerprint).exists()
    
    def invalidate(self, fingerprint: str) -> bool:
        if not isinstance(fingerprint, str):
            raise TypeError(f"fingerprint must be a string, got {type(fingerprint).__name__}")
        
        target_path = self._key_to_path(fingerprint)
        if target_path.exists():
            target_path.unlink()
            return True
        return False


class StructuralCache:
    
    def __init__(self, store: ContentAddressedStore) -> None:
        if not isinstance(store, ContentAddressedStore):
            raise TypeError(f"store must be ContentAddressedStore, got {type(store).__name__}")
        self._store = store
        self._memory_cache: Dict[str, SerializedPayload] = {}
    
    def get(self, fingerprint: str) -> Optional[SerializedPayload]:
        if fingerprint in self._memory_cache:
            return self._memory_cache[fingerprint]
        
        payload = self._store.load(fingerprint)
        if payload is not None:
            self._memory_cache[fingerprint] = payload
        return payload
    
    def put(self, payload: SerializedPayload) -> str:
        fingerprint = self._store.store(payload)
        self._memory_cache[fingerprint] = payload
        return fingerprint
    
    def get_or_compute(
        self,
        fingerprint: str,
        compute_fn: Callable[[], SerializedPayload]
    ) -> SerializedPayload:
        cached = self.get(fingerprint)
        if cached is not None:
            return cached
        
        payload = compute_fn()
        self.put(payload)
        return payload
    
    def is_valid(self, fingerprint: str) -> bool:
        return fingerprint in self._memory_cache or self._store.exists(fingerprint)
    
    def invalidate(self, fingerprint: str) -> bool:
        self._memory_cache.pop(fingerprint, None)
        return self._store.invalidate(fingerprint)
    
    def clear_memory_cache(self) -> None:
        self._memory_cache.clear()
