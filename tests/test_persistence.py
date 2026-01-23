# tests/test_persistence.py

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from dreamsApp.analytics.emotion_timeline import EmotionEvent, EmotionTimeline
from dreamsApp.analytics.serialization import EmotionTimelineSerializer, SerializedPayload
from dreamsApp.analytics.persistence import ContentAddressedStore, StructuralCache


@pytest.fixture
def base_time() -> datetime:
    return datetime(2024, 1, 1, 12, 0, 0)


@pytest.fixture
def temp_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield ContentAddressedStore(Path(tmpdir))


@pytest.fixture
def temp_cache(temp_store):
    return StructuralCache(temp_store)


class TestContentAddressedStore:
    
    def test_store_and_load(self, temp_store: ContentAddressedStore, base_time: datetime) -> None:
        events = (EmotionEvent(timestamp=base_time, emotion_label="neutral"),)
        timeline = EmotionTimeline(subject_id="test", events=events)
        payload = EmotionTimelineSerializer.serialize(timeline)
        
        fingerprint = temp_store.store(payload)
        assert fingerprint == payload.fingerprint
        
        loaded = temp_store.load(fingerprint)
        assert loaded is not None
        assert loaded.fingerprint == payload.fingerprint
        assert loaded.data == payload.data
    
    def test_load_nonexistent(self, temp_store: ContentAddressedStore) -> None:
        result = temp_store.load("nonexistent_fingerprint")
        assert result is None
    
    def test_exists(self, temp_store: ContentAddressedStore, base_time: datetime) -> None:
        timeline = EmotionTimeline(subject_id="test", events=())
        payload = EmotionTimelineSerializer.serialize(timeline)
        
        assert not temp_store.exists(payload.fingerprint)
        temp_store.store(payload)
        assert temp_store.exists(payload.fingerprint)
    
    def test_invalidate(self, temp_store: ContentAddressedStore, base_time: datetime) -> None:
        timeline = EmotionTimeline(subject_id="test", events=())
        payload = EmotionTimelineSerializer.serialize(timeline)
        
        temp_store.store(payload)
        assert temp_store.exists(payload.fingerprint)
        
        result = temp_store.invalidate(payload.fingerprint)
        assert result is True
        assert not temp_store.exists(payload.fingerprint)
    
    def test_invalidate_nonexistent(self, temp_store: ContentAddressedStore) -> None:
        result = temp_store.invalidate("nonexistent")
        assert result is False
    
    def test_idempotent_store(self, temp_store: ContentAddressedStore, base_time: datetime) -> None:
        timeline = EmotionTimeline(subject_id="test", events=())
        payload = EmotionTimelineSerializer.serialize(timeline)
        
        fp1 = temp_store.store(payload)
        fp2 = temp_store.store(payload)
        assert fp1 == fp2


class TestStructuralCache:
    
    def test_cache_hit(self, temp_cache: StructuralCache, base_time: datetime) -> None:
        timeline = EmotionTimeline(subject_id="test", events=())
        payload = EmotionTimelineSerializer.serialize(timeline)
        
        temp_cache.put(payload)
        retrieved = temp_cache.get(payload.fingerprint)
        
        assert retrieved is not None
        assert retrieved.fingerprint == payload.fingerprint
    
    def test_cache_miss(self, temp_cache: StructuralCache) -> None:
        result = temp_cache.get("nonexistent")
        assert result is None
    
    def test_is_valid(self, temp_cache: StructuralCache, base_time: datetime) -> None:
        timeline = EmotionTimeline(subject_id="test", events=())
        payload = EmotionTimelineSerializer.serialize(timeline)
        
        assert not temp_cache.is_valid(payload.fingerprint)
        temp_cache.put(payload)
        assert temp_cache.is_valid(payload.fingerprint)
    
    def test_get_or_compute_cache_hit(self, temp_cache: StructuralCache, base_time: datetime) -> None:
        timeline = EmotionTimeline(subject_id="test", events=())
        payload = EmotionTimelineSerializer.serialize(timeline)
        temp_cache.put(payload)
        
        compute_called = []
        def compute_fn():
            compute_called.append(True)
            return payload
        
        result = temp_cache.get_or_compute(payload.fingerprint, compute_fn)
        assert result.fingerprint == payload.fingerprint
        assert len(compute_called) == 0
    
    def test_get_or_compute_cache_miss(self, temp_cache: StructuralCache, base_time: datetime) -> None:
        timeline = EmotionTimeline(subject_id="test", events=())
        payload = EmotionTimelineSerializer.serialize(timeline)
        
        compute_called = []
        def compute_fn():
            compute_called.append(True)
            return payload
        
        result = temp_cache.get_or_compute(payload.fingerprint, compute_fn)
        assert result.fingerprint == payload.fingerprint
        assert len(compute_called) == 1
    
    def test_invalidate(self, temp_cache: StructuralCache, base_time: datetime) -> None:
        timeline = EmotionTimeline(subject_id="test", events=())
        payload = EmotionTimelineSerializer.serialize(timeline)
        temp_cache.put(payload)
        
        assert temp_cache.is_valid(payload.fingerprint)
        temp_cache.invalidate(payload.fingerprint)
        assert not temp_cache.is_valid(payload.fingerprint)
    
    def test_clear_memory_cache_still_loads_from_disk(self, temp_cache: StructuralCache, base_time: datetime) -> None:
        timeline = EmotionTimeline(subject_id="test", events=())
        payload = EmotionTimelineSerializer.serialize(timeline)
        temp_cache.put(payload)
        
        temp_cache.clear_memory_cache()
        
        retrieved = temp_cache.get(payload.fingerprint)
        assert retrieved is not None
        assert retrieved.fingerprint == payload.fingerprint


class TestStructuralChangeInvalidation:
    
    def test_different_structure_different_fingerprint(self, temp_cache: StructuralCache, base_time: datetime) -> None:
        timeline1 = EmotionTimeline(
            subject_id="test",
            events=(EmotionEvent(timestamp=base_time, emotion_label="neutral"),)
        )
        timeline2 = EmotionTimeline(
            subject_id="test",
            events=(
                EmotionEvent(timestamp=base_time, emotion_label="neutral"),
                EmotionEvent(timestamp=base_time + timedelta(minutes=10), emotion_label="happy"),
            )
        )
        
        payload1 = EmotionTimelineSerializer.serialize(timeline1)
        payload2 = EmotionTimelineSerializer.serialize(timeline2)
        
        assert payload1.fingerprint != payload2.fingerprint
        
        temp_cache.put(payload1)
        assert temp_cache.is_valid(payload1.fingerprint)
        assert not temp_cache.is_valid(payload2.fingerprint)
