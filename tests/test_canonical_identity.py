# tests/test_canonical_identity.py

import pytest
from datetime import datetime, timedelta

from dreamsApp.analytics.emotion_timeline import EmotionEvent, EmotionTimeline
from dreamsApp.analytics.emotion_episode import Episode
from dreamsApp.analytics.temporal_narrative_graph import build_narrative_graph


@pytest.fixture
def base_time() -> datetime:
    return datetime(2024, 1, 1, 12, 0, 0)


class TestTimelineFingerprint:
    
    def test_fingerprint_deterministic(self, base_time: datetime) -> None:
        events = (
            EmotionEvent(timestamp=base_time, emotion_label="neutral"),
            EmotionEvent(timestamp=base_time + timedelta(minutes=10), emotion_label="positive"),
        )
        timeline = EmotionTimeline(subject_id="test", events=events)
        
        fp1 = timeline.fingerprint()
        fp2 = timeline.fingerprint()
        assert fp1 == fp2
    
    def test_fingerprint_same_structure_different_labels(self, base_time: datetime) -> None:
        events1 = (
            EmotionEvent(timestamp=base_time, emotion_label="neutral"),
            EmotionEvent(timestamp=base_time + timedelta(minutes=10), emotion_label="positive"),
        )
        events2 = (
            EmotionEvent(timestamp=base_time, emotion_label="angry"),
            EmotionEvent(timestamp=base_time + timedelta(minutes=10), emotion_label="sad"),
        )
        timeline1 = EmotionTimeline(subject_id="test1", events=events1)
        timeline2 = EmotionTimeline(subject_id="test2", events=events2)
        
        assert timeline1.fingerprint() == timeline2.fingerprint()
    
    def test_fingerprint_same_structure_different_scores(self, base_time: datetime) -> None:
        events1 = (
            EmotionEvent(timestamp=base_time, emotion_label="neutral", score=0.5),
            EmotionEvent(timestamp=base_time + timedelta(minutes=10), emotion_label="positive", score=0.9),
        )
        events2 = (
            EmotionEvent(timestamp=base_time, emotion_label="neutral", score=0.1),
            EmotionEvent(timestamp=base_time + timedelta(minutes=10), emotion_label="positive", score=0.3),
        )
        timeline1 = EmotionTimeline(subject_id="test", events=events1)
        timeline2 = EmotionTimeline(subject_id="test", events=events2)
        
        assert timeline1.fingerprint() == timeline2.fingerprint()
    
    def test_fingerprint_different_structure_different_fingerprint(self, base_time: datetime) -> None:
        events1 = (
            EmotionEvent(timestamp=base_time, emotion_label="neutral"),
            EmotionEvent(timestamp=base_time + timedelta(minutes=10), emotion_label="positive"),
        )
        events2 = (
            EmotionEvent(timestamp=base_time, emotion_label="neutral"),
            EmotionEvent(timestamp=base_time + timedelta(minutes=20), emotion_label="positive"),
        )
        timeline1 = EmotionTimeline(subject_id="test", events=events1)
        timeline2 = EmotionTimeline(subject_id="test", events=events2)
        
        assert timeline1.fingerprint() != timeline2.fingerprint()
    
    def test_fingerprint_different_event_count(self, base_time: datetime) -> None:
        events1 = (
            EmotionEvent(timestamp=base_time, emotion_label="neutral"),
        )
        events2 = (
            EmotionEvent(timestamp=base_time, emotion_label="neutral"),
            EmotionEvent(timestamp=base_time + timedelta(minutes=10), emotion_label="neutral"),
        )
        timeline1 = EmotionTimeline(subject_id="test", events=events1)
        timeline2 = EmotionTimeline(subject_id="test", events=events2)
        
        assert timeline1.fingerprint() != timeline2.fingerprint()
    
    def test_empty_timeline_fingerprint(self) -> None:
        timeline = EmotionTimeline(subject_id="empty", events=())
        fp = timeline.fingerprint()
        assert isinstance(fp, str)
        assert len(fp) == 32


class TestEpisodeId:
    
    def test_episode_id_deterministic(self, base_time: datetime) -> None:
        episode = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            events=(),
            source_subject_id="test"
        )
        id1 = episode.episode_id
        id2 = episode.episode_id
        assert id1 == id2
    
    def test_episode_id_same_temporal_bounds_same_id(self, base_time: datetime) -> None:
        events1 = (EmotionEvent(timestamp=base_time + timedelta(minutes=10), emotion_label="neutral"),)
        events2 = (
            EmotionEvent(timestamp=base_time + timedelta(minutes=10), emotion_label="happy"),
            EmotionEvent(timestamp=base_time + timedelta(minutes=20), emotion_label="sad"),
        )
        
        ep1 = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            events=events1,
            source_subject_id="test"
        )
        ep2 = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            events=events2,
            source_subject_id="test"
        )
        assert ep1.episode_id == ep2.episode_id
    
    def test_episode_id_different_times_different_id(self, base_time: datetime) -> None:
        ep1 = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            events=(),
        )
        ep2 = Episode(
            start_time=base_time + timedelta(hours=1),
            end_time=base_time + timedelta(hours=2),
            events=(),
        )
        assert ep1.episode_id != ep2.episode_id
    
    def test_episode_id_different_subject_different_id(self, base_time: datetime) -> None:
        ep1 = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            events=(),
            source_subject_id="subject1"
        )
        ep2 = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            events=(),
            source_subject_id="subject2"
        )
        assert ep1.episode_id != ep2.episode_id
    
    def test_episode_id_in_to_dict(self, base_time: datetime) -> None:
        episode = Episode(
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            events=(),
        )
        d = episode.to_dict()
        assert 'episode_id' in d
        assert d['episode_id'] == episode.episode_id


class TestGraphId:
    
    def test_graph_id_deterministic(self, base_time: datetime) -> None:
        episodes = [
            Episode(start_time=base_time, end_time=base_time + timedelta(hours=1), events=()),
            Episode(start_time=base_time + timedelta(hours=1), end_time=base_time + timedelta(hours=2), events=()),
        ]
        graph = build_narrative_graph(episodes)
        id1 = graph.graph_id
        id2 = graph.graph_id
        assert id1 == id2
    
    def test_graph_id_same_episodes_same_id(self, base_time: datetime) -> None:
        episodes = [
            Episode(start_time=base_time, end_time=base_time + timedelta(hours=1), events=()),
            Episode(start_time=base_time + timedelta(hours=1), end_time=base_time + timedelta(hours=2), events=()),
        ]
        graph1 = build_narrative_graph(episodes)
        graph2 = build_narrative_graph(episodes)
        assert graph1.graph_id == graph2.graph_id
    
    def test_graph_id_different_episodes_different_id(self, base_time: datetime) -> None:
        episodes1 = [
            Episode(start_time=base_time, end_time=base_time + timedelta(hours=1), events=()),
        ]
        episodes2 = [
            Episode(start_time=base_time, end_time=base_time + timedelta(hours=2), events=()),
        ]
        graph1 = build_narrative_graph(episodes1)
        graph2 = build_narrative_graph(episodes2)
        assert graph1.graph_id != graph2.graph_id
    
    def test_graph_id_in_to_dict(self, base_time: datetime) -> None:
        graph = build_narrative_graph([])
        d = graph.to_dict()
        assert 'graph_id' in d
