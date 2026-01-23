# dreamsApp/analytics/__init__.py

from .emotion_timeline import EmotionEvent, EmotionTimeline
from .emotion_episode import Episode
from .emotion_segmentation import TimeWindow, segment_timeline_by_gaps
from .episode_segmentation import segment_timeline_to_episodes
from .episode_proximity import (
    ProximityRelation,
    compute_temporal_overlap,
    compute_temporal_gap,
    are_episodes_adjacent,
    classify_episode_proximity,
)
from .temporal_narrative_graph import (
    NarrativeEdge,
    TemporalNarrativeGraph,
    build_narrative_graph,
)
from .serialization import (
    SCHEMA_VERSION,
    SerializedPayload,
    EmotionTimelineSerializer,
    EpisodeSerializer,
    TemporalNarrativeGraphSerializer,
)
from .persistence import (
    ContentAddressedStore,
    StructuralCache,
)
from .frontend_contract import (
    FrontendNode,
    FrontendEdge,
    FrontendGraphPayload,
    build_frontend_payload,
)
from .timeline_builder import (
    FacialFrame,
    build_timeline_from_frames,
    build_timeline_from_ml_outputs,
)

__all__ = [
    'EmotionEvent',
    'EmotionTimeline',
    'Episode',
    'TimeWindow',
    'segment_timeline_by_gaps',
    'segment_timeline_to_episodes',
    'ProximityRelation',
    'compute_temporal_overlap',
    'compute_temporal_gap',
    'are_episodes_adjacent',
    'classify_episode_proximity',
    'NarrativeEdge',
    'TemporalNarrativeGraph',
    'build_narrative_graph',
    'SCHEMA_VERSION',
    'SerializedPayload',
    'EmotionTimelineSerializer',
    'EpisodeSerializer',
    'TemporalNarrativeGraphSerializer',
    'ContentAddressedStore',
    'StructuralCache',
    'FrontendNode',
    'FrontendEdge',
    'FrontendGraphPayload',
    'build_frontend_payload',
    'FacialFrame',
    'build_timeline_from_frames',
    'build_timeline_from_ml_outputs',
]

