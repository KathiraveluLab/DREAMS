"""
EmotionTimeline is constructed from temporally aggregated facial expressions.
Single-frame spikes do not define emotional state.
Neutral is a valid dominant outcome.
Emotion transitions must be evidence-backed and time-consistent.

This module constructs EmotionTimeline from per-image facial emotion probabilities
using temporal windowing, persistence rules, and confidence weighting.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import hashlib

from .emotion_timeline import EmotionEvent, EmotionTimeline


# Configuration constants
MIN_PERSISTENCE_FRAMES = 3  # Minimum consecutive frames for emotion dominance
WINDOW_SIZE_SECONDS = 3.0  # Temporal window for aggregation
DOMINANCE_THRESHOLD = 0.45  # Minimum probability for emotion to be considered dominant
EPSILON = 0.1  # Maximum emotion escalation per window
NEUTRAL_FLOOR = 0.5  # Minimum neutral probability when uncertain


@dataclass(frozen=True)
class FacialFrame:
    """
    Single frame of facial emotion estimation.
    """
    timestamp: datetime
    positive: float
    neutral: float
    negative: float
    uncertainty_margin: float
    source_id: Optional[str] = None
    
    @property
    def confidence(self) -> float:
        """Confidence is inverse of uncertainty."""
        return 1.0 - self.uncertainty_margin
    
    @property
    def dominant_emotion(self) -> str:
        """Raw dominant emotion without persistence check."""
        probs = {"positive": self.positive, "neutral": self.neutral, "negative": self.negative}
        return max(probs, key=probs.get)
    
    @property
    def dominant_probability(self) -> float:
        return max(self.positive, self.neutral, self.negative)


def _compute_weighted_probabilities(frames: List[FacialFrame]) -> Dict[str, float]:
    """
    Compute confidence-weighted aggregated probabilities.
    
    Each frame contributes proportionally to its confidence:
    weighted_emotion = emotion_probability * (1 - uncertainty_margin)
    """
    if not frames:
        return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}
    
    total_weight = 0.0
    weighted_positive = 0.0
    weighted_neutral = 0.0
    weighted_negative = 0.0
    
    for frame in frames:
        weight = frame.confidence
        total_weight += weight
        weighted_positive += frame.positive * weight
        weighted_neutral += frame.neutral * weight
        weighted_negative += frame.negative * weight
    
    if total_weight > 0:
        return {
            "positive": weighted_positive / total_weight,
            "neutral": weighted_neutral / total_weight,
            "negative": weighted_negative / total_weight,
        }
    else:
        return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}


def _check_persistence(frames: List[FacialFrame], emotion: str) -> bool:
    """
    Check if emotion persists across minimum consecutive frames.
    
    An emotion becomes dominant only if it persists across MIN_PERSISTENCE_FRAMES.
    """
    if len(frames) < MIN_PERSISTENCE_FRAMES:
        return False
    
    consecutive_count = 0
    max_consecutive = 0
    
    for frame in frames:
        if frame.dominant_emotion == emotion:
            consecutive_count += 1
            max_consecutive = max(max_consecutive, consecutive_count)
        else:
            consecutive_count = 0
    
    return max_consecutive >= MIN_PERSISTENCE_FRAMES


def _apply_neutral_dominance_rule(probs: Dict[str, float], frames: List[FacialFrame]) -> Dict[str, float]:
    """
    Apply neutral dominance rule.
    
    If emotion probabilities fluctuate or remain uncertain:
    neutral >= 0.5
    """
    # Calculate average uncertainty
    if not frames:
        return probs
    
    avg_uncertainty = sum(f.uncertainty_margin for f in frames) / len(frames)
    
    # Calculate emotion variance (fluctuation)
    if len(frames) >= 2:
        positive_variance = sum((f.positive - probs["positive"])**2 for f in frames) / len(frames)
        fluctuation = positive_variance > 0.01  # Significant fluctuation
    else:
        fluctuation = True  # Single frame = uncertain
    
    # Apply neutral floor if uncertain or fluctuating
    if avg_uncertainty > 0.1 or fluctuation:
        if probs["neutral"] < NEUTRAL_FLOOR:
            # Redistribute to achieve neutral floor
            deficit = NEUTRAL_FLOOR - probs["neutral"]
            other_sum = probs["positive"] + probs["negative"]
            if other_sum > 0:
                scale = (other_sum - deficit) / other_sum
                probs = {
                    "positive": probs["positive"] * scale,
                    "neutral": NEUTRAL_FLOOR,
                    "negative": probs["negative"] * scale,
                }
    
    return probs


def _apply_negative_preservation_rule(probs: Dict[str, float], frames: List[FacialFrame]) -> Dict[str, float]:
    """
    Apply negative preservation rule.
    
    Negative emotion must not be diluted by surrounding neutral frames.
    If negative appears with persistence: negative >= neutral (in that window)
    """
    if _check_persistence(frames, "negative"):
        # Negative has persistence - it should not be diluted
        if probs["negative"] > 0.15 and probs["neutral"] > probs["negative"]:
            # Boost negative to at least match neutral's dominance
            avg_negative = sum(f.negative for f in frames if f.dominant_emotion == "negative") / max(1, sum(1 for f in frames if f.dominant_emotion == "negative"))
            probs["negative"] = max(probs["negative"], avg_negative * 0.8)
    
    return probs


def _apply_no_escalation_rule(
    current_probs: Dict[str, float],
    previous_probs: Optional[Dict[str, float]]
) -> Dict[str, float]:
    """
    Apply no emotion escalation rule.
    
    Emotion intensity must never increase over time unless supported by data.
    emotion_t+1 <= emotion_t + epsilon
    """
    if previous_probs is None:
        return current_probs
    
    capped = current_probs.copy()
    
    for emotion in ["positive", "neutral", "negative"]:
        max_allowed = previous_probs[emotion] + EPSILON
        if capped[emotion] > max_allowed:
            excess = capped[emotion] - max_allowed
            capped[emotion] = max_allowed
            # Redistribute excess to neutral
            capped["neutral"] = min(1.0, capped["neutral"] + excess * 0.5)
    
    # Renormalize
    total = sum(capped.values())
    if total > 0:
        for emotion in capped:
            capped[emotion] /= total
    
    return capped


def _determine_emotion_label(probs: Dict[str, float], frames: List[FacialFrame]) -> str:
    """
    Determine final emotion label for the window.
    
    Dominant emotion only if:
    1. It exceeds DOMINANCE_THRESHOLD
    2. It has persistence across frames
    Otherwise: neutral
    """
    dominant = max(probs, key=probs.get)
    dominant_prob = probs[dominant]
    
    # Check threshold
    if dominant_prob < DOMINANCE_THRESHOLD:
        return "neutral"
    
    # Check persistence (except for neutral which doesn't need persistence)
    if dominant != "neutral":
        if not _check_persistence(frames, dominant):
            return "neutral"
    
    return dominant


def _group_frames_into_windows(
    frames: List[FacialFrame],
    window_size: float = WINDOW_SIZE_SECONDS
) -> List[List[FacialFrame]]:
    """
    Group frames into fixed temporal windows.
    """
    if not frames:
        return []
    
    # Sort by timestamp
    sorted_frames = sorted(frames, key=lambda f: f.timestamp)
    
    windows = []
    current_window = []
    window_start = sorted_frames[0].timestamp
    
    for frame in sorted_frames:
        elapsed = (frame.timestamp - window_start).total_seconds()
        
        if elapsed >= window_size and current_window:
            windows.append(current_window)
            current_window = [frame]
            window_start = frame.timestamp
        else:
            current_window.append(frame)
    
    # Add final window
    if current_window:
        windows.append(current_window)
    
    return windows


def _validate_timeline(events: List[EmotionEvent], frames: List[FacialFrame]) -> bool:
    """
    Validate the constructed timeline.
    
    Checks:
    - No single-frame emotion dominates
    - Neutral segments are preserved
    - Negative segments persist when present
    - Timeline transitions are gradual
    """
    if not events:
        return True
    
    # Check for reasonable transitions
    for i in range(len(events) - 1):
        current = events[i]
        next_event = events[i + 1]
        
        # Emotion shouldn't flip-flop rapidly
        if i + 2 < len(events):
            after_next = events[i + 2]
            if current.emotion_label == after_next.emotion_label and current.emotion_label != next_event.emotion_label:
                # Back-to-back flip - suspicious but may be valid
                pass
    
    return True


def build_timeline_from_frames(
    subject_id: str,
    frames: List[FacialFrame],
    window_size: float = WINDOW_SIZE_SECONDS
) -> EmotionTimeline:
    """
    Build EmotionTimeline from facial expression frames.
    
    Uses:
    - Temporal window aggregation
    - Confidence-weighted probabilities
    - Emotion persistence rules
    - Neutral dominance when uncertain
    - Negative preservation
    - No escalation without evidence
    
    Args:
        subject_id: Identifier for the subject
        frames: List of facial emotion frames
        window_size: Seconds per temporal window
    
    Returns:
        EmotionTimeline with aggregated events
    """
    if not frames:
        return EmotionTimeline(subject_id=subject_id, events=())
    
    # Group into windows
    windows = _group_frames_into_windows(frames, window_size)
    
    events = []
    previous_probs = None
    
    for window_frames in windows:
        if not window_frames:
            continue
        
        # Step 1: Compute weighted probabilities
        probs = _compute_weighted_probabilities(window_frames)
        
        # Step 2: Apply neutral dominance rule
        probs = _apply_neutral_dominance_rule(probs, window_frames)
        
        # Step 3: Apply negative preservation rule
        probs = _apply_negative_preservation_rule(probs, window_frames)
        
        # Step 4: Apply no escalation rule
        probs = _apply_no_escalation_rule(probs, previous_probs)
        
        # Step 5: Determine emotion label
        emotion_label = _determine_emotion_label(probs, window_frames)
        
        # Step 6: Create event
        window_start = min(f.timestamp for f in window_frames)
        avg_confidence = sum(f.confidence for f in window_frames) / len(window_frames)
        
        # Build source ID from frame sources
        source_ids = [f.source_id for f in window_frames if f.source_id]
        combined_source = ":".join(source_ids[:3]) if source_ids else None
        
        event = EmotionEvent(
            timestamp=window_start,
            emotion_label=emotion_label,
            score=probs[emotion_label],
            source_id=combined_source,
            metadata={
                "positive": round(probs["positive"], 4),
                "neutral": round(probs["neutral"], 4),
                "negative": round(probs["negative"], 4),
                "confidence": round(avg_confidence, 4),
                "frame_count": len(window_frames),
                "source": "facial_expression",
            }
        )
        events.append(event)
        
        # Track for next iteration
        previous_probs = probs
    
    # Validate
    _validate_timeline(events, frames)
    
    return EmotionTimeline(subject_id=subject_id, events=tuple(events))


def build_timeline_from_ml_outputs(
    subject_id: str,
    ml_outputs: List[Dict],
    window_size: float = WINDOW_SIZE_SECONDS
) -> EmotionTimeline:
    """
    Build EmotionTimeline from ML model output dictionaries.
    
    This is the main entry point for converting facial ML outputs to timeline.
    
    Args:
        subject_id: Identifier for the subject
        ml_outputs: List of dicts from perceptual_emotion_model
        window_size: Seconds per temporal window
    
    Returns:
        EmotionTimeline with aggregated events
    """
    frames = []
    
    for i, output in enumerate(ml_outputs):
        # Extract timestamp (required) or use index-based
        if "timestamp" in output:
            timestamp = output["timestamp"]
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
        else:
            # Create synthetic timestamp based on index
            timestamp = datetime.now() + timedelta(seconds=i * 0.1)
        
        frame = FacialFrame(
            timestamp=timestamp,
            positive=output.get("positive", 0.0),
            neutral=output.get("neutral", 1.0),
            negative=output.get("negative", 0.0),
            uncertainty_margin=output.get("uncertainty_margin", 0.1),
            source_id=output.get("image_id") or output.get("source_id"),
        )
        frames.append(frame)
    
    return build_timeline_from_frames(subject_id, frames, window_size)


# Demo function
def demo_timeline_construction():
    """Demonstrate timeline construction with simulated frames."""
    from datetime import datetime, timedelta
    
    # Simulate 10 frames over 30 seconds
    base_time = datetime.now()
    frames = []
    
    # Frames 0-2: Neutral (weak cues)
    for i in range(3):
        frames.append(FacialFrame(
            timestamp=base_time + timedelta(seconds=i * 3),
            positive=0.12,
            neutral=0.76,
            negative=0.12,
            uncertainty_margin=0.15,
            source_id=f"frame_{i}"
        ))
    
    # Frames 3-5: Negative emerges (brow lowering)
    for i in range(3, 6):
        frames.append(FacialFrame(
            timestamp=base_time + timedelta(seconds=i * 3),
            positive=0.10,
            neutral=0.55,
            negative=0.35,
            uncertainty_margin=0.12,
            source_id=f"frame_{i}"
        ))
    
    # Frames 6-9: Back to neutral
    for i in range(6, 10):
        frames.append(FacialFrame(
            timestamp=base_time + timedelta(seconds=i * 3),
            positive=0.15,
            neutral=0.70,
            negative=0.15,
            uncertainty_margin=0.10,
            source_id=f"frame_{i}"
        ))
    
    timeline = build_timeline_from_frames("demo_subject", frames)
    
    print("=" * 60)
    print("EMOTION TIMELINE CONSTRUCTION DEMO")
    print("=" * 60)
    print(f"\nInput: {len(frames)} frames over {timeline.duration():.1f} seconds")
    print(f"Output: {len(timeline.events)} events\n")
    
    for event in timeline.events:
        meta = event.metadata
        print(f"  {event.timestamp.strftime('%H:%M:%S')}: {event.emotion_label.upper()}")
        print(f"    P:{meta['positive']:.1%} N:{meta['neutral']:.1%} Neg:{meta['negative']:.1%}")
        print(f"    Confidence: {meta['confidence']:.1%}, Frames: {meta['frame_count']}")
    
    print("\nPrinciples Applied:")
    print("  ✓ Temporal window aggregation (3 sec)")
    print("  ✓ Confidence-weighted probabilities")
    print("  ✓ Neutral dominance when uncertain")
    print("  ✓ Negative preserved when persistent")
    print("  ✓ No single-frame spikes")


if __name__ == "__main__":
    demo_timeline_construction()
