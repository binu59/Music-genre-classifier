from dataclasses import dataclass

import numpy as np

from src import config
from src.feature_extraction import extract_features, segment_track


@dataclass
class PredictionResult:
    predicted_genre: str
    confidence: float                  # 0-1, the winning genre's probability
    genre_probabilities: dict          # {genre_name: probability}, all genres
    n_segments_used: int               # how many 3s segments the clip was split into


def predict_clip(waveform: np.ndarray, model, label_names: list) -> PredictionResult:
    """
    waveform: 1D float array at config.SAMPLE_RATE (already resampled/mono).
    model: loaded Keras model.
    label_names: list of genre names, index-aligned with model output.

    Splits the clip into 3s segments (same as training), runs the model on
    each, and averages the per-segment probabilities into one whole-clip
    prediction. Averaging (rather than majority vote) uses the model's full
    confidence on each segment, so a few uncertain segments don't outweigh
    several confident ones.
    """
    segments = segment_track(waveform)
    if not segments:
        raise ValueError(
            f"Clip is too short — need at least {config.SEGMENT_DURATION_SEC}s "
            f"of audio, got {len(waveform) / config.SAMPLE_RATE:.1f}s."
        )

    X = np.stack([extract_features(seg) for seg in segments])
    per_segment_probs = model.predict(X, verbose=0)   # (n_segments, n_classes)
    avg_probs = per_segment_probs.mean(axis=0)          # (n_classes,)

    pred_idx = int(np.argmax(avg_probs))
    genre_probabilities = {
        str(label_names[i]): float(avg_probs[i]) for i in range(len(label_names))
    }

    return PredictionResult(
        predicted_genre=str(label_names[pred_idx]),
        confidence=float(avg_probs[pred_idx]),
        genre_probabilities=genre_probabilities,
        n_segments_used=len(segments),
    )