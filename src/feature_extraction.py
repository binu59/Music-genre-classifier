import librosa
import numpy as np

from src import config


def _pad_or_truncate(feature_2d: np.ndarray, max_frames: int) -> np.ndarray:
    
    n_bins, frames = feature_2d.shape

    if frames == max_frames:
        return feature_2d
    
    if frames > max_frames:
        return feature_2d[:, :max_frames]
    
    pad_width = max_frames - frames

    return np.pad(feature_2d, ((0, 0), (0, pad_width)), mode="constant")


def extract_features(segment: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    
    feature_stack = []

    mfcc = librosa.feature.mfcc(
        y=segment, sr=sr, n_mfcc=config.N_MFCC,
        n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
    )
    feature_stack.append(_pad_or_truncate(mfcc, config.MAX_FRAMES))

    if config.USE_SPECTRAL_CENTROID:
        centroid = librosa.feature.spectral_centroid(
            y=segment, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
        )
        feature_stack.append(_pad_or_truncate(centroid, config.MAX_FRAMES))

    if config.USE_CHROMA:
        chroma = librosa.feature.chroma_stft(
            y=segment, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
        )
        feature_stack.append(_pad_or_truncate(chroma, config.MAX_FRAMES))

    if config.USE_SPECTRAL_CONTRAST:
        contrast = librosa.feature.spectral_contrast(
            y=segment, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
        )
        feature_stack.append(_pad_or_truncate(contrast, config.MAX_FRAMES))

    stacked = np.concatenate(feature_stack, axis=0)   
    return stacked.T.astype(np.float32)               


def segment_track(waveform: np.ndarray) -> list[np.ndarray]:
    
    segments = []

    total_samples = len(waveform)

    for start in range(0, total_samples - config.SEGMENT_SAMPLES + 1, config.SEGMENT_SAMPLES):
        segments.append(waveform[start:start + config.SEGMENT_SAMPLES])

    return segments


def feature_dim() -> int:
    
    dim = config.N_MFCC

    if config.USE_SPECTRAL_CENTROID:
        dim += 1

    if config.USE_CHROMA:
        dim += 12

    if config.USE_SPECTRAL_CONTRAST:
        dim += 7

    return dim


if __name__ == "__main__":
    
    fake_waveform = np.random.randn(config.SAMPLE_RATE * config.TRACK_DURATION_SEC).astype(np.float32)

    segments = segment_track(fake_waveform)

    print(f"Segments per track: {len(segments)} (expected {config.SEGMENTS_PER_TRACK})")

    feats = extract_features(segments[0])
    
    print(f"Feature shape per segment: {feats.shape} "
          f"(expected ({config.MAX_FRAMES}, {feature_dim()}))")