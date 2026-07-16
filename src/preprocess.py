"""
Day 1-2 pipeline entry point.

Loads GTZAN (via Hugging Face datasets), keeps only TARGET_GENRES, splits
each track into segments, extracts features per segment, and saves the
result to disk as four numpy arrays:

    data/processed/features.npy     shape (n_segments, MAX_FRAMES, n_features)
    data/processed/labels.npy       shape (n_segments,)   int class indices
    data/processed/label_names.npy  shape (n_classes,)    genre name strings,
                                     index-aligned with the label ints above
    data/processed/track_ids.npy    shape (n_segments,)   int track id --
                                     every segment from the same original
                                     track shares the same id. Needed by
                                     split.py to do a track-level (not
                                     segment-level) train/test split, since
                                     splitting at the segment level would
                                     leak parts of the same song into both
                                     train and test and inflate accuracy.

Run this once. Every later script (split.py, train.py, evaluate.py) loads
from these cached arrays instead of re-downloading/re-extracting audio.

Usage:
    python -m src.preprocess
"""

import logging
import time

import numpy as np

from src import config
from src.data_loader import load_gtzan_subset, get_label_names
from src.feature_extraction import extract_features, segment_track

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_dataset():
    label_names = get_label_names()
    name_to_idx = {name: i for i, name in enumerate(label_names)}

    all_features = []
    all_labels = []
    all_track_ids = []

    start = time.time()
    n_tracks = 0

    for waveform, genre_name in load_gtzan_subset():
        track_id = n_tracks   # unique id per track, assigned in loading order
        n_tracks += 1
        segments = segment_track(waveform)
        label_idx = name_to_idx[genre_name]

        for seg in segments:
            feats = extract_features(seg)
            all_features.append(feats)
            all_labels.append(label_idx)
            all_track_ids.append(track_id)

        if n_tracks % 50 == 0:
            elapsed = time.time() - start
            logger.info("Processed %d tracks (%.1fs elapsed)", n_tracks, elapsed)

    X = np.stack(all_features, axis=0)   # (n_segments, MAX_FRAMES, n_features)
    y = np.array(all_labels, dtype=np.int64)
    track_ids = np.array(all_track_ids, dtype=np.int64)

    elapsed = time.time() - start
    logger.info(
        "Done: %d tracks -> %d segments in %.1fs. X.shape=%s, y.shape=%s",
        n_tracks, len(all_labels), elapsed, X.shape, y.shape,
    )
    return X, y, track_ids, np.array(label_names)


def main():
    import os
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)

    X, y, track_ids, label_names = build_dataset()

    np.save(config.FEATURES_PATH, X)
    np.save(config.LABELS_PATH, y)
    np.save(config.LABEL_NAMES_PATH, label_names)
    np.save(config.TRACK_IDS_PATH, track_ids)

    logger.info("Saved features to %s", config.FEATURES_PATH)
    logger.info("Saved labels to %s", config.LABELS_PATH)
    logger.info("Saved label names to %s", config.LABEL_NAMES_PATH)
    logger.info("Saved track ids to %s", config.TRACK_IDS_PATH)

    # Quick sanity check on class balance -- catches genre-filtering bugs
    # or a corrupted/missing file before you waste time on Day 3 training.
    unique, counts = np.unique(y, return_counts=True)
    logger.info("Class balance:")
    for idx, count in zip(unique, counts):
        logger.info("  %s: %d segments", label_names[idx], count)


if __name__ == "__main__":
    main()