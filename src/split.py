import logging

import numpy as np

from src import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def track_level_split(track_ids: np.ndarray, labels: np.ndarray,
                       test_size: float = config.TEST_SIZE,
                       seed: int = config.RANDOM_SEED):
    
    rng = np.random.default_rng(seed)

    
    track_to_label = dict(zip(track_ids, labels))
    unique_tracks = np.array(list(track_to_label.keys()))
    track_labels = np.array(list(track_to_label.values()))

    train_ids, test_ids = [], []

    for label in np.unique(track_labels):

        genre_tracks = unique_tracks[track_labels == label].copy()
        rng.shuffle(genre_tracks)
        n_test = max(1, round(len(genre_tracks) * test_size))
        test_ids.extend(genre_tracks[:n_test])
        train_ids.extend(genre_tracks[n_test:])

    return set(train_ids), set(test_ids)


def main():

    X = np.load(config.FEATURES_PATH)
    y = np.load(config.LABELS_PATH)
    track_ids = np.load(config.TRACK_IDS_PATH)
    label_names = np.load(config.LABEL_NAMES_PATH)

    train_track_ids, test_track_ids = track_level_split(track_ids, y)

    train_mask = np.isin(track_ids, list(train_track_ids))
    test_mask = np.isin(track_ids, list(test_track_ids))

    
    overlap = train_track_ids & test_track_ids

    assert not overlap, f"Track leakage between train/test: {overlap}"

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    np.save(config.X_TRAIN_PATH, X_train)
    np.save(config.y_TRAIN_PATH, y_train)
    np.save(config.X_TEST_PATH, X_test)
    np.save(config.y_TEST_PATH, y_test)

    logger.info(
        "Split %d tracks -> %d train tracks / %d test tracks",
        len(train_track_ids) + len(test_track_ids),
        len(train_track_ids), len(test_track_ids),
    )

    logger.info(
        "Segments: %d train, %d test (%.1f%% test)",
        len(y_train), len(y_test), 100 * len(y_test) / (len(y_train) + len(y_test)),
    )

    logger.info("Per-genre track counts (train / test):")
    track_to_label = dict(zip(track_ids, y))
    
    for idx, name in enumerate(label_names):
        genre_train = sum(1 for tid in train_track_ids if track_to_label[tid] == idx)
        genre_test = sum(1 for tid in test_track_ids if track_to_label[tid] == idx)
        logger.info("  %s: %d / %d", name, genre_train, genre_test)

    logger.info("Saved X_train%s, y_train%s, X_test%s, y_test%s",
                 X_train.shape, y_train.shape, X_test.shape, y_test.shape)


if __name__ == "__main__":
    main()