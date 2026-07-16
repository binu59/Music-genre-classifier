import logging

import numpy as np
from datasets import load_dataset

from src import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_gtzan_subset():

    ds = load_dataset("marsyas/gtzan", "all", trust_remote_code=True)

    
    if hasattr(ds, "keys") and "train" in ds:
        ds = ds["train"]

    label_feature = ds.features["genre"]
    label_names = label_feature.names
    target_set = set(config.TARGET_GENRES)

    missing = target_set - set(label_names)
    if missing:
        raise ValueError(
            f"TARGET_GENRES contains genres not in GTZAN: {missing}. "
            f"Available genres: {label_names}"
        )

    kept, skipped = 0, 0
    for example in ds:
        genre_name = label_feature.int2str(example["genre"])
        if genre_name not in target_set:
            skipped += 1
            continue
        kept += 1
        waveform = np.asarray(example["audio"]["array"], dtype=np.float32)
        yield waveform, genre_name

    logger.info("Loaded %d clips (%d skipped, not in TARGET_GENRES)", kept, skipped)


def get_label_names():
    
    return sorted(config.TARGET_GENRES)


if __name__ == "__main__":
    # Quick smoke test: confirms TFDS can reach the dataset and that the
    # genre filter works. Run this first after cloning the repo.
    names = get_label_names()
    print(f"Target genres (sorted): {names}")

    count = 0
    for waveform, genre in load_gtzan_subset():
        count += 1
        if count <= 3:
            print(f"  sample {count}: genre={genre!r}, "
                  f"waveform shape={waveform.shape}, dtype={waveform.dtype}")
    print(f"Total clips loaded for target genres: {count}")