import json
import logging
import os

import numpy as np
from tensorflow import keras

from src import config
from src.model import build_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EPOCHS = 50
BATCH_SIZE = 32


def load_split():
    missing = [p for p in [config.X_TRAIN_PATH, config.y_TRAIN_PATH,
                            config.X_TEST_PATH, config.y_TEST_PATH] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing split files: {missing}. "
        )

    X_train = np.load(config.X_TRAIN_PATH)
    y_train = np.load(config.y_TRAIN_PATH)
    X_test = np.load(config.X_TEST_PATH)
    y_test = np.load(config.y_TEST_PATH)

    label_names = np.load(config.LABEL_NAMES_PATH)

    return X_train, y_train, X_test, y_test, label_names


def main():
    X_train, y_train, X_test, y_test, label_names = load_split()
    num_classes = len(label_names)

    logger.info("Train: X=%s y=%s | Test: X=%s y=%s",
                X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    model = build_model(num_classes=num_classes)
    model.summary()

    os.makedirs(config.MODEL_DIR, exist_ok=True)

    callbacks = [
        
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            config.MODEL_PATH, monitor="val_accuracy", save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )

    
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(config.HISTORY_PATH, "w") as f:
        json.dump(history_dict, f, indent=2)

    final_val_acc = max(history.history["val_accuracy"])

    logger.info("Best validation accuracy: %.4f", final_val_acc)
    logger.info("Model saved to %s", config.MODEL_PATH)
    logger.info("History saved to %s", config.HISTORY_PATH)


    chance_level = 1.0 / num_classes
    logger.info("Chance-level accuracy for %d classes: %.4f", num_classes, chance_level)


if __name__ == "__main__":
    main()