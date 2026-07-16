from tensorflow import keras
from tensorflow.keras import layers

from src import config
from src.feature_extraction import feature_dim


def build_model(num_classes: int,
                 timesteps: int = config.MAX_FRAMES,
                 n_features: int = None) -> keras.Model:

    if n_features is None:
        n_features = feature_dim()

    model = keras.Sequential([
        layers.Input(shape=(timesteps, n_features)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(32),
        layers.Dense(32, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    
    import numpy as np

    model = build_model(num_classes=5)
    model.summary()

    fake_batch = np.random.randn(4, config.MAX_FRAMES, feature_dim()).astype("float32")
    preds = model.predict(fake_batch, verbose=0)
    print(f"\nOutput shape: {preds.shape} (expected (4, 5))")
    print(f"Row sums (should be ~1.0 each, softmax): {preds.sum(axis=1)}")