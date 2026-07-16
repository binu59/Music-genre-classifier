"""
Central configuration for the LSTM genre classifier.
Every other script imports from here so all settings live in one place.
"""

import os

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# TFDS 'gtzan' ships all 10 genres with a single 'train' split (1000 clips,
# 100 per genre). We train on a subset first and can widen TARGET_GENRES
# later without touching any other code.
ALL_GTZAN_GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]

TARGET_GENRES = ["classical", "hiphop", "jazz", "metal", "rock"]

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
SAMPLE_RATE = 22050          # native rate for GTZAN clips
TRACK_DURATION_SEC = 30      # full clip length in GTZAN
SEGMENT_DURATION_SEC = 3     # split each 30s track into shorter segments
SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_DURATION_SEC
SEGMENTS_PER_TRACK = TRACK_DURATION_SEC // SEGMENT_DURATION_SEC  # -> 10

# Why segment at all: 100 tracks/genre is a small dataset for an LSTM.
# Splitting each 30s clip into 3s segments turns 1000 tracks into up to
# ~10,000 training sequences and keeps each sequence short enough for the
# LSTM to learn from without needing an enormous hidden state.

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
N_MFCC = 20
N_FFT = 2048
HOP_LENGTH = 512

USE_SPECTRAL_CENTROID = True
USE_CHROMA = True
USE_SPECTRAL_CONTRAST = True

# Every segment is padded/truncated to this many time frames so all
# sequences fed to the LSTM have an identical (timesteps, features) shape.
# librosa yields ~1 + SEGMENT_SAMPLES / HOP_LENGTH frames per segment
# (~130 frames for 3s @ 22050Hz, hop=512); we fix it explicitly here.
MAX_FRAMES = 130

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

FEATURES_PATH = os.path.join(PROCESSED_DIR, "features.npy")
LABELS_PATH = os.path.join(PROCESSED_DIR, "labels.npy")
LABEL_NAMES_PATH = os.path.join(PROCESSED_DIR, "label_names.npy")
TRACK_IDS_PATH = os.path.join(PROCESSED_DIR, "track_ids.npy")

# Train/test split, done at the TRACK level (not segment level) to avoid
# leakage -- see split.py for why this matters.
TEST_SIZE = 0.2  # fraction of TRACKS (not segments) held out per genre

X_TRAIN_PATH = os.path.join(PROCESSED_DIR, "X_train.npy")

y_TRAIN_PATH = os.path.join(PROCESSED_DIR, "y_train.npy")
X_TEST_PATH = os.path.join(PROCESSED_DIR, "X_test.npy")
y_TEST_PATH = os.path.join(PROCESSED_DIR, "y_test.npy")

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_genre_classifier.h5")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.json")

REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
CONFUSION_MATRIX_PATH = os.path.join(REPORTS_DIR, "confusion_matrix.png")
CLASSIFICATION_REPORT_PATH = os.path.join(REPORTS_DIR, "classification_report.txt")
MISCLASSIFIED_SAMPLES_PATH = os.path.join(REPORTS_DIR, "misclassified_samples.txt")

RANDOM_SEED = 42