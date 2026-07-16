import logging
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import keras

from src import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_test_set():

    X_test = np.load(config.X_TEST_PATH)
    y_test = np.load(config.y_TEST_PATH)
    label_names = np.load(config.LABEL_NAMES_PATH)

    return X_test, y_test, label_names


def plot_confusion_matrix(y_true, y_pred, label_names, save_path):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 6))

    sns.heatmap(

        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names,

    )

    plt.xlabel("Predicted genre")
    plt.ylabel("True genre")
    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return cm


def save_misclassified_samples(y_true, y_pred, y_proba, label_names, save_path, max_samples=20):
   
    wrong_idx = np.where(y_true != y_pred)[0]

    lines = [

        f"Total misclassified segments: {len(wrong_idx)} / {len(y_true)} "
        f"({100 * len(wrong_idx) / len(y_true):.1f}%)",
        "",
    ]

    for i in wrong_idx[:max_samples]:

        true_genre = label_names[y_true[i]]
        pred_genre = label_names[y_pred[i]]
        confidence = y_proba[i][y_pred[i]]

        lines.append(

            f"segment {i}: true={true_genre:<10} predicted={pred_genre:<10} "
            f"confidence={confidence:.2f}"

        )

    with open(save_path, "w") as f:

        f.write("\n".join(lines))

    return len(wrong_idx)


def main():
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    X_test, y_test, label_names = load_test_set()
    logger.info("Test set: X=%s y=%s", X_test.shape, y_test.shape)

    model = keras.models.load_model(config.MODEL_PATH)

    y_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)

    test_accuracy = np.mean(y_pred == y_test)
    logger.info("Test accuracy: %.4f", test_accuracy)

    cm = plot_confusion_matrix(y_test, y_pred, label_names, config.CONFUSION_MATRIX_PATH)

    logger.info("Confusion matrix saved to %s", config.CONFUSION_MATRIX_PATH)

    report = classification_report(y_test, y_pred, target_names=label_names, digits=3)

    with open(config.CLASSIFICATION_REPORT_PATH, "w") as f:

        f.write(f"Test accuracy: {test_accuracy:.4f}\n\n")
        f.write(report)

    logger.info("Classification report saved to %s", config.CLASSIFICATION_REPORT_PATH)
    print("\n" + report)

    n_wrong = save_misclassified_samples(

        y_test, y_pred, y_proba, label_names, config.MISCLASSIFIED_SAMPLES_PATH,

    )

    logger.info("Misclassified samples (%d total) saved to %s",
                n_wrong, config.MISCLASSIFIED_SAMPLES_PATH)

    
    cm_off_diag = cm.copy()

    np.fill_diagonal(cm_off_diag, 0)

    if cm_off_diag.max() > 0:

        true_idx, pred_idx = np.unravel_index(cm_off_diag.argmax(), cm_off_diag.shape)
        
        logger.info(
            "Most common confusion: true=%s predicted=%s (%d times)",
            label_names[true_idx], label_names[pred_idx], cm_off_diag[true_idx, pred_idx],
        )


if __name__ == "__main__":
    main()