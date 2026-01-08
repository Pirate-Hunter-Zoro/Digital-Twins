from pathlib import Path
import matplotlib.pyplot as plt 
import seaborn
import sklearn.metrics
import sklearn.calibration as calibration
import numpy as np
import os

from dotenv import load_dotenv
load_dotenv()

RESULTS_DIR = Path(os.environ['RESULTS_DIR'])

def plot_receiving_operator_characteristic(y_true: np.array, y_prob: np.array):
    """
    Create and save the ROC area under curve graph for the given values and predictions
    
    :param y_true: Actual labels
    :type y_true: np.array
    :param y_prob: Predicted probability labels
    :type y_prob: np.array
    """
    score = sklearn.metrics.roc_auc_score(y_true=y_true, y_score=y_prob)
    false_positive_rate, true_positive_rate, _ = sklearn.metrics.roc_curve(y_true=y_true, y_score=y_prob)
    plt.plot(false_positive_rate, true_positive_rate, color='red', label=f'ROC curve (score {score:.2f})')
    plt.plot([0,1], [0,1], color='green', linestyle='--')
    plt.title("Receiver Operating Characteristic")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(f"{str(RESULTS_DIR)}/roc_curve.png")
    plt.close()

def plot_precision_recall(y_true: np.array, y_prob: np.array):
    """
    Create and save the precision recall graph for the given values and predictions
    
    :param y_true: Actual labels
    :type y_true: np.array
    :param y_prob: Predicted probability labels
    :type y_prob: np.array
    """
    score = sklearn.metrics.average_precision_score(y_true=y_true, y_score=y_prob)
    precision, recall, _ = sklearn.metrics.precision_recall_curve(y_true=y_true, probas_pred=y_prob)
    plt.plot(recall, precision, label=f'PR Curve (Average Precision = {score:.2f})')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve")
    plt.savefig(f"{str(RESULTS_DIR)}/pr_curve.png")
    plt.close()

def plot_calibration(y_true: np.array, y_prob: np.array):
    """
    Create and save the calibration graph for the given values and predictions
    
    :param y_true: Actual labels
    :type y_true: np.array
    :param y_prob: Predicted probability labels
    :type y_prob: np.array
    """
    # For each bin, calculate average probability, and calculate true probability (average positive rating)
    prob_true_per_bin, prob_pred_per_bin = calibration.calibration_curve(y_true=y_true, y_prob=y_prob, n_bins=10)
    plt.plot(prob_true_per_bin, prob_pred_per_bin, marker='o', label="Model")
    plt.plot([0,1],[0,1]) # Representing perfect calibration
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.savefig(f"{str(RESULTS_DIR)}/calibration_curve.png")
    plt.close()

def plot_decision_curve_analysis():
    pass

def plot_effective_sample_size_distribution():
    pass