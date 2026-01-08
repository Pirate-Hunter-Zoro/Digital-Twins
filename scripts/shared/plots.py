from pathlib import Path
import matplotlib.pyplot as plt 
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
    plt.legend()
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
    plt.legend()
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
    plt.plot(prob_pred_per_bin, prob_true_per_bin, marker='o', label="Model")
    plt.plot([0,1],[0,1]) # Representing perfect calibration
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.savefig(f"{str(RESULTS_DIR)}/calibration_curve.png")
    plt.close()

def plot_decision_curve_analysis(y_true: np.array, y_prob: np.array):
    """
    Plot decision curve benefits - when only assuming patients above a certain threshold are positive, what is the benefit
    
    :param y_true: Actual labels
    :type y_true: np.array
    :param y_prob: Predicted probability labels
    :type y_prob: np.array
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    TP_ASSIGN_ALL_POSITIVE = np.sum(y_true) # Count of true positives
    FP_ASSIGN_ALL_POSITIVE = np.sum(1-y_true)
    N = y_true.shape[0]
    
    def positive_all_benefit(threshold: float) -> float:
        """
        Helper method to return the benefit attributed with applying the given threshold/penalty classifying all observations as positive
        
        :param threshold: Penalty for false positive
        :type threshold: float
        :return: Resulting benefit
        :rtype: float
        """
        return TP_ASSIGN_ALL_POSITIVE/N - FP_ASSIGN_ALL_POSITIVE/N*threshold/(1-threshold)
     
    plt.plot(thresholds, np.zeros_like(thresholds), label="Threshold One (All Negative) Benefit")
    
    # Calculate benefits over all thresholds
    expanded_y_prob = y_prob[:, None] # N x 1
    assign_at_thresholds = expanded_y_prob >= thresholds # row is observation, column is threshold, boolean value is if observation is positive at that threshold
    expanded_y_true = y_true[:, None]
    TP = assign_at_thresholds & (expanded_y_true == 1) # True positive flags at each threshold over all patients - N x 100
    FP = assign_at_thresholds & (expanded_y_true == 0) # False positive flags at each threshold over all patients
    TP_OVER_THRESHOLDS = np.sum(TP, axis=0) # (100,)
    FP_OVER_THRESHOLDS = np.sum(FP, axis=0) # (100,)
    benefits_by_threshold = TP_OVER_THRESHOLDS / N - (FP_OVER_THRESHOLDS / N)*(thresholds/(1-thresholds))
    plt.plot(thresholds, benefits_by_threshold, label="Model Benefit by Threshold")
    
    benefits_assign_all_positive = positive_all_benefit(thresholds)
    plt.plot(thresholds, benefits_assign_all_positive, label="Threshold Zero (All Positive) Benefit by False Positive Penalty")
    
    plt.xlabel("Threshold / False Positive Penalty")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis")
    plt.ylim(bottom=-0.1)
    plt.legend()
    plt.savefig(f"{str(RESULTS_DIR)}/decision_curve_analysis.png")
    plt.close()

def plot_effective_sample_size_distribution(ess_values: np.array):
    """
    Create a histogram of the effective sample sizes observed in the predictions
    
    :param ess_values: Effective sample sizes from predictions
    :type ess_values: np.array
    """
    plt.hist(ess_values, bins=20, edgecolor='black')
    plt.xlabel("Effective Sample Size")
    plt.ylabel("Frequency")
    plt.title("Effective Sample Size Distribution")
    plt.axvline(x=int(os.environ['LOW_CONFIDENCE_ESS_THRESHOLD']), color='red', linestyle='--', linewidth=2, label='Low Confidence (<20)')
    plt.legend()
    plt.savefig(f"{str(RESULTS_DIR)}/effective_sample_size_distribution.png")
    plt.close()