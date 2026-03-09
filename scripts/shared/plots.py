from pathlib import Path
import matplotlib.pyplot as plt 
import sklearn.metrics
import sklearn.calibration as calibration
import numpy as np
import os

from dotenv import load_dotenv
load_dotenv()

RESULTS_DIR = Path(os.environ['RESULTS_DIR'])

def plot_receiving_operator_characteristic(y_true: np.array, y_prob: np.array, mode: str):
    """
    Create and save the ROC area under curve graph for the given values and predictions
    
    :param y_true: Actual labels
    :type y_true: np.array
    :param y_prob: Predicted probability labels
    :type y_prob: np.array
    :param mode: llm weighting, cosine, weighting, uniform weighting
    :type mode: str
    """
    score = sklearn.metrics.roc_auc_score(y_true=y_true, y_score=y_prob)
    false_positive_rate, true_positive_rate, _ = sklearn.metrics.roc_curve(y_true=y_true, y_score=y_prob)
    plt.plot(false_positive_rate, true_positive_rate, color='red', label=f'ROC curve (score {score:.2f})')
    plt.plot([0,1], [0,1], color='green', linestyle='--')
    plt.title("Receiver Operating Characteristic")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    
    # Bootstrapping for error bands
    base_false_positive_rate = np.linspace(0,1,100)
    interpolated_true_positive_rates = []
    for _ in range(1000):
        # Bootstrap - sample entire sample size with replacement
        sample_indices = np.random.randint(low=0, high=y_true.shape[0], size=y_true.shape[0])
        y_true_sample = y_true[sample_indices]
        y_prob_sample = y_prob[sample_indices]
        if len(np.unique(y_true_sample)) < 2: # Make sure by chance we did not sample only one class
            continue
        sample_fp, sample_tp, _ = sklearn.metrics.roc_curve(y_true=y_true_sample, y_score=y_prob_sample)
        # For FP and TP values, interpolate them on standard np.linspace(0,1,100) to force false positive rates on grid and then estimating respective true positive values
        interpolated_roc_curve = np.interp(base_false_positive_rate, sample_fp, sample_tp)
        interpolated_roc_curve[0] = 0.0
        interpolated_true_positive_rates.append(interpolated_roc_curve)
    
    # Error bands are 2.5 percentile and 97.5 percentile for each FP x-value on ROC curve which generates 95% confidence interval
    interpolated_tp = np.array(interpolated_true_positive_rates)
    q_low = np.percentile(interpolated_tp, 2.5, axis=0)
    q_high = np.percentile(interpolated_tp, 97.5, axis=0)
    plt.fill_between(base_false_positive_rate, q_low, q_high, color='gray', alpha=0.2, label='95% CI')
    
    plt.legend()
    save_path = RESULTS_DIR / "roc_curves" / f"roc_curve_{mode}.png"
    os.makedirs(save_path.parent, exist_ok=True)
    plt.savefig(str(save_path))
    plt.close()

def plot_precision_recall(y_true: np.array, y_prob: np.array, mode: str):
    """
    Create and save the precision recall graph for the given values and predictions
    
    :param y_true: Actual labels
    :type y_true: np.array
    :param y_prob: Predicted probability labels
    :type y_prob: np.array
    :param mode: llm weighting, cosine, weighting, uniform weighting
    :type mode: str
    """
    score = sklearn.metrics.average_precision_score(y_true=y_true, y_score=y_prob)
    precision, recall, _ = sklearn.metrics.precision_recall_curve(y_true=y_true, y_score=y_prob)
    plt.plot(recall, precision, label=f'PR Curve (Average Precision = {score:.2f})')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve")
    plt.legend()
    save_path = RESULTS_DIR / "pr_curves" / f"pr_curve_{mode}.png"
    os.makedirs(save_path.parent, exist_ok=True)
    plt.savefig(str(save_path))
    plt.close()

def plot_calibration(y_true: np.array, y_prob: np.array, mode: str):
    """
    Create and save the calibration graph for the given values and predictions
    
    :param y_true: Actual labels
    :type y_true: np.array
    :param y_prob: Predicted probability labels
    :type y_prob: np.array
    :param mode: llm weighting, cosine, weighting, uniform weighting
    :type mode: str
    """
    # For each bin, calculate average probability, and calculate true probability (average positive rating)
    prob_true_per_bin, prob_pred_per_bin = calibration.calibration_curve(y_true=y_true, y_prob=y_prob, n_bins=10)
    plt.plot(prob_pred_per_bin, prob_true_per_bin, marker='o', label="Model")
    plt.plot([0,1],[0,1]) # Representing perfect calibration
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend()
    save_path = RESULTS_DIR / "calibration_curves" / f"calibration_curve_{mode}.png"
    os.makedirs(save_path.parent, exist_ok=True)
    plt.savefig(str(save_path))
    plt.close()

def plot_decision_curve_analysis(y_true: np.array, y_prob: np.array, mode: str):
    """
    Plot decision curve benefits - when only assuming patients above a certain threshold are positive, what is the benefit
    
    :param y_true: Actual labels
    :type y_true: np.array
    :param y_prob: Predicted probability labels
    :type y_prob: np.array
    :param mode: llm weighting, cosine, weighting, uniform weighting
    :type mode: str
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    TP_ASSIGN_ALL_POSITIVE = np.sum(y_true) # Count of true positives
    FP_ASSIGN_ALL_POSITIVE = np.sum(1-y_true)
    N = y_true.shape[0]
    
    def positive_all_benefit(threshold: np.array) -> np.array:
        """
        Helper method to return the benefit attributed with applying the given threshold/penalty classifying all observations as positive
        
        :param threshold: Penalty for false positive
        :type threshold: np.array
        :return: Resulting benefit
        :rtype: np.array
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
    save_path = RESULTS_DIR / "decision_curves" / f"decision_curve_{mode}.png"
    os.makedirs(save_path.parent, exist_ok=True)
    plt.savefig(str(save_path))
    plt.close()

def plot_effective_sample_size_distribution(ess_values: np.array, mode: str):
    """
    Create a histogram of the effective sample sizes observed in the predictions
    
    :param ess_values: Effective sample sizes from predictions
    :type ess_values: np.array
    :param mode: llm weighting, cosine, weighting, uniform weighting
    :type mode: str
    """
    plt.hist(ess_values, bins=100, edgecolor='black')
    plt.xlabel("Effective Sample Size")
    plt.ylabel("Frequency")
    plt.title("Effective Sample Size Distribution")
    plt.axvline(x=int(os.environ['LOW_CONFIDENCE_ESS_THRESHOLD']), color='red', linestyle='--', linewidth=2, label='Low Confidence (<20)')
    plt.legend()
    save_path = RESULTS_DIR / "ess_distributions" / f"ess_distribution_{mode}.png"
    os.makedirs(save_path.parent, exist_ok=True)
    plt.savefig(str(save_path))
    plt.close()
    
def plot_optimal_confusion_matrix(y_true: np.array, y_prob: np.array, mode: str):
    """
    Create confusion matrix for the given probability estimates with the optimal threshold
    
    :param y_true: Actual labels
    :type y_true: np.array
    :param y_prob: Predicted probability labels
    :type y_prob: np.array
    :param mode: llm weighting, cosine, weighting, uniform weighting
    :type mode: str
    """
    false_positive_rates, true_positive_rates, thresholds = sklearn.metrics.roc_curve(y_true=y_true, y_score=y_prob)
    # Find threshold that accomplished peak model performance
    j_statistics = true_positive_rates - false_positive_rates
    threshold = thresholds[np.argmax(j_statistics)]
    # Use threshold to make predictions
    predictions = np.where(y_prob >= threshold, 1, 0)
    matrix = sklearn.metrics.confusion_matrix(y_true=y_true, y_pred=predictions)
    
    # Obtain metric on the confusion matrix
    raveled_matrix = np.ravel(matrix)
    tn, fp, fn, tp = raveled_matrix[0], raveled_matrix[1], raveled_matrix[2], raveled_matrix[3]
    sensitivity = tp / (tp + fn) # proportion of all positive samples correctly flagged
    specificity = tn / (tn + fp) # proportion of all negative samples correctly flagged
    f_score = 2 * tp / (2 * tp + fp + fn)
    positive_likelihood_ratio = sensitivity / (1 - specificity + 1e-9)
    negative_likelihood_ratio = (1 - sensitivity) / (specificity + 1e-9)
    metrics = f"\
Sensitivity: {sensitivity:.2f}\n\
Specificity: {specificity:.2f}\n\
F_Score: {f_score:.2f}\n\
Positive Likelihood Ratio: {positive_likelihood_ratio:.2f}\n\
Negative Likelihood Ratio: {negative_likelihood_ratio:.2f}\
"

    # Create the confusion matrix display with the text report
    display = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=['Non-TRD', 'TRD'])
    display.plot(cmap='Blues')
    plt.figtext(x=0.5, y=-0.25, ha='center', s=metrics)
    plt.title(f'Threshold: {threshold}')
    save_path = RESULTS_DIR / "confusion_matrices" / f"confusion_matrix_{mode}.png"
    os.makedirs(save_path.parent, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()