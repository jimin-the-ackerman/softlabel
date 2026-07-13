import numpy as np
from typing import Union, Sequence


def expected_calibration_error(
    y_true: Union[Sequence[int], np.ndarray],
    y_pred_probs: Union[Sequence[Sequence[float]], np.ndarray],
    n_bins: int = 10
) -> float:
    """
    Computes the Expected Calibration Error (ECE) for multiclass classification.
    
    For multiclass tasks, ECE is calculated by:
    1. Taking the maximum probability (confidence) for each prediction
    2. Binning samples by their confidence values
    3. Computing the difference between accuracy and confidence in each bin
    4. Weighting by bin size and summing

    Parameters:
        y_true (array-like): True class labels (0 to num_classes-1).
        y_pred_probs (array-like): Predicted probabilities of shape (n_samples, n_classes).
        n_bins (int): Number of bins to use for calibration.

    Returns:
        float: The Expected Calibration Error.
    """
    y_true = np.asarray(y_true)
    y_pred_probs = np.asarray(y_pred_probs)

    # Ensure inputs are valid
    if len(y_true) != len(y_pred_probs):
        raise ValueError("y_true and y_pred_probs must have the same number of samples.")
    if y_pred_probs.ndim != 2:
        raise ValueError("y_pred_probs must be a 2D array of shape (n_samples, n_classes).")
    if not np.all((y_pred_probs >= 0) & (y_pred_probs <= 1)):
        raise ValueError("y_pred_probs must contain probabilities in the range [0, 1].")
    if not np.allclose(y_pred_probs.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Probabilities must sum to 1 for each sample.")

    # Get predicted classes and their confidences
    predicted_classes = np.argmax(y_pred_probs, axis=1)
    confidences = np.max(y_pred_probs, axis=1)
    
    # Check if predicted classes are within valid range
    num_classes = y_pred_probs.shape[1]
    if np.any((y_true < 0) | (y_true >= num_classes)):
        raise ValueError(f"y_true contains invalid class labels. Expected 0 to {num_classes-1}.")

    # Create bins based on confidence values
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges, right=True) - 1

    ece = 0.0
    for i in range(n_bins):
        bin_mask = bin_indices == i  # which data points belong to the i-th bin
        bin_size = np.sum(bin_mask)  # how many data points belong to the i-th bin
        if bin_size > 0:
            # Accuracy in this bin: fraction of correct predictions
            bin_accuracy = np.mean(predicted_classes[bin_mask] == y_true[bin_mask])
            # Confidence in this bin: average confidence of predictions
            bin_confidence = np.mean(confidences[bin_mask])
            # weighted average of bin-wise calibration error
            ece += (bin_size / len(y_true)) * abs(bin_accuracy - bin_confidence)

    return ece


def expected_calibration_error_per_class(
    y_true: Union[Sequence[int], np.ndarray],
    y_pred_probs: Union[Sequence[Sequence[float]], np.ndarray],
    n_bins: int = 10,
    average: str = 'macro'
) -> Union[float, np.ndarray]:
    """
    Computes the Expected Calibration Error (ECE) for each class separately.
    
    This computes ECE for each class in a one-vs-rest manner, then averages
    the results. This can be useful when you want to understand calibration
    per class rather than overall.

    Parameters:
        y_true (array-like): True class labels (0 to num_classes-1).
        y_pred_probs (array-like): Predicted probabilities of shape (n_samples, n_classes).
        n_bins (int): Number of bins to use for calibration.
        average (str): How to average the per-class ECE values.
                      'macro': Simple average across classes
                      'weighted': Weighted average by class frequency

    Returns:
        float or np.ndarray: The averaged ECE value(s).
    """
    y_true = np.asarray(y_true)
    y_pred_probs = np.asarray(y_pred_probs)
    num_classes = y_pred_probs.shape[1]
    
    # Compute ECE for each class
    ece_per_class = np.zeros(num_classes)
    class_counts = np.zeros(num_classes)
    
    for class_idx in range(num_classes):
        # Create binary labels for this class (1 if true class, 0 otherwise)
        binary_labels = (y_true == class_idx).astype(int)
        # Get probabilities for this class
        class_probs = y_pred_probs[:, class_idx]
        
        # Compute ECE for this class using the binary ECE function
        from .binary import expected_calibration_error as binary_ece
        ece_per_class[class_idx] = binary_ece(binary_labels, class_probs, n_bins)
        class_counts[class_idx] = np.sum(binary_labels)
    
    # Average the per-class ECE values
    if average == 'macro':
        return np.mean(ece_per_class)
    elif average == 'weighted':
        weights = class_counts / np.sum(class_counts)
        return np.average(ece_per_class, weights=weights)
    else:
        raise ValueError("average must be 'macro' or 'weighted'.")


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate example data
    n_samples = 1000
    n_classes = 3
    
    # True labels
    y_true = np.random.randint(0, n_classes, n_samples)
    
    # Predicted probabilities (well-calibrated)
    y_pred_probs = np.random.dirichlet([1, 1, 1], n_samples)
    
    # Make some predictions overconfident (poorly calibrated)
    for i in range(n_samples):
        true_class = y_true[i]
        # Increase confidence for true class
        y_pred_probs[i, true_class] *= 1.5
        # Renormalize
        y_pred_probs[i] /= y_pred_probs[i].sum()
    
    # Test the functions
    ece = expected_calibration_error(y_true, y_pred_probs, n_bins=10)
    ece_macro = expected_calibration_error_per_class(y_true, y_pred_probs, n_bins=10, average='macro')
    ece_weighted = expected_calibration_error_per_class(y_true, y_pred_probs, n_bins=10, average='weighted')
    
    print(f"Multiclass ECE: {ece:.4f}")
    print(f"Per-class ECE (macro): {ece_macro:.4f}")
    print(f"Per-class ECE (weighted): {ece_weighted:.4f}") 