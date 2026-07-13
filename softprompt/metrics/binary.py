import numpy as np
from typing import Union, Sequence


def expected_calibration_error(
    y_true: Union[Sequence[int], np.ndarray],
    y_pred: Union[Sequence[float], np.ndarray],
    n_bins: int = 10
) -> float:
    """
    Computes the Expected Calibration Error (ECE) between true labels and predicted probabilities.

    Parameters:
        y_true (array-like): True binary labels (0 or 1).
        y_pred (array-like): Predicted probabilities for the positive class.
        n_bins (int): Number of bins to use for calibration.

    Returns:
        float: The Expected Calibration Error.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Ensure inputs are valid
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if not np.all((y_pred >= 0) & (y_pred <= 1)):
        raise ValueError("y_pred must contain probabilities in the range [0, 1].")

    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred, bin_edges, right=True) - 1

    ece = 0.0
    for i in range(n_bins):
        bin_mask = bin_indices == i  # which data points belong to the i-th bin
        bin_size = np.sum(bin_mask)  # how many data points belong to the i-th bin
        if bin_size > 0:
            bin_accuracy = np.mean(y_true[bin_mask])
            bin_confidence = np.mean(y_pred[bin_mask])
            # weighted average of bin-wise calibration error
            ece += (bin_size / len(y_true)) * abs(bin_accuracy - bin_confidence)

    return ece


if __name__ == "__main__":
    # Example usage
    y_true = [0, 1, 1, 0, 1, 0, 1, 0, 1, 0]
    y_pred = [0.1, 0.9, 0.8, 0.4, 0.95, 0.2, 0.85, 0.3, 0.7, 0.1]
    n_bins = 10

    ece = expected_calibration_error(y_true, y_pred, n_bins)
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
