import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import average_precision_score
from typing import Optional, List, Dict

from .binary import expected_calibration_error


class _EvaluatorBase:
    _default_metrics = (
        "accuracy", "precision", "recall", "f1_score", "roc_auc", "auprc", "ece"
    )
    @staticmethod
    def is_valid_y_true(y_true: np.ndarray) -> bool:
        return isinstance(y_true, np.ndarray) and y_true.ndim == 1
    
    @staticmethod
    def is_valid_y_prob(y_prob: np.ndarray) -> bool:
        return isinstance(y_prob, np.ndarray) and y_prob.ndim == 2


class MulticlassEvaluator(_EvaluatorBase):
    _supported_averaging_methods = ('macro', 'weighted')
    def __init__(
        self,
        metrics_to_compute: Optional[List[str]] = None,
        average: str = "macro",
        ) -> None:
    
        self.metrics_to_compute = metrics_to_compute or self._default_metrics
        if average not in self._supported_averaging_methods:
            raise ValueError(
                f"'{average}' is not supported. Choose one of {self._supported_averaging_methods}"
            )
        self.average = average
        self.results = []  # where the results are stored

    def evaluate(self, y_true: np.ndarray, y_prob: np.ndarray, name: str) -> None:
        metrics = self.__call__(y_true, y_prob)
        self.results.append((name, metrics))

    def __call__(self, y_true: np.ndarray, y_prob: np.ndarray):
        
        # Ensure y_true is a 1D array
        if not self.is_valid_y_true(y_true):
            raise ValueError("y_true must be a 1D numpy array")
        
        # Ensure y_prob is a 2D numpy array
        if not self.is_valid_y_prob(y_prob):
            raise ValueError("y_prob must be a 2D numpy array")
        
        # Compute predicted labels
        y_pred = np.argmax(y_prob, axis=1)

        # Compute metrics
        metrics = {}
        if "accuracy" in self.metrics_to_compute:
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
        if "precision" in self.metrics_to_compute:
            metrics["precision"] = precision_score(y_true, y_pred, average=self.average, zero_division=0)
        if "recall" in self.metrics_to_compute:
            metrics["recall"] = recall_score(y_true, y_pred, average=self.average)
        if "f1_score" in self.metrics_to_compute:
            metrics["f1_score"] = f1_score(y_true, y_pred, average=self.average)
        if "roc_auc" in self.metrics_to_compute:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob, average=self.average, multi_class="ovr")
        if "auprc" in self.metrics_to_compute:
            # Compute AUPRC for each class in a one-vs-rest manner (higher the better)
            auprc_scores = np.empty(y_prob.shape[1])
            counts = np.empty_like(auprc_scores)
            for j in range(y_prob.shape[1]):
                y_true_binary = (y_true == j).astype(int)
                auprc_scores[j] = \
                    average_precision_score(y_true_binary, y_prob[:, j])
                counts[j] = y_true_binary.sum()
            if self.average == "weighted":
                metrics["auprc"] = np.sum(counts / counts.sum() * auprc_scores)
            elif self.average == "macro":
                metrics["auprc"] = auprc_scores.mean()
        if "ece" in self.metrics_to_compute:
            # Compute ECE for each class in a one-vs-rest manner (lower the better)
            ece_scores = np.zeros(y_prob.shape[1])
            counts = np.zeros_like(ece_scores)
            for j in range(y_prob.shape[1]):
                y_true_binary = (y_true == j).astype(int)
                ece_scores[j] = \
                    expected_calibration_error(y_true_binary, y_prob[:, j])
                counts = y_true_binary.sum()
            if self.average == "weighted":
                metrics["ece"] = np.sum(counts / counts.sum() * ece_scores)
            elif self.average == "macro":
                metrics["ece"] = ece_scores.mean()

        return metrics  # dict


class BinaryEvaluator(_EvaluatorBase):
    def __init__(
        self,
        threshold: float = 0.5,
        metrics_to_compute: Optional[List[str]] = None,
        ) -> None:
        """
        Initialize the Evaluator with optional threshold and metrics to compute.
        
        Args:
            threshold (float): Threshold for binary classification (default: 0.5).
            metrics_to_compute (list): List of metric names to compute (default: all metrics).
        """
        self.threshold = threshold
        self.metrics_to_compute = metrics_to_compute or self._default_metrics
        self.results = []  # tuples will be stored

    def evaluate(self, y_true: np.ndarray, y_prob: np.ndarray, name: str) -> None:
        metrics = self.__call__(y_true, y_prob)
        self.results.append((name, metrics))

    def __call__(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        
        # Ensure y_true is a 1D array
        if not self.is_valid_y_true(y_true):
            raise ValueError("y_true must be a 1D numpy array")
        
        # Ensure y_prob is a 2D numpy array
        if not self.is_valid_y_prob(y_prob):
            raise ValueError("y_prob must be a 2D numpy array")

        # Compute predicted labels
        y_pred = (y_prob[:, 1] >= self.threshold).astype(int)

        # Compute metrics
        metrics = {}
        if "accuracy" in self.metrics_to_compute:
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
        if "precision" in self.metrics_to_compute:
            metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        if "recall" in self.metrics_to_compute:
            metrics["recall"] = recall_score(y_true, y_pred)
        if "f1_score" in self.metrics_to_compute:
            metrics["f1_score"] = f1_score(y_true, y_pred)
        if "roc_auc" in self.metrics_to_compute:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
        if "auprc" in self.metrics_to_compute:
            metrics["auprc"] = average_precision_score(y_true, y_prob[:, 1])
        if "ece" in self.metrics_to_compute:
            metrics["ece"] = expected_calibration_error(y_true, y_prob[:, 1])
            
        return metrics
