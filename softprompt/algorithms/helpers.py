
import copy
import numpy as np
from typing import Tuple, Dict, List, Union, Optional

from sklearn.linear_model import LogisticRegressionCV
from .sklearn.linear_model import SoftLogisticRegressionCV
from ..metrics.evaluator import BinaryEvaluator, MulticlassEvaluator


def evaluate_logreg_cv_experiment(
    X_train_full: np.ndarray,
    y_train_full_probs: np.ndarray,  # soft training labels
    X_test: np.ndarray,
    y_test_hard: np.ndarray,         # discrete test labels
    subsample_size: int,
    model_variant: str,              # 'soft' or 'standard'
    Cs: Union[int, List[float], np.ndarray] = 10,
    cv_folds: int = 5,
    solver: str = 'lbfgs',
    max_iter_solver: int = 1000,
    n_jobs_cv: Optional[int] = 8,
    base_random_seed: int = 42,
    bootstrap: bool = True,
    n_trials: int = 50
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:

    if X_train_full.shape[0] != len(y_train_full_probs):
        raise ValueError("X_train_full and y_train_full_probs must have the same number of samples.")
    if model_variant not in ['soft', 'standard', 'gce']:
        raise ValueError("model_variant must be 'soft' or 'standard'.")

    if y_train_full_probs.ndim == 2:
        if y_train_full_probs.shape[1] == 1:
            raise ValueError("For binary tasks, y_train_full_probs should be 1d.")
        evaluator = MulticlassEvaluator()
        multiclass = True
    else:
        evaluator = BinaryEvaluator()
        multiclass = False
    
    # Initialize dictionaries to store metric arrays for each trial
    # The keys will be metric names from evaluator.metrics_to_compute
    # This assumes evaluator.metrics_to_compute is available and populated before this call
    if not hasattr(evaluator, 'metrics_to_compute') or not evaluator.metrics_to_compute:
        # Fallback if metrics_to_compute is not set, use a default or raise error
        # For demonstration, let's assume some common metrics if not available.
        # In a real scenario, BinaryEvaluator should define this.
        print("Warning: evaluator.metrics_to_compute not found or empty. Using default metrics for collection: ['accuracy', 'roc_auc']")
        default_metrics = ['accuracy', 'roc_auc'] # Example
        tr_metric_to_array = {m: np.empty(n_trials) for m in default_metrics}
    else:
        tr_metric_to_array = {m: np.empty(n_trials) for m in evaluator.metrics_to_compute}
    te_metric_to_array = copy.deepcopy(tr_metric_to_array)

    full_idx = np.arange(X_train_full.shape[0])
    
    # Convert full y_train_full_probs to hard labels for evaluation consistency
    if multiclass:
        y_train_full_hard_for_eval = y_train_full_probs.argmax(axis=-1)
    else:
        y_train_full_hard_for_eval = (y_train_full_probs > 0.5).astype(int)

    for i in range(n_trials):
        current_trial_seed = base_random_seed + i
        rng = np.random.default_rng(current_trial_seed)

        # Get sub-sampled indices to use for training
        if bootstrap:
            subsample_idx = rng.choice(full_idx, size=subsample_size, replace=True)
        else:
            subsample_idx = rng.permutation(full_idx)[:subsample_size]

        X_train_subsample = X_train_full[subsample_idx]
        y_train_subsample_probs = y_train_full_probs[subsample_idx]

        # Fit model
        if model_variant == 'soft':
            lg = SoftLogisticRegressionCV(
                Cs=Cs, cv=cv_folds, solver=solver,
                max_iter=max_iter_solver, n_jobs=n_jobs_cv,
                random_state=current_trial_seed,
                scoring='neg_log_loss',
            )
            # Prepare P_train for soft labels (num_samples, num_classes)
            # Assuming binary classification where y_train_subsample_probs are probs of class 1
            if y_train_subsample_probs.ndim == 1:
                 P_train_fit = np.stack(
                    [1 - y_train_subsample_probs, y_train_subsample_probs], axis=1
                )
            elif y_train_subsample_probs.ndim == 2:
                 P_train_fit = y_train_subsample_probs # Assume it's already [prob_class_0, prob_class_1]
            else:
                raise ValueError(
                    "y_train_subsample_probs must be 1D (if binary; probs for class 1)"
                    " or 2D with shape (if multiclass; n_samples, C)."
                )

            lg.fit(X_train_subsample, P_train_fit);

        else:
            lg = LogisticRegressionCV(
                Cs=Cs, cv=cv_folds, solver=solver,
                max_iter=max_iter_solver, n_jobs=n_jobs_cv,
                random_state=current_trial_seed,
                penalty='l2',  # Assuming L2 penalty to align with SoftLogisticRegression
                scoring='neg_log_loss', # To align CV optimization goal with 'cross_entropy'
            )
            # Standard LogisticRegressionCV expects 1D hard labels for y
            if multiclass:
                y_train_fit_hard = y_train_subsample_probs.argmax(axis=-1)
            else:
                y_train_fit_hard = (y_train_subsample_probs > 0.5).astype(int) 
            lg.fit(X_train_subsample, y_train_fit_hard)

        # Evaluate on the full training set
        # The evaluator expects y_true as hard labels and y_prob as predicted probabilities
        tr_metrics_i = evaluator(
            y_true=y_train_full_hard_for_eval, # Hard labels from original full y_train
            y_prob=lg.predict_proba(X_train_full)
        )
        for m, v in tr_metrics_i.items():
            if m in tr_metric_to_array:
                 tr_metric_to_array[m][i] = v

        # Evaluate on the test set
        te_metrics_i = evaluator(
            y_true=y_test_hard, # y_test is already hard labels
            y_prob=lg.predict_proba(X_test)
        )
        for m, v in te_metrics_i.items():
            if m in te_metric_to_array:
                te_metric_to_array[m][i] = v
    
    # Aggregate metrics (train)
    tr_out_agg = {}
    for m, arr in tr_metric_to_array.items():
        if not np.all(np.isnan(arr)): # Ensure array is not all NaNs before mean/std
            tr_out_agg[m] = [np.nanmean(arr), np.nanstd(arr, ddof=1)]
        else:
            tr_out_agg[m] = [np.nan, np.nan]

    # Aggregate metrics (test)
    te_out_agg = {}
    for m, arr in te_metric_to_array.items():
        if not np.all(np.isnan(arr)):
            te_out_agg[m] = [np.nanmean(arr), np.nanstd(arr, ddof=1)]
        else:
            te_out_agg[m] = [np.nan, np.nan]

    return tr_out_agg, te_out_agg