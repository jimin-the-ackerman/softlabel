import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted, _check_y # For P validation
from joblib import Parallel, delayed
from typing import Optional, List, Union, Dict, Tuple # For type hints
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, 
    precision_score, recall_score, log_loss
)

# ---Common Helper Functions ---
def _validate_and_normalize_soft_labels(P_input: np.ndarray, 
                                        atol_val_check: float = 1e-5,
                                        atol_sum_check: float = 1e-5) -> np.ndarray:
    """
    Validates and normalizes a soft label array P.
    Ensures values are [0,1], and rows with non-zero sums are normalized to sum to 1.
    Rows that initially sum to zero are left as all zeros.
    """
    if not isinstance(P_input, np.ndarray):
        P = np.asarray(P_input, dtype=np.float64)
    else:
        P = P_input.astype(np.float64, copy=True) 

    _check_y(P, multi_output=True, y_numeric=True) 
    if P.ndim != 2:
        raise ValueError("P (soft labels) must be a 2D array.")

    if np.any(P < -atol_val_check) or np.any(P > 1 + atol_val_check):
        raise ValueError(
            f"Elements of P must be between 0 and 1 (tolerance {atol_val_check}). Min: {P.min()}, Max: {P.max()}"
        )
    P = np.clip(P, 0.0, 1.0)

    P_row_sums = np.sum(P, axis=1, keepdims=True)
    zero_sum_mask = np.isclose(P_row_sums.flatten(), 0.0, atol=1e-8)
    not_sum_to_one_mask = ~np.isclose(P_row_sums.flatten(), 1.0, atol=atol_sum_check)

    if np.any(zero_sum_mask & not_sum_to_one_mask):
         problematic_indices = np.where(zero_sum_mask & not_sum_to_one_mask)[0]
         raise ValueError(f"Rows of P at indices {problematic_indices} sum to zero and cannot be normalized.")

    needs_normalization_mask = not_sum_to_one_mask & (P_row_sums.flatten() > 1e-8)
    if np.any(needs_normalization_mask):
        P_to_normalize = P[needs_normalization_mask]
        sums_to_normalize = P_row_sums[needs_normalization_mask]
        P[needs_normalization_mask] = P_to_normalize / sums_to_normalize
        
        final_sums_normalized = np.sum(P[needs_normalization_mask], axis=1)
        if not np.allclose(final_sums_normalized, 1.0, atol=1e-8):
            raise ValueError("Normalization of P failed to make some rows sum to 1.")
            
    final_row_sums = np.sum(P, axis=1)
    is_zero_sum_final = np.isclose(final_row_sums, 0.0, atol=1e-8)
    is_one_sum_final = np.isclose(final_row_sums, 1.0, atol=1e-8)
    if not np.all(is_zero_sum_final | is_one_sum_final):
        problem_indices = np.where(~(is_zero_sum_final | is_one_sum_final))[0]
        raise ValueError(f"Rows of P at indices {problem_indices} do not sum to 0.0 or 1.0 after validation.")
    return P


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Computes softmax activations for a batch of logits in a numerically stable way.
    """
    max_logits = np.max(logits, axis=axis, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    sum_exp_logits = np.sum(exp_logits, axis=axis, keepdims=True)
    probabilities = exp_logits / (sum_exp_logits + 1e-12)
    return probabilities


# --- GCE Specific Helper Functions ---
def gce_loss_per_sample(P_true: np.ndarray, 
                        y_pred_proba: np.ndarray, 
                        q: float) -> np.ndarray:
    """
    Computes the Generalized Cross-Entropy (GCE) loss for each sample.
    Returns an array of losses, one for each sample.
    """
    if not (0 < q <= 1):
        raise ValueError("GCE parameter 'q' must be in the range (0, 1].")
    if P_true.shape != y_pred_proba.shape:
        raise ValueError(f"Shape mismatch: P_true {P_true.shape} vs y_pred_proba {y_pred_proba.shape}")

    epsilon = 1e-12
    y_pred_safe = np.clip(y_pred_proba, epsilon, 1.0 - epsilon)
    
    term_pred_q = np.power(y_pred_safe, q)
    sample_class_losses = P_true * (1 - term_pred_q) / q 
    sample_losses = np.sum(sample_class_losses, axis=1)
    return sample_losses


def gce_loss_grad_wrt_logits(P_true: np.ndarray, 
                             logits: np.ndarray, 
                             q: float) -> np.ndarray:
    """
    Computes the gradient of the GCE loss with respect to the logits.
    Returns dL/d(logits) per sample.
    """
    if not (0 < q <= 1):
        raise ValueError("GCE parameter 'q' must be in the range (0, 1].")
    if P_true.shape != logits.shape:
        raise ValueError(f"Shape mismatch: P_true {P_true.shape} vs logits {logits.shape}")

    y_pred_proba = softmax(logits, axis=1)
    epsilon = 1e-12
    y_pred_safe = np.clip(y_pred_proba, epsilon, 1.0 - epsilon)

    y_pred_pow_q = np.power(y_pred_safe, q)
    S_q_per_sample = np.sum(P_true * y_pred_pow_q, axis=1, keepdims=True)
    grad_logits = y_pred_safe * S_q_per_sample - P_true * y_pred_pow_q
    return grad_logits

# --- Soft Logistic Regression (Standard Cross-Entropy) ---
class SoftLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Soft-label logistic regression with L2 penalty and standard cross-entropy loss,
    supporting sample weights.
    """
    def __init__(self, C: float = 1.0, max_iter: int = 100, tol: float = 1e-4,
                 random_state: Optional[Union[int, np.random.RandomState]] = None,
                 fit_intercept: bool = True, penalize_intercept: bool = False,
                 solver: str = 'lbfgs'):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.fit_intercept = fit_intercept
        self.penalize_intercept = penalize_intercept
        self.solver = solver

    def fit(self, X: np.ndarray, P: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        X = check_array(X, accept_sparse=False, dtype=[np.float64, np.float32])
        P_internal = _validate_and_normalize_soft_labels(P)

        if X.shape[0] != P_internal.shape[0]:
            raise ValueError(f"X and P have inconsistent samples: {X.shape[0]} vs {P_internal.shape[0]}")
        
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.shape[0] != X.shape[0]:
                raise ValueError(f"sample_weight shape {sample_weight.shape} inconsistent with X shape {X.shape}")
            if np.any(sample_weight < 0):
                raise ValueError("sample_weight must be non-negative.")
        else:
            sample_weight = np.ones(X.shape[0], dtype=np.float64)

        if self.C <= 0:
            raise ValueError(f"C must be positive; got (C={self.C})")

        self.n_features_in_ = X.shape[1]
        self.num_classes_ = P_internal.shape[1]
        if self.num_classes_ < 2:
            raise ValueError(f"Num classes must be >= 2. Got {self.num_classes_}")
        self.classes_ = np.arange(self.num_classes_) 

        rng = check_random_state(self.random_state)
        param_size = self.num_classes_ * self.n_features_in_ + \
                     (self.num_classes_ if self.fit_intercept else 0)
        initial_params = rng.normal(loc=0, scale=0.01, size=param_size)
        alpha = 1.0 / (2.0 * self.C)

        def _unpack_params(params_unpacked):
            W_flat = params_unpacked[:self.num_classes_ * self.n_features_in_]
            W_unpacked = W_flat.reshape((self.num_classes_, self.n_features_in_))
            if self.fit_intercept:
                b_unpacked = params_unpacked[self.num_classes_ * self.n_features_in_:]
            else:
                b_unpacked = np.zeros(self.num_classes_) 
            return W_unpacked, b_unpacked

        def _loss_and_grad(params_lg):
            W_lg, b_lg = _unpack_params(params_lg)
            logits = X.dot(W_lg.T)
            if self.fit_intercept: 
                logits += b_lg
            
            pred = softmax(logits, axis=1)
            log_pred_manual = np.log(pred + 1e-12) 
            per_sample_ce_loss = -np.sum(P_internal * log_pred_manual, axis=1)
            data_loss = np.sum(sample_weight * per_sample_ce_loss)

            penalty_val = alpha * np.sum(W_lg ** 2) 
            if self.fit_intercept and self.penalize_intercept:
                penalty_val += alpha * np.sum(b_lg ** 2)
            total_loss = data_loss + penalty_val
            
            diff_per_sample = (pred - P_internal)
            weighted_diff = sample_weight[:, np.newaxis] * diff_per_sample
            grad_W = weighted_diff.T.dot(X) + (2.0 * alpha * W_lg)

            if self.fit_intercept:
                grad_b = np.sum(weighted_diff, axis=0)  
                if self.penalize_intercept:
                    grad_b += 2.0 * alpha * b_lg
                grad = np.concatenate([grad_W.ravel(), grad_b])
            else:
                grad = grad_W.ravel()
            return total_loss, grad

        solver_map = {'lbfgs': 'L-BFGS-B', 'cg': 'CG', 'bfgs': 'BFGS', 
                      'newton-cg': 'Newton-CG', 'tnc': 'TNC'}
        scipy_method = solver_map.get(self.solver.lower())
        if scipy_method is None:
            raise NotImplementedError(f"Solver '{self.solver}' not supported.")
        
        minimize_options = {'maxiter': self.max_iter, 'ftol': self.tol, 'disp': False}
        opt_res = minimize(fun=_loss_and_grad, x0=initial_params, method=scipy_method, 
                           jac=True, options=minimize_options)

        self.params_ = opt_res.x
        self.W_, self.b_internal_ = _unpack_params(self.params_)
        self.coef_ = self.W_
        self.intercept_ = self.b_internal_ if self.fit_intercept else np.zeros(self.num_classes_)
        self.n_iter_ = opt_res.nit 
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False, dtype=[np.float64, np.float32])
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expecting {self.n_features_in_}.")
        logits = X.dot(self.coef_.T)
        if self.fit_intercept: 
             logits += self.intercept_
        return softmax(logits, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


def _fit_and_score_slr_fold( # Helper for SoftLogisticRegressionCV
    X_train_fold: np.ndarray, P_train_fold: np.ndarray, 
    X_val_fold: np.ndarray, P_val_fold: np.ndarray,     
    sample_weight_train_fold: Optional[np.ndarray], 
    C_val: float, solver_fold: str, max_iter_fold: int, tol_fold: float,
    random_state_fold: Optional[Union[int, np.random.RandomState]],
    fit_intercept_fold: bool, penalize_intercept_fold: bool,
    scoring_metric_fold: str
) -> float:
    model = SoftLogisticRegression(
        C=C_val, solver=solver_fold, max_iter=max_iter_fold, tol=tol_fold,
        random_state=random_state_fold, fit_intercept=fit_intercept_fold,
        penalize_intercept=penalize_intercept_fold
    )
    model.fit(X_train_fold, P_train_fold, sample_weight=sample_weight_train_fold) 
    pred_proba_val = model.predict_proba(X_val_fold)

    if P_val_fold.shape != pred_proba_val.shape: 
        raise ValueError(f"Shape mismatch: P_val_fold {P_val_fold.shape} vs pred_proba_val {pred_proba_val.shape}")

    y_true_hard_for_scoring = np.argmax(P_val_fold, axis=1)
    y_pred_hard_for_scoring = np.argmax(pred_proba_val, axis=1)
    num_classes = P_val_fold.shape[1]
    class_labels = np.arange(num_classes) 

    score = 0.0
    if scoring_metric_fold in ('cross_entropy', 'neg_log_loss', 'neg_logloss'): 
        try:
            # **CRITICAL FIX**: Always pass 1D hard labels (y_true_hard_for_scoring)
            # to sklearn.metrics.log_loss when used for SCORING.
            # The model itself is trained on soft P_train_fold.
            raw_loss = log_loss(y_true_hard_for_scoring, pred_proba_val, labels=class_labels)
            score = -raw_loss
        except ValueError as e:
            print(f"Warning: log_loss calculation failed in CV fold with C={C_val}: {e}. "
                  f"P_val_fold sample (first row if exists): {P_val_fold[0] if P_val_fold.shape[0]>0 else 'empty'}. "
                  f"pred_proba_val sample (first row if exists): {pred_proba_val[0] if pred_proba_val.shape[0]>0 else 'empty'}. "
                  f"Defaulting score to -inf.")
            score = -float('inf')
    elif scoring_metric_fold == 'accuracy':
        score = accuracy_score(y_true_hard_for_scoring, y_pred_hard_for_scoring)
    elif scoring_metric_fold == 'roc_auc':
         score = roc_auc_score(y_true_hard_for_scoring, pred_proba_val if num_classes > 2 else pred_proba_val[:,1], 
                              multi_class='ovr' if num_classes > 2 else None, 
                              average='macro' if num_classes > 2 else None, labels=class_labels if num_classes > 2 else None)
    elif scoring_metric_fold == 'roc_auc_ovr':
        score = roc_auc_score(y_true_hard_for_scoring, pred_proba_val, multi_class='ovr', average='macro', labels=class_labels)
    elif scoring_metric_fold == 'roc_auc_ovo':
        score = roc_auc_score(y_true_hard_for_scoring, pred_proba_val, multi_class='ovo', average='macro', labels=class_labels)
    elif scoring_metric_fold.startswith('f1_') or \
         scoring_metric_fold.startswith('precision_') or \
         scoring_metric_fold.startswith('recall_'):
        try:
            metric_name, avg_type = scoring_metric_fold.split('_', 1)
            if metric_name == 'f1':
                score = f1_score(y_true_hard_for_scoring, y_pred_hard_for_scoring, labels=class_labels, average=avg_type, zero_division=0)
            elif metric_name == 'precision':
                score = precision_score(y_true_hard_for_scoring, y_pred_hard_for_scoring, labels=class_labels, average=avg_type, zero_division=0)
            elif metric_name == 'recall':
                score = recall_score(y_true_hard_for_scoring, y_pred_hard_for_scoring, labels=class_labels, average=avg_type, zero_division=0)
            else: raise ValueError(f"Unknown metric prefix: {metric_name}")
        except ValueError: raise ValueError(f"Invalid format for metric: {scoring_metric_fold}")
    else:
        raise ValueError(f"Unsupported scoring: {scoring_metric_fold}.")
    return score

class SoftLogisticRegressionCV(BaseEstimator, ClassifierMixin):
    """
    Cross-validation for SoftLogisticRegression with sample weighting.
    """
    def __init__(self, Cs: Union[int, List[float], np.ndarray] = 10, cv: int = 5,
                 max_iter: int = 100, tol: float = 1e-4,
                 random_state: Optional[Union[int, np.random.RandomState]] = None,
                 fit_intercept: bool = True, penalize_intercept: bool = False,
                 scoring: str = 'neg_log_loss', 
                 solver: str = 'lbfgs',
                 n_jobs: Optional[int] = None,
                 borderline_weighting_config: Optional[Dict[str, float]] = None):
        self.Cs = Cs; self.cv = cv; self.max_iter = max_iter; self.tol = tol
        self.random_state = random_state; self.fit_intercept = fit_intercept
        self.penalize_intercept = penalize_intercept; self.scoring = scoring
        self.solver = solver; self.n_jobs = n_jobs
        self.borderline_weighting_config = borderline_weighting_config

    def _calculate_sample_weights(self, P: np.ndarray) -> Optional[np.ndarray]:
        if self.borderline_weighting_config is None or P.shape[1] != 2: 
            return None
        exponent = self.borderline_weighting_config.get('exponent', 1.0)
        min_weight = self.borderline_weighting_config.get('min_weight', 0.0)
        if not (0 <= min_weight <= 1): raise ValueError("min_weight must be [0,1].")
        if exponent < 0: raise ValueError("exponent must be non-negative.")
        s_scores = P[:, 1]
        distance_from_center = np.abs(s_scores - 0.5) 
        scaled_distance_factor = (2 * distance_from_center) ** exponent
        weights = min_weight + (1 - min_weight) * scaled_distance_factor
        return np.clip(weights, 0.0, 1.0)

    def fit(self, X: np.ndarray, P: np.ndarray):
        X = check_array(X, accept_sparse=False, dtype=[np.float64, np.float32])
        P_fit = _validate_and_normalize_soft_labels(P)
        if X.shape[0] != P_fit.shape[0]:
            raise ValueError(f"X {X.shape} and P_fit {P_fit.shape} inconsistent samples.")
        self.n_features_in_ = X.shape[1]
        self.num_classes_ = P_fit.shape[1]
        if self.num_classes_ < 2: raise ValueError(f"Num classes must be >= 2. Got {self.num_classes_}")
        self.classes_ = np.arange(self.num_classes_)
        overall_sample_weights = self._calculate_sample_weights(P_fit)

        if isinstance(self.Cs, int):
            if self.Cs <= 0: raise ValueError("If Cs is int, must be positive.")
            self.Cs_ = np.logspace(-4, 4, self.Cs) 
        else:
            self.Cs_ = np.array(self.Cs, dtype=float)
            if not np.all(self.Cs_ > 0): raise ValueError("All C values must be positive.")
        if len(self.Cs_) == 0: raise ValueError("Cs grid is empty.")

        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        self.scores_: Dict[float, np.ndarray] = {}
        best_avg_score = -float('inf') 
        self.C_: Optional[float] = None 
        fold_job_params = {
            'solver_fold': self.solver, 'max_iter_fold': self.max_iter, 'tol_fold': self.tol,
            'random_state_fold': self.random_state, 'fit_intercept_fold': self.fit_intercept,
            'penalize_intercept_fold': self.penalize_intercept, 'scoring_metric_fold': self.scoring
        }
        for C_val in self.Cs_:
            tasks = []
            for train_idx, val_idx in kf.split(X, P_fit):
                current_sample_weights_train_fold = overall_sample_weights[train_idx] if overall_sample_weights is not None else None
                tasks.append(
                    delayed(_fit_and_score_slr_fold)(
                        X[train_idx], P_fit[train_idx], X[val_idx], P_fit[val_idx],
                        current_sample_weights_train_fold, C_val, **fold_job_params
                    )
                )
            current_C_fold_scores = np.array(Parallel(n_jobs=self.n_jobs)(tasks))
            valid_scores = current_C_fold_scores[np.isfinite(current_C_fold_scores)]
            avg_score_for_C = np.mean(valid_scores) if len(valid_scores) > 0 else -float('inf')
            if len(valid_scores) < len(current_C_fold_scores): 
                print(f"Warning: Some CV folds for C={C_val} had non-finite scores.")
            self.scores_[C_val] = current_C_fold_scores 
            if avg_score_for_C > best_avg_score:
                best_avg_score = avg_score_for_C
                self.C_ = C_val
        if self.C_ is None and len(self.Cs_) > 0: 
             self.C_ = self.Cs_[0] 
             print(f"Warning: Could not determine best C. Defaulting to first C: {self.C_}")
        self.best_estimator_ = SoftLogisticRegression(
            C=self.C_, max_iter=self.max_iter, tol=self.tol, random_state=self.random_state,
            fit_intercept=self.fit_intercept, penalize_intercept=self.penalize_intercept, solver=self.solver
        )
        self.best_estimator_.fit(X, P_fit, sample_weight=overall_sample_weights) 
        self.coef_ = self.best_estimator_.coef_
        self.intercept_ = self.best_estimator_.intercept_
        self.n_iter_ = self.best_estimator_.n_iter_ 
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self); return self.best_estimator_.predict_proba(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self); return self.best_estimator_.predict(X)

# --- GCE Logistic Regression ---
class GCELogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression with Generalized Cross-Entropy (GCE) loss for soft labels,
    supporting sample weights.
    """
    def __init__(self, C: float = 1.0, q_gce: float = 0.7, 
                 max_iter: int = 100, tol: float = 1e-4,
                 random_state: Optional[Union[int, np.random.RandomState]] = None,
                 fit_intercept: bool = True, penalize_intercept: bool = False,
                 solver: str = 'lbfgs'):
        self.C = C
        self.q_gce = q_gce
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.fit_intercept = fit_intercept
        self.penalize_intercept = penalize_intercept
        self.solver = solver

    def fit(self, X: np.ndarray, P: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        X = check_array(X, accept_sparse=False, dtype=[np.float64, np.float32])
        P_internal = _validate_and_normalize_soft_labels(P)
        if X.shape[0] != P_internal.shape[0]:
            raise ValueError(f"X and P inconsistent samples: {X.shape[0]} vs {P_internal.shape[0]}")
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.shape[0] != X.shape[0]:
                raise ValueError(f"sample_weight {sample_weight.shape} inconsistent with X {X.shape}")
            if np.any(sample_weight < 0): raise ValueError("sample_weight must be non-negative.")
        else:
            sample_weight = np.ones(X.shape[0], dtype=np.float64)
        if self.C <= 0: raise ValueError(f"C must be positive; got (C={self.C})")
        if not (0 < self.q_gce <= 1): raise ValueError(f"q_gce must be (0, 1]; got {self.q_gce}")

        self.n_features_in_ = X.shape[1]
        self.num_classes_ = P_internal.shape[1]
        if self.num_classes_ < 2: raise ValueError(f"Num classes must be >= 2. Got {self.num_classes_}")
        self.classes_ = np.arange(self.num_classes_) 
        rng = check_random_state(self.random_state)
        param_size = self.num_classes_ * self.n_features_in_ + \
                     (self.num_classes_ if self.fit_intercept else 0)
        initial_params = rng.normal(loc=0, scale=0.01, size=param_size)
        alpha = 1.0 / (2.0 * self.C) 

        def _unpack_params(params_unpacked):
            W_flat = params_unpacked[:self.num_classes_ * self.n_features_in_]
            W_unpacked = W_flat.reshape((self.num_classes_, self.n_features_in_))
            b_unpacked = params_unpacked[self.num_classes_ * self.n_features_in_:] if self.fit_intercept \
                         else np.zeros(self.num_classes_)
            return W_unpacked, b_unpacked

        def _loss_and_grad(params_lg):
            W_lg, b_lg = _unpack_params(params_lg)
            logits = X.dot(W_lg.T)
            if self.fit_intercept: logits += b_lg
            y_pred_proba_lg = softmax(logits, axis=1)
            
            per_sample_loss_terms = gce_loss_per_sample(P_internal, y_pred_proba_lg, self.q_gce)
            data_loss = np.sum(sample_weight * per_sample_loss_terms)
            penalty_val = alpha * np.sum(W_lg ** 2)
            if self.fit_intercept and self.penalize_intercept: penalty_val += alpha * np.sum(b_lg ** 2)
            total_loss = data_loss + penalty_val
            
            grad_logits_per_sample = gce_loss_grad_wrt_logits(P_internal, logits, self.q_gce)
            weighted_grad_logits = sample_weight[:, np.newaxis] * grad_logits_per_sample
            grad_W = weighted_grad_logits.T.dot(X) + (2.0 * alpha * W_lg)
            
            if self.fit_intercept:
                grad_b = np.sum(weighted_grad_logits, axis=0)  
                if self.penalize_intercept: grad_b += 2.0 * alpha * b_lg
                grad = np.concatenate([grad_W.ravel(), grad_b])
            else:
                grad = grad_W.ravel()
            return total_loss, grad

        solver_map = {'lbfgs': 'L-BFGS-B', 'cg': 'CG', 'bfgs': 'BFGS', 
                      'newton-cg': 'Newton-CG', 'tnc': 'TNC'}
        scipy_method = solver_map.get(self.solver.lower())
        if scipy_method is None: raise NotImplementedError(f"Solver '{self.solver}' not supported.")
        
        minimize_options = {'maxiter': self.max_iter, 'ftol': self.tol, 'disp': False}
        opt_res = minimize(fun=_loss_and_grad, x0=initial_params, method=scipy_method, 
                           jac=True, options=minimize_options)

        self.params_ = opt_res.x
        self.W_, self.b_internal_ = _unpack_params(self.params_)
        self.coef_ = self.W_
        self.intercept_ = self.b_internal_ if self.fit_intercept else np.zeros(self.num_classes_)
        self.n_iter_ = opt_res.nit 
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False, dtype=[np.float64, np.float32])
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expecting {self.n_features_in_}.")
        logits = X.dot(self.coef_.T)
        if self.fit_intercept: logits += self.intercept_
        return softmax(logits, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

# --- CV Helper for GCE Logistic Regression ---
def _fit_and_score_gce_lr_fold(
    X_train_fold: np.ndarray, P_train_fold: np.ndarray, 
    X_val_fold: np.ndarray, P_val_fold: np.ndarray,     
    sample_weight_train_fold: Optional[np.ndarray], 
    C_val: float, q_gce_fold: float, 
    solver_fold: str, max_iter_fold: int, tol_fold: float,
    random_state_fold: Optional[Union[int, np.random.RandomState]],
    fit_intercept_fold: bool, penalize_intercept_fold: bool,
    scoring_metric_fold: str
) -> float:
    model = GCELogisticRegression(
        C=C_val, q_gce=q_gce_fold, 
        solver=solver_fold, max_iter=max_iter_fold, tol=tol_fold,
        random_state=random_state_fold, fit_intercept=fit_intercept_fold,
        penalize_intercept=penalize_intercept_fold
    )
    model.fit(X_train_fold, P_train_fold, sample_weight=sample_weight_train_fold) 
    pred_proba_val = model.predict_proba(X_val_fold)
    # Scoring logic is identical to _fit_and_score_slr_fold, can be refactored
    # For brevity, reusing the logic from _fit_and_score_slr_fold by calling it:
    # This is a conceptual shortcut; in practice, you might duplicate or refactor the scoring.
    # To avoid actual recursive call or direct dependency, let's assume the scoring logic is here.
    
    if P_val_fold.shape != pred_proba_val.shape:
        raise ValueError(f"Shape mismatch in GCE fold: P_val {P_val_fold.shape} vs pred_proba {pred_proba_val.shape}")

    y_true_hard_for_scoring = np.argmax(P_val_fold, axis=1)
    y_pred_hard_for_scoring = np.argmax(pred_proba_val, axis=1)
    num_classes = P_val_fold.shape[1]
    class_labels = np.arange(num_classes)
    score = 0.0
    if scoring_metric_fold in ('cross_entropy', 'neg_log_loss', 'neg_logloss'):
        is_P_val_fold_genuinely_soft = np.any((P_val_fold > 1e-8) & (P_val_fold < 1.0 - 1e-8))
        try:
            y_true_for_logloss = P_val_fold if is_P_val_fold_genuinely_soft or num_classes <=1 else y_true_hard_for_scoring
            raw_loss = log_loss(y_true_for_logloss, pred_proba_val, labels=class_labels)
            score = -raw_loss
        except ValueError as e:
            print(f"Warning: log_loss failed for C={C_val}, q={q_gce_fold}: {e}. Defaulting to -inf.")
            score = -float('inf')
    elif scoring_metric_fold == 'accuracy':
        score = accuracy_score(y_true_hard_for_scoring, y_pred_hard_for_scoring)
    elif scoring_metric_fold == 'roc_auc':
         score = roc_auc_score(y_true_hard_for_scoring, pred_proba_val if num_classes > 2 else pred_proba_val[:,1], 
                              multi_class='ovr' if num_classes > 2 else None, 
                              average='macro' if num_classes > 2 else None, labels=class_labels if num_classes > 2 else None)
    elif scoring_metric_fold == 'roc_auc_ovr':
        score = roc_auc_score(y_true_hard_for_scoring, pred_proba_val, multi_class='ovr', average='macro', labels=class_labels)
    elif scoring_metric_fold == 'roc_auc_ovo':
        score = roc_auc_score(y_true_hard_for_scoring, pred_proba_val, multi_class='ovo', average='macro', labels=class_labels)
    elif scoring_metric_fold.startswith('f1_') or \
         scoring_metric_fold.startswith('precision_') or \
         scoring_metric_fold.startswith('recall_'):
        try:
            metric_name, avg_type = scoring_metric_fold.split('_', 1)
            if metric_name == 'f1': score = f1_score(y_true_hard_for_scoring, y_pred_hard_for_scoring, labels=class_labels, average=avg_type, zero_division=0)
            elif metric_name == 'precision': score = precision_score(y_true_hard_for_scoring, y_pred_hard_for_scoring, labels=class_labels, average=avg_type, zero_division=0)
            elif metric_name == 'recall': score = recall_score(y_true_hard_for_scoring, y_pred_hard_for_scoring, labels=class_labels, average=avg_type, zero_division=0)
            else: raise ValueError(f"Unknown metric prefix: {metric_name}")
        except ValueError: raise ValueError(f"Invalid format for metric: {scoring_metric_fold}")
    else: raise ValueError(f"Unsupported scoring: {scoring_metric_fold}.")
    return score


# --- GCE Logistic Regression CV ---
class GCELogisticRegressionCV(BaseEstimator, ClassifierMixin):
    """
    Cross-validation for GCELogisticRegression with sample weighting.
    """
    def __init__(self, Cs: Union[int, List[float], np.ndarray] = 10, 
                 q_gce: float = 0.7, 
                 cv: int = 5,
                 max_iter: int = 100, tol: float = 1e-4,
                 random_state: Optional[Union[int, np.random.RandomState]] = None,
                 fit_intercept: bool = True, penalize_intercept: bool = False,
                 scoring: str = 'neg_log_loss', 
                 solver: str = 'lbfgs',
                 n_jobs: Optional[int] = None,
                 borderline_weighting_config: Optional[Dict[str, float]] = None):
        self.Cs = Cs; self.q_gce = q_gce; self.cv = cv; self.max_iter = max_iter; self.tol = tol
        self.random_state = random_state; self.fit_intercept = fit_intercept
        self.penalize_intercept = penalize_intercept; self.scoring = scoring
        self.solver = solver; self.n_jobs = n_jobs
        self.borderline_weighting_config = borderline_weighting_config

    def _calculate_sample_weights(self, P: np.ndarray) -> Optional[np.ndarray]:
        # Identical to SoftLogisticRegressionCV._calculate_sample_weights
        if self.borderline_weighting_config is None or P.shape[1] != 2: return None
        exponent = self.borderline_weighting_config.get('exponent', 1.0)
        min_weight = self.borderline_weighting_config.get('min_weight', 0.0)
        if not (0 <= min_weight <= 1): raise ValueError("min_weight must be [0,1].")
        if exponent < 0: raise ValueError("exponent must be non-negative.")
        s_scores = P[:, 1]
        distance_from_center = np.abs(s_scores - 0.5) 
        scaled_distance_factor = (2 * distance_from_center) ** exponent
        weights = min_weight + (1 - min_weight) * scaled_distance_factor
        return np.clip(weights, 0.0, 1.0)

    def fit(self, X: np.ndarray, P: np.ndarray):
        X = check_array(X, accept_sparse=False, dtype=[np.float64, np.float32])
        P_fit = _validate_and_normalize_soft_labels(P)
        if X.shape[0] != P_fit.shape[0]:
            raise ValueError(f"X {X.shape} and P_fit {P_fit.shape} inconsistent samples.")
        self.n_features_in_ = X.shape[1]
        self.num_classes_ = P_fit.shape[1]
        if self.num_classes_ < 2: raise ValueError(f"Num classes must be >= 2. Got {self.num_classes_}")
        self.classes_ = np.arange(self.num_classes_)
        overall_sample_weights = self._calculate_sample_weights(P_fit)

        if isinstance(self.Cs, int):
            if self.Cs <= 0: raise ValueError("If Cs is int, must be positive.")
            self.Cs_ = np.logspace(-4, 4, self.Cs) 
        else:
            self.Cs_ = np.array(self.Cs, dtype=float)
            if not np.all(self.Cs_ > 0): raise ValueError("All C values must be positive.")
        if len(self.Cs_) == 0: raise ValueError("Cs grid is empty.")

        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        self.scores_: Dict[float, np.ndarray] = {}
        best_avg_score = -float('inf') 
        self.C_: Optional[float] = None 
        fold_job_params = {
            'q_gce_fold': self.q_gce, # Pass q_gce
            'solver_fold': self.solver, 'max_iter_fold': self.max_iter, 'tol_fold': self.tol,
            'random_state_fold': self.random_state, 'fit_intercept_fold': self.fit_intercept,
            'penalize_intercept_fold': self.penalize_intercept, 'scoring_metric_fold': self.scoring
        }
        for C_val in self.Cs_:
            tasks = []
            for train_idx, val_idx in kf.split(X, P_fit):
                current_sample_weights_train_fold = overall_sample_weights[train_idx] if overall_sample_weights is not None else None
                tasks.append(
                    delayed(_fit_and_score_gce_lr_fold)( # Use GCE helper
                        X[train_idx], P_fit[train_idx], X[val_idx], P_fit[val_idx],
                        current_sample_weights_train_fold, C_val, **fold_job_params
                    )
                )
            current_C_fold_scores = np.array(Parallel(n_jobs=self.n_jobs)(tasks))
            valid_scores = current_C_fold_scores[np.isfinite(current_C_fold_scores)]
            avg_score_for_C = np.mean(valid_scores) if len(valid_scores) > 0 else -float('inf')
            if len(valid_scores) < len(current_C_fold_scores): 
                print(f"Warning: Some CV folds for C={C_val}, q={self.q_gce} had non-finite scores.")
            self.scores_[C_val] = current_C_fold_scores 
            if avg_score_for_C > best_avg_score:
                best_avg_score = avg_score_for_C
                self.C_ = C_val
        if self.C_ is None and len(self.Cs_) > 0: 
             self.C_ = self.Cs_[0] 
             print(f"Warning: Could not determine best C. Defaulting to first C: {self.C_}")
        self.best_estimator_ = GCELogisticRegression( # Use GCELogisticRegression
            C=self.C_, q_gce=self.q_gce, # Pass q_gce
            max_iter=self.max_iter, tol=self.tol, random_state=self.random_state,
            fit_intercept=self.fit_intercept, penalize_intercept=self.penalize_intercept, 
            solver=self.solver
        )
        self.best_estimator_.fit(X, P_fit, sample_weight=overall_sample_weights) 
        self.coef_ = self.best_estimator_.coef_
        self.intercept_ = self.best_estimator_.intercept_
        self.n_iter_ = self.best_estimator_.n_iter_ 
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self); return self.best_estimator_.predict_proba(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self); return self.best_estimator_.predict(X)

