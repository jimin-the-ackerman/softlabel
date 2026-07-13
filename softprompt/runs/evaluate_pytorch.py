import os
import sys
sys.path.insert(0, os.path.abspath("./"))


import logging
from rich.logging import RichHandler

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F # Added for one_hot

import argparse # For command-line arguments
import numpy as np
import matplotlib.pyplot as plt
import json # Added for loading jsonl
import time # Added for timing

from rich.console import Console
from rich.table import  Table
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, Subset

from softprompt.utils import generate_unique_identifier
from softprompt.utils import write_argparse_args_to_yaml
from softprompt.datasets.loaders import (load_oracle_data,
                                         load_synthetic_data)


DATA_TO_TRAIN_SIZE = {
    "imdb": 25_000,
    "emotion": 89_832,
    "subj": 8_000,
}


EMBEDDING_MODEL_NAME = "openai/text-embedding-3-small"
DEFAULT_EMBEDDING_DIM = 1536


class LogisticRegression(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)  # for KLDivLoss compatibility

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        logits = self.linear(x)
        log_probs = self.log_softmax(logits)
        return log_probs


class SimpleMLP(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.layer_1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_dim, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)  # for KLDivLoss compatibility

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.relu(self.layer_1(x))
        logits = self.layer_2(x)
        log_probs = self.log_softmax(logits)
        return log_probs


def train_model(model: nn.Module,
                dataloader: DataLoader,
                optimizer: optim.Optimizer,
                criterion: Union[nn.KLDivLoss, nn.NLLLoss],
                device: str = 'cuda',
                epochs: int = 10):
    model.train()
    model.to(device)
    total_steps = len(dataloader) * epochs
    current_step = 0

    console = Console()
    logger.info(f"Starting training for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(epochs):
        running_loss = 0.0
        epoch_start_time = time.time()
        for i, (inputs, targets) in enumerate(dataloader):
            current_step += 1
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            log_probs = model(inputs)
            loss = criterion(log_probs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Print progress less frequently for large datasets
            if max(1, len(dataloader) // args.log_interval) == 0:
                logger.info(f'\rEpoch [{epoch+1:>3}/{epochs:>3}], Step [{i+1:>6}/{len(dataloader):>6}], Loss: {loss.item():.4f}')

        epoch_duration = time.time() - epoch_start_time
        epoch_loss = running_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1:>3} completed in {epoch_duration:.2f}s. Average Loss: {epoch_loss:.4f}")

    total_training_time = time.time() - start_time
    logger.info(f'Finished Training in {total_training_time:.2f}s')



import torch
from typing import Optional, Dict, List, Any

def calculate_binary_positive_class_calibration(
    all_probs: torch.FloatTensor,
    all_true_labels: torch.LongTensor,
    num_bins: Optional[int] = 10
) -> Dict[str, Any]:
    """
    Calculates calibration metrics for a binary classification task, focusing
    on the calibration of the predicted probability of the positive class (P(class 1)).

    The output dictionary structure matches 'calculate_calibration_metrics', but
    the interpretations are specific to P(class 1) calibration:
    - 'ece': Expected Calibration Error for P(class 1) vs. observed fraction of positives.
    - 'bin_confidences': Average P(class 1) in each bin.
    - 'bin_accuracies': Observed fraction of true positives in each bin.
    - 'bin_counts': Number of samples in each bin.
    - 'avg_confidence': Overall average of P(class 1) across all samples.
    - 'overall_accuracy': Standard classification accuracy of the model, assuming
                          prediction of class 1 if P(class 1) >= 0.5.

    Args:
        all_probs (torch.FloatTensor): A 2D tensor of shape (N, 2), where
            all_probs[:, 0] are probabilities for class 0 and
            all_probs[:, 1] are probabilities for class 1.
        all_true_labels (torch.LongTensor): A 1D tensor of shape (N,)
            containing the true labels (0 or 1).
        num_bins (Optional[int]): The number of bins to use for the reliability diagram.
            Defaults to 10.

    Returns:
        Dict[str, Any]: A dictionary with calibration metrics.
    """
    num_samples = len(all_true_labels)
    if num_samples == 0:
        return {
            'ece': 0.0,
            'bin_confidences': [],
            'bin_accuracies': [],
            'bin_counts': [],
            'avg_confidence': 0.0,
            'overall_accuracy': 0.0
        }

    # Validate the input shape for binary task
    if not (all_probs.ndim == 2 and all_probs.shape[1] == 2):
        raise ValueError(
            f"For binary_positive_class_calibration, all_probs is expected to be of shape (N, 2), "
            f"but got shape {all_probs.shape}"
        )

    # Scores for binning are the probabilities of the positive class (class 1)
    # These will become the basis for 'bin_confidences' and 'avg_confidence'
    p_class1_scores = all_probs[:, 1]

    # For 'bin_accuracies', we need the observed fraction of positive labels in each bin
    is_positive_class_true = (all_true_labels.to(p_class1_scores.device) == 1).float()

    # For 'overall_accuracy', we derive standard binary predictions
    # (predict class 1 if P(class 1) >= 0.5, else class 0)
    binary_predictions = (p_class1_scores >= 0.5).long()
    model_correctness = binary_predictions.eq(all_true_labels.to(binary_predictions.device)).float()

    # Ensure bin_boundaries are on the same device
    bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=p_class1_scores.device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece_sum_weighted_diffs = torch.zeros(1, device=p_class1_scores.device)
    # These lists will directly populate the output dictionary
    output_bin_confidences: List[float] = []
    output_bin_accuracies: List[float] = []
    output_bin_counts: List[int] = []

    for i in range(num_bins):
        in_bin = (p_class1_scores > bin_lowers[i]) & (p_class1_scores <= bin_uppers[i])
        
        count_in_bin = in_bin.sum()
        output_bin_counts.append(count_in_bin.item())

        if count_in_bin.item() > 0:
            # 'bin_confidences' = average P(class 1) in this bin
            avg_p_class1_in_bin = p_class1_scores[in_bin].mean()
            # 'bin_accuracies' = observed fraction of true positives in this bin
            fraction_positives_in_bin = is_positive_class_true[in_bin].mean()
            
            ece_sum_weighted_diffs += count_in_bin * torch.abs(avg_p_class1_in_bin - fraction_positives_in_bin)
            
            output_bin_confidences.append(avg_p_class1_in_bin.item())
            output_bin_accuracies.append(fraction_positives_in_bin.item())
        else:
            output_bin_confidences.append(0.0)
            output_bin_accuracies.append(0.0)

    final_ece = ece_sum_weighted_diffs / num_samples if num_samples > 0 else torch.tensor(0.0, device=ece_sum_weighted_diffs.device)

    # Overall metrics
    if num_samples > 0:
        # 'avg_confidence' = overall average of P(class 1)
        avg_p_class1_overall = p_class1_scores.mean().item()
        # 'overall_accuracy' = standard model classification accuracy
        overall_model_accuracy = model_correctness.mean().item()
    else:
        avg_p_class1_overall = 0.0
        overall_model_accuracy = 0.0

    return {
        'ece': final_ece.item(),
        'bin_confidences': output_bin_confidences,
        'bin_accuracies': output_bin_accuracies,
        'bin_counts': output_bin_counts,
        'avg_confidence': avg_p_class1_overall,
        'overall_accuracy': overall_model_accuracy
    }


def calculate_calibration_metrics(all_probs: torch.FloatTensor,
                                  all_true_labels: torch.LongTensor,
                                  num_bins: Optional[int] = 10):
    """
    Calculates ECE and data for reliability diagrams.
    Works for both multiclass and binary classification tasks.

    For binary classification, all_probs MUST be a 2D tensor of shape (N, 2),
    where all_probs[:, 0] are probabilities for class 0 and
    all_probs[:, 1] are probabilities for class 1.

    Confidences are calculated as the probability of the predicted class.
    True labels for binary should be 0 or 1.
    """
    num_samples = len(all_true_labels)
    if num_samples == 0:
        return {
            'ece': 0.0,
            'bin_confidences': [],
            'bin_accuracies': [],
            'bin_counts': [],
            'avg_confidence': 0.0,
            'overall_accuracy': 0.0
        }

    # Ensure true labels are on the same device as predictions for comparison
    confidences, predictions = torch.max(all_probs, dim=1)
    correctness = predictions.eq(all_true_labels.to(predictions.device)).float()

    # Ensure bin_boundaries are on the same device as confidences
    bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=confidences.device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # Ensure ece tensor is initialized on the correct device
    ece = torch.zeros(1, device=confidences.device)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(num_bins):
        # Binning based on confidence: (lower_bound, upper_bound]
        in_bin = (confidences > bin_lowers[i]) & (confidences <= bin_uppers[i])
        
        count_in_bin = in_bin.sum() # Tensor, number of samples in the current bin
        bin_counts.append(count_in_bin.item())

        if count_in_bin.item() > 0:
            accuracy_in_bin = correctness[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            # ECE contribution: |Bm| * |acc(Bm) - conf(Bm)|
            ece += count_in_bin * torch.abs(avg_confidence_in_bin - accuracy_in_bin)
            bin_accuracies.append(accuracy_in_bin.item())
            bin_confidences.append(avg_confidence_in_bin.item())
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)

    # Final ECE is the weighted average, divided by total number of samples
    ece = ece / num_samples if num_samples > 0 else torch.tensor(0.0, device=ece.device)

    avg_conf_val = confidences.mean().item() if num_samples > 0 else 0.0
    overall_acc_val = correctness.mean().item() if num_samples > 0 else 0.0
    
    # Ensure avg_confidence and overall_accuracy are calculated only if num_samples > 0
    # to avoid issues with empty confidences/correctness tensors if they were somehow formed.
    # The initial num_samples check largely handles this, but this is an extra safeguard.
    if num_samples > 0:
        avg_conf_val = confidences.mean().item()
        overall_acc_val = correctness.mean().item()
    else:
        avg_conf_val = 0.0
        overall_acc_val = 0.0

    return {
        'ece': ece.item(),
        'bin_confidences': bin_confidences,
        'bin_accuracies': bin_accuracies,
        'bin_counts': bin_counts,
        'avg_confidence': avg_conf_val,
        'overall_accuracy': overall_acc_val
    }


@torch.no_grad()
def evaluate_model(model: nn.Module,
                   dataloader: DataLoader,
                   device: str,
                   num_classes: int,
                   num_bins: Optional[int] = 10):
    """Evaluates the model on the test set and calculates calibration."""
    model.eval()
    model.to(device)

    all_log_probs = []
    all_true_labels = []
    all_predicted_labels = [] # Store predicted labels for F1 score

    eval_criterion = nn.NLLLoss(reduction='sum')
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, hard_labels in dataloader:
        inputs = inputs.to(device)
        hard_labels = hard_labels.to(device)

        log_probs = model(inputs)
        loss = eval_criterion(log_probs, hard_labels)
        total_loss += loss.item()

        _, predicted_labels = torch.max(log_probs, 1)
        correct_predictions += (predicted_labels == hard_labels).sum().item()
        total_samples += hard_labels.size(0)

        all_log_probs.append(log_probs)
        all_true_labels.append(hard_labels)
        all_predicted_labels.append(predicted_labels) # Append predicted labels

    if total_samples == 0:
        return None

    # Concatenate results on the device
    all_log_probs_tensor = torch.cat(all_log_probs, dim=0)
    all_true_labels_tensor = torch.cat(all_true_labels, dim=0)
    all_predicted_labels_tensor = torch.cat(all_predicted_labels, dim=0)

    all_probs_tensor = torch.exp(all_log_probs_tensor)

    accuracy = correct_predictions / total_samples
    avg_nll = total_loss / total_samples

    task = 'multiclass' if args.data in ('emotion', 'agnews') else 'binary'
    if task == 'multiclass':
        calibration_results = calculate_calibration_metrics(all_probs_tensor, all_true_labels_tensor, num_bins=num_bins)
    else:
        calibration_results = calculate_binary_positive_class_calibration(
            all_probs_tensor, all_true_labels_tensor, num_bins=num_bins
        )

    true_one_hot = F.one_hot(all_true_labels_tensor, num_classes=num_classes).float()
    brier_score = torch.mean(torch.sum((all_probs_tensor - true_one_hot)**2, dim=1)).item()

    # --- Calculate Macro F1 Score ---
    # Move tensors to CPU and convert to NumPy for scikit-learn
    y_true_np = all_true_labels_tensor.cpu().numpy()
    y_pred_np = all_predicted_labels_tensor.cpu().numpy()
    y_prob_np = all_probs_tensor.cpu().numpy()

    # Calculate Macro F1
    from sklearn.metrics import f1_score
    macro_f1 = f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    
    # Calculate auc_roc
    from sklearn.metrics import roc_auc_score
    if task == 'multiclass':
        auroc = roc_auc_score(y_true_np, y_prob_np, average='macro', multi_class="ovr")
    else:
        auroc = roc_auc_score(y_true_np, y_prob_np[:, 1])
    # ---------------------------------

    print(f'Evaluation Results:')
    print(f'  Accuracy: {accuracy:.4f}')
    print(f'  Macro F1: {macro_f1:.4f}') # Print Macro F1
    print(f'  Macro AUROC: {auroc:.4f}')
    print(f'  Avg NLL: {avg_nll:.4f}')
    print(f'  Brier Score: {brier_score:.4f}')
    print(f'  ECE ({num_bins} bins): {calibration_results["ece"]:.4f}')
    print(f'  Avg Confidence: {calibration_results["avg_confidence"]:.4f}')

    calibration_results_cpu = {
        'ece': calibration_results['ece'],
        'bin_confidences': calibration_results['bin_confidences'],
        'bin_accuracies': calibration_results['bin_accuracies'],
        'bin_counts': calibration_results['bin_counts'],
    }

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1, # Add Macro F1 to results
        'avg_nll': avg_nll,
        'brier_score': brier_score,
        'calibration_results': calibration_results_cpu
    }


def plot_reliability_diagram(calibration_results: Dict[str, float],
                             title="Reliability Diagram"):
    """Plots the reliability diagram."""
    fig, ax = plt.subplots(figsize=(6, 6))
    # Ensure these keys exist and data is valid before plotting
    bin_confidences = calibration_results.get('bin_confidences', [])
    bin_accuracies = calibration_results.get('bin_accuracies', [])
    bin_counts = calibration_results.get('bin_counts', [])
    num_bins = len(bin_confidences)

    if num_bins == 0:
        print("Warning: No data to plot in reliability diagram.")
        plt.close(fig) # Close empty figure
        return

    bins = np.arange(num_bins) + 1
    bar_width = 0.9 / num_bins
    bar_centers = np.linspace(0, 1, num_bins * 2 + 1)[1::2]

    gap_bars = ax.bar(bar_centers, bin_confidences, width=bar_width, edgecolor='black', color='lightcoral', label='Confidence')
    acc_bars = ax.bar(bar_centers, bin_accuracies, width=bar_width, edgecolor='black', color='cornflowerblue', alpha=0.7, label='Accuracy')

    ax.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Perfect Calibration')
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle=':')

    ece = calibration_results.get('ece', float('nan'))
    ax.text(0.05, 0.80, f"ECE = {ece:.4f}", transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    # Consider saving the plot instead of showing directly
    # plt.savefig("reliability_diagram.png")
    plt.show()

    return fig


def main(args: argparse.Namespace):
    
    torch.manual_seed(args.random_state)

    console = Console()

    # --- Setup Device ---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Load Data ---
    # Real data (primarily for test set)
    try:
        _, _, X_test_np, y_test_np = load_oracle_data(data=args.data, model=EMBEDDING_MODEL_NAME)
        if X_test_np is None or y_test_np is None:
            raise ValueError("Failed to load oracle test data.")
        X_test = torch.from_numpy(X_test_np.astype(np.float32))
        y_test = torch.from_numpy(y_test_np).long() # Ensure hard labels are long
        test_dataset = TensorDataset(X_test, y_test)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                     drop_last=False, shuffle=False)
        logger.info(f"Loaded oracle test data: {len(test_dataset):,} samples.")
    except Exception as e:
        logger.info(f"Error loading oracle data: {e}")
        return # Exit if data loading fails

    def _downsample(X: np.ndarray, y: np.ndarray, size: int, random_state: int = 42):
        assert X.shape[0] == y.shape[0], "X and y must have same number of rows."
        assert X.shape[0] >= size, "len(X) should be larger than `size`."
        rng = np.random.default_rng(random_state)
        sample_idx = rng.permutation(X.shape[0])[:size]
        return X[sample_idx], y[sample_idx]
    
    def _filter_by_top_gap(X: np.ndarray, y: np.ndarray, gap: float = 0.2):
        assert y.ndim == 2
        sorted_probs = np.sort(y, axis=-1)
        diff = sorted_probs[:, -1] - sorted_probs[:, -2]
        idx = np.where(diff >= gap)[0]
        return X[idx], y[idx]
    
    def _filter_by_margin(X: np.ndarray, y: np.ndarray, margin: float = 0.2):
        assert y.ndim == 2
        pos_probs = y[:, 1]
        mask = (pos_probs < 0.5 - margin) | (pos_probs > 0.5 + margin)
        idx = np.where(mask)[0]
        return X[idx], y[idx]

    # Synthetic data (for training set)
    try:
        X_train_np, Y_train_np = load_synthetic_data(data=args.data,
                                                     directory=args.synthetic_data_dir,
                                                     model=EMBEDDING_MODEL_NAME)
        logger.info(f"Original synthetic data size: {X_train_np.shape[0]:,}")
        if args.data == 'emotion':
            X_train_np, Y_train_np = _filter_by_top_gap(X_train_np, Y_train_np, gap=args.margin)
            logger.info(f"Synthetic data size after filtering ({args.margin:.2f}): {X_train_np.shape[0]}")
        elif args.data in ('imdb', 'sst', 'sst2', 'subj'):
            X_train_np, Y_train_np = _filter_by_margin(X_train_np, Y_train_np, margin=args.margin)
            logger.info(f"Synthetic data size after filtering ({args.margin:.2f}): {X_train_np.shape[0]}")
        if args.downsample:
            _oracle_tr_size = DATA_TO_TRAIN_SIZE[args.data]
            X_train_np, Y_train_np = _downsample(X_train_np, Y_train_np, size=_oracle_tr_size, random_state=args.random_state)
            logger.info(f"Downsampled synthetic data size: {X_train_np.shape[0]:,}")
        X_train = torch.from_numpy(X_train_np.astype(np.float32))
        num_classes: int = Y_train_np.shape[1]
        if args.force_hard_labels:
            logger.info("Forcing soft labels to hard labels (not one-hot, but indicators) for training.")
            Y_hard_train_np = np.argmax(Y_train_np, axis=1)
            Y_train = torch.from_numpy(Y_hard_train_np).long()
            criterion = nn.NLLLoss().to(device) # NLLLoss for hard labels (model outputs log_softmax)
            logger.info(f"Using {criterion.__class__.__name__} for training with hard labels.")
        else:
            Y_train = torch.from_numpy(Y_train_np.astype(np.float32)) # Soft target probabilities
            criterion = nn.KLDivLoss(reduction='batchmean', log_target=False).to(device) # KLDiv for soft
            logger.info(f"Using {criterion.__class__.__name__} for training with soft labels.")
        
        # torch datasets and loades
        train_dataset = TensorDataset(X_train, Y_train)
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        )
        actual_embedding_dim = X_train.shape[1]
        assert actual_embedding_dim == X_test.shape[1]
        logger.info(f"Loaded synthetic training data: {len(train_dataset)} samples.")
        logger.info(f"Num Classes: {num_classes}, Embedding Dim: {actual_embedding_dim}")
    except Exception as e:
        logger.info(f"Error loading synthetic data: {e}")
        return # Exit if data loading fails

    # --- Initialize Model ---
    print(f"Initializing model: {args.model}")
    if args.model == 'mlp':
        model = SimpleMLP(actual_embedding_dim, args.hidden_dim, num_classes)
    elif args.model == 'logistic':
        model = LogisticRegression(actual_embedding_dim, num_classes)
    else:
        # Should not happen due to argparse choices, but good practice
        raise ValueError(f"Unknown model type: {args.model}")
    model.to(device)
    logger.info(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")


    # --- Optimizer ---
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)
    else:
        raise NotImplementedError

    # --- Train ---
    logger.info("\n--- Training Model ---")
    train_model(
        model,
        train_dataloader,
        optimizer,
        criterion,
        device,
        epochs=args.num_epochs)

    # --- Evaluate ---
    logger.info("\n--- Evaluating Model ---")
    evaluation_results = evaluate_model(
        model,
        test_dataloader,
        device,
        num_classes,
        num_bins=args.num_bins # Pass num_bins from args
    )

    # --- Plot ---
    if evaluation_results:
        fig = plot_reliability_diagram(
            evaluation_results['calibration_results'],
            title=args.plot_title # Use title from args
        )
        # You might want to save the plot and results dictionary here
        fig.savefig(Path(args.output_dir) / "reliability_diagram.pdf", dpi=200)
        # plt.savefig(Path(args.output_dir) / "reliability_diagram.png")
        with open(Path(args.output_dir) / "results.json", "w") as f:
            json.dump(evaluation_results, f, indent=2)

    print("\n--- Script Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate classifiers on synthetic/real text embeddings.")

    # Data Arguments
    parser.add_argument('--data', type=str, required=True, choices=('imdb', 'sst', 'subj', 'emotion', 'agnews'))
    parser.add_argument('--synthetic_data_dir', type=str, required=True,
                        help='Directory containing the synthetic data (data.jsonl) and embeddings/.')

    # Model Arguments
    parser.add_argument('--model', type=str, default='logistic', choices=['mlp', 'logistic'],
                        help='Type of downstream classifier model to train (default: logistic).')
    parser.add_argument('--embedding_dim', type=int, default=DEFAULT_EMBEDDING_DIM,
                        help=f'Dimension of the text embeddings (default: {DEFAULT_EMBEDDING_DIM} for {EMBEDDING_MODEL_NAME}). Verified against loaded data.')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension for the MLP model (default: 128).')

    # Training Arguments
    parser.add_argument('--optimizer', type=str, default='adam', choices=('adam', 'sgd'))
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for the optimizer (default: 0.001).')
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training and evaluation (default: 32).')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of training epochs (default: 5).')

    parser.add_argument('--log_interval', type=int, default=10000)

    # Evaluation Arguments
    parser.add_argument('--num_bins', type=int, default=10,
                        help='Number of bins for ECE calculation and reliability diagrams (default: 10).')

    # Other Arguments
    parser.add_argument('--device', type=str, default="cuda",
                        help='Device to use (e.g., "cpu", "cuda", "cuda:0"). Autodetects if None.')
    parser.add_argument('--plot_title', type=str, default='Reliability Diagram',
                        help='Title for the reliability diagram plot.')
    parser.add_argument('--output_dir', type=str, default='./results_pytorch',
                        help='Directory to save results or plots (default: current directory).')
    
    parser.add_argument('--force_hard_labels', action='store_true')
    parser.add_argument('--downsample', action='store_true')
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('-m', '--margin', type=float, default=0.0)

    args = parser.parse_args()
    
    console = Console()
    console.print(vars(args))

    # Create output directory if it doesn't exist
    setattr(args,
            "output_dir",
            os.path.join(args.output_dir, f"{args.data}/{generate_unique_identifier()}"))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        # save argparse config
    config_output_file = os.path.join(args.output_dir, 'config.yaml')
    write_argparse_args_to_yaml(args, filepath=config_output_file)
    logger.info(f"Configurations saved to: {config_output_file}")
    
    # logging
    rich_handler = RichHandler(
        level=logging.DEBUG, # Set the logging level for this handler
        show_time=True,
        show_level=True,
        show_path=True, # Set to True to see the file path and line number
        markup=True, # Enable Rich's markup for log messages
        rich_tracebacks=True # Enable rich formatting for tracebacks
    )
    logger.addHandler(rich_handler)

    # --- File Handler (main.log) ---
    # Create a handler for the file
    file_handler = logging.FileHandler(os.path.join(args.output_dir, 'main.log'), mode='a') # 'a' for append
    # Set the logging level for this handler (e.g., DEBUG)
    file_handler.setLevel(logging.DEBUG)
    # Create a formatter and add it to the handler
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    file_handler.setFormatter(file_formatter)
    # Add the handler to the logger
    logger.addHandler(file_handler)

    logger.info("Start!")
    main(args);
