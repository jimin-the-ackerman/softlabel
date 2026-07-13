import os
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../")
    )
)  # appending project folder to list of system paths

import time
import json
import argparse
import warnings
import logging
import glob
import random
from typing import List, Dict, Any, Optional, Tuple, Union

from rich.logging import RichHandler
from rich.console import Console
from dotenv import load_dotenv

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr, spearmanr

from rebuttal.loaders import load_oracle_embeddings, load_synthetic_embeddings

# Configure the logging system
def setup_logging(output_dir: Optional[str] = None):
    """Setup logging to both console and file."""
    # Create formatters
    console_formatter = logging.Formatter('%(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                      datefmt='%Y-%m-%d %H:%M:%S')
    
    # Create handlers
    console_handler = RichHandler()
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()  # Remove any existing handlers
    root_logger.addHandler(console_handler)
    
    # Add file handler if output_dir is provided
    if output_dir:
        log_file = os.path.join(output_dir, 'main.log')
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)

# Initial setup with console only
setup_logging()


def get_next_experiment_number(data_folder: str, model: str) -> int:
    """Get the next available experiment number (exp0, exp1, exp2, ...) for a specific model."""
    judgments_dir = os.path.join(data_folder, "gold_judgments", model)
    if not os.path.exists(judgments_dir):
        return 0
    
    # Find existing exp* directories
    exp_pattern = os.path.join(judgments_dir, "exp*")
    existing_exps = glob.glob(exp_pattern)
    
    if not existing_exps:
        return 0
    
    # Extract numbers from existing exp directories
    exp_numbers = []
    for exp_path in existing_exps:
        exp_name = os.path.basename(exp_path)
        if exp_name.startswith("exp"):
            try:
                number = int(exp_name[3:])  # Remove "exp" prefix
                exp_numbers.append(number)
            except ValueError:
                continue
    
    if not exp_numbers:
        return 0
    
    return max(exp_numbers) + 1


def train_gold_judge(oracle_embeddings: np.ndarray,
                     oracle_labels: np.ndarray, 
                     data_type: str,
                     cv_folds: int = 5,
                     n_jobs: int = 4) -> Tuple[Union[LogisticRegressionCV, LogisticRegression], Dict[str, float]]:
    """
    Train a gold judge classifier on oracle data.
    
    Args:
        oracle_embeddings: Training embeddings
        oracle_labels: Training labels
        data_type: Type of data (imdb, sst, subj, emotion)
        cv_folds: Number of cross-validation folds
        
    Returns:
        Trained model and validation metrics
    """
    logging.info(f"Training gold judge for {data_type} with {len(oracle_embeddings)} samples")
    
    if data_type == 'emotion':
        # For emotion, we'll discretize the 6-dimensional vectors into classes
        # and use multi-class LogisticRegression
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        
        # Train multi-class LogisticRegression
        model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            solver='lbfgs',  # or saga?
        )
        
        # Train the model
        model.fit(oracle_embeddings, oracle_labels)
        
        # Calculate cross-validation scores
        cv_scores = cross_val_score(model, oracle_embeddings, oracle_labels, cv=cv_folds)
        
        # Calculate validation metrics
        validation_metrics = {
            "cv_mean_accuracy": float(np.mean(cv_scores)),
            "cv_std_accuracy": float(np.std(cv_scores)),
            "cv_scores": cv_scores.tolist(),
            "model_type": "multiclass_logistic_regression"
        }
        
        logging.info(f"Gold judge training completed:")
        logging.info(f"  CV Accuracy: {validation_metrics['cv_mean_accuracy']:.4f} ± {validation_metrics['cv_std_accuracy']:.4f}")
        logging.info(f"  Model Type: {validation_metrics['model_type']}")
        
        return model, validation_metrics
    else:
        # For binary classification tasks (sentiment/subjectivity)
        # Initialize LogisticRegressionCV with cross-validation
        model = LogisticRegressionCV(
            cv=cv_folds,
            random_state=42,
            max_iter=1000,
            solver='lbfgs',  # Faster solver
            n_jobs=n_jobs    # Parallel processing for CV
        )
        
        # Train the model
        model.fit(oracle_embeddings, oracle_labels)
        
        # Calculate cross-validation scores
        cv_scores = cross_val_score(model, oracle_embeddings, oracle_labels, cv=cv_folds)
        
        # Calculate validation metrics
        validation_metrics = {
            "cv_mean_accuracy": float(np.mean(cv_scores)),
            "cv_std_accuracy": float(np.std(cv_scores)),
            "cv_scores": cv_scores.tolist(),
            "best_c": float(model.C_[0]),  # Best regularization parameter
            "model_type": "logistic_regression"
        }
        
        logging.info(f"Gold judge training completed:")
        logging.info(f"  CV Accuracy: {validation_metrics['cv_mean_accuracy']:.4f} ± {validation_metrics['cv_std_accuracy']:.4f}")
        logging.info(f"  Best C: {validation_metrics['best_c']:.4f}")
        
        return model, validation_metrics


def predict_with_gold_judge(model: Union[LogisticRegressionCV, LogisticRegression], synthetic_embeddings: np.ndarray, data_type: str) -> np.ndarray:
    """
    Use trained gold judge to predict on synthetic data.
    
    Args:
        model: Trained model (LogisticRegressionCV for binary, LogisticRegression for emotion)
        synthetic_embeddings: Synthetic data embeddings
        data_type: Type of data (imdb, sst, subj, emotion)
        
    Returns:
        Predicted scores (binary probabilities or discretized emotion classes)
    """
    if data_type == 'emotion':
        # For emotion, we use the trained multi-class model
        # Get probability predictions for all classes
        predictions = model.predict_proba(synthetic_embeddings)
        
        # Convert back to 6-dimensional emotion vectors
        # Each row represents the probability distribution across 6 emotions
        return predictions
    else:
        # For binary classification tasks
        # Get probability predictions for positive class
        predictions = model.predict_proba(synthetic_embeddings)
        
        # Return probabilities for positive class (class 1)
        # For binary classification, this gives us a score between 0 and 1
        return predictions[:, 1]


def calculate_bin_wise_mae(original_labels: List[float], judged_scores: List[float], 
                          bin_edges: List[float] = None) -> Dict[str, Any]:
    """Calculate bin-wise mean absolute error analysis."""
    if bin_edges is None:
        # Default: 10 bins from 0.0 to 1.0
        bin_edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Filter out error cases
    valid_pairs = [(orig, judged) for orig, judged in zip(original_labels, judged_scores) 
                   if isinstance(judged, (int, float)) and not isinstance(judged, bool)]
    
    if not valid_pairs:
        return {"bins": [], "bin_config": {"strategy": "fixed_10_bins", "bin_edges": bin_edges}}
    
    orig_values, judged_values = zip(*valid_pairs)
    orig_array = np.array(orig_values)
    judged_array = np.array(judged_values)
    errors = np.abs(orig_array - judged_array)
    
    bins = []
    
    for i in range(len(bin_edges) - 1):
        bin_start, bin_end = bin_edges[i], bin_edges[i + 1]
        
        # Find samples in this bin (based on original labels)
        mask = (orig_array >= bin_start) & (orig_array < bin_end)
        
        if i == len(bin_edges) - 2:  # Last bin includes the upper bound
            mask = (orig_array >= bin_start) & (orig_array <= bin_end)
        
        bin_orig = orig_array[mask]
        bin_judged = judged_array[mask]
        bin_errors = errors[mask]
        
        if len(bin_orig) > 0:
            bin_info = {
                "bin_index": i,
                "bin_range": [float(bin_start), float(bin_end)],
                "count": int(len(bin_orig)),
                "mean_absolute_error": float(np.mean(bin_errors)),
                "mean_original_label": float(np.mean(bin_orig)),
                "mean_judged_score": float(np.mean(bin_judged)),
                "std_error": float(np.std(bin_errors)),
                "min_error": float(np.min(bin_errors)),
                "max_error": float(np.max(bin_errors))
            }
        else:
            bin_info = {
                "bin_index": i,
                "bin_range": [float(bin_start), float(bin_end)],
                "count": 0,
                "mean_absolute_error": 0.0,
                "mean_original_label": 0.0,
                "mean_judged_score": 0.0,
                "std_error": 0.0,
                "min_error": 0.0,
                "max_error": 0.0
            }
        
        bins.append(bin_info)
    
    return {
        "bin_config": {
            "strategy": "fixed_10_bins",
            "bin_edges": bin_edges
        },
        "bins": bins
    }


def create_comparison_plot(original_labels: List[float], judged_scores: List[float], 
                          output_dir: str, model_name: str, data_type: str) -> str:
    """Create a scatter plot comparing original labels vs judged scores."""
    # Filter out error cases
    valid_pairs = [(orig, judged) for orig, judged in zip(original_labels, judged_scores) 
                   if isinstance(judged, (int, float)) and not isinstance(judged, bool)]
    
    if not valid_pairs:
        logging.warning("No valid pairs for plotting")
        return ""
    
    orig_values, judged_values = zip(*valid_pairs)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(orig_values, judged_values, alpha=0.6, s=20, color='blue')
    
    # Perfect correlation line (y=x)
    min_val = min(min(orig_values), min(judged_values))
    max_val = max(max(orig_values), max(judged_values))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Correlation')
    
    # Calculate correlation for title
    correlation = np.corrcoef(orig_values, judged_values)[0, 1]
    
    # Get task-specific labels
    if data_type in ['imdb', 'sst']:
        xlabel = 'Original Sentiment Labels'
        ylabel = 'Gold Judge Sentiment Scores'
        title_task = 'Sentiment'
    elif data_type == 'subj':
        xlabel = 'Original Subjectivity Labels'
        ylabel = 'Gold Judge Subjectivity Scores'
        title_task = 'Subjectivity'
    else:
        xlabel = 'Original Labels'
        ylabel = 'Gold Judge Scores'
        title_task = data_type.upper()
    
    # Customize the plot
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f'{title_task} Alignment: Original vs Gold Judge\nModel: {model_name}, Correlation: {correlation:.3f}', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set axis limits
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Add text box with metrics
    mae = np.mean(np.abs(np.array(orig_values) - np.array(judged_values)))
    rmse = np.sqrt(np.mean((np.array(orig_values) - np.array(judged_values)) ** 2))
    
    textstr = f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nN: {len(valid_pairs)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path


def calculate_alignment_metrics(original_labels: Union[List[float], List[List[float]]], judged_scores: Union[List[float], List[List[float]]], data_type: str) -> Dict[str, Any]:
    """Calculate alignment metrics between original labels and judged scores."""
    if len(original_labels) != len(judged_scores):
        logging.error("Mismatch in number of labels and judgments")
        return {}
    
    # Filter out error cases
    valid_pairs = [(orig, judged) for orig, judged in zip(original_labels, judged_scores) 
                   if isinstance(judged, (int, float)) and not isinstance(judged, bool)]
    
    if not valid_pairs:
        logging.error("No valid pairs for alignment calculation")
        return {}
    
    orig_values, judged_values = zip(*valid_pairs)
    
    # Correlation coefficients
    try:
        pearson_corr, pearson_p = pearsonr(orig_values, judged_values)
        spearman_corr, spearman_p = spearmanr(orig_values, judged_values)
    except (ValueError, TypeError) as e:
        logging.warning(f"Error calculating correlations: {e}")
        pearson_corr = pearson_p = spearman_corr = spearman_p = np.nan
    
    # Mean absolute error
    mae = np.mean(np.abs(np.array(orig_values) - np.array(judged_values)))
    
    # Root mean square error
    rmse = np.sqrt(np.mean((np.array(orig_values) - np.array(judged_values)) ** 2))
    
    # Calculate bin-wise analysis
    bin_analysis = calculate_bin_wise_mae(original_labels, judged_scores)
    
    return {
        "overall_metrics": {
            "pearson_correlation": float(pearson_corr),
            "pearson_p_value": float(pearson_p),
            "spearman_correlation": float(spearman_corr),
            "spearman_p_value": float(spearman_p),
            "mean_absolute_error": float(mae),
            "root_mean_square_error": float(rmse),
            "num_valid_pairs": len(valid_pairs),
            "total_pairs": len(original_labels)
        },
        "bin_wise_analysis": bin_analysis
    }


def calculate_l2_distance(orig_vec: List[float], judged_vec: List[float]) -> float:
    """Calculate L2 distance between two emotion vectors."""
    import numpy as np
    orig_array = np.array(orig_vec)
    judged_array = np.array(judged_vec)
    return float(np.linalg.norm(orig_array - judged_array))


def calculate_cosine_similarity(orig_vec: List[float], judged_vec: List[float]) -> float:
    """Calculate cosine similarity between two emotion vectors."""
    import numpy as np
    from scipy.spatial.distance import cosine
    orig_array = np.array(orig_vec)
    judged_array = np.array(judged_vec)
    return float(1 - cosine(orig_array, judged_array))


def calculate_emotion_alignment_metrics(original_labels: List[List[float]], judged_scores: List[List[float]]) -> Dict[str, Any]:
    """Calculate alignment metrics for emotion classification using cosine similarity."""
    if len(original_labels) != len(judged_scores):
        logging.error("Mismatch in number of labels and judgments")
        return {}
    
    # Filter out error cases
    valid_pairs = [(orig, judged) for orig, judged in zip(original_labels, judged_scores) 
                   if isinstance(judged, list) and len(judged) == 6 and all(isinstance(x, (int, float)) for x in judged)]
    
    if not valid_pairs:
        logging.error("No valid pairs for emotion alignment calculation")
        return {}
    
    orig_values, judged_values = zip(*valid_pairs)
    
    # Calculate cosine similarities for each pair
    import numpy as np
    from scipy.spatial.distance import cosine
    
    cosine_similarities = []
    l2_distances = []
    
    for orig_vec, judged_vec in valid_pairs:
        # Calculate cosine similarity
        cos_sim = calculate_cosine_similarity(orig_vec, judged_vec)
        cosine_similarities.append(cos_sim)
        
        # Calculate L2 distance
        l2_dist = calculate_l2_distance(orig_vec, judged_vec)
        l2_distances.append(l2_dist)
    
    # Calculate summary statistics
    mean_cosine_similarity = np.mean(cosine_similarities)
    std_cosine_similarity = np.std(cosine_similarities)
    mean_l2_distance = np.mean(l2_distances)
    std_l2_distance = np.std(l2_distances)
    
    return {
        "overall_metrics": {
            "mean_cosine_similarity": float(mean_cosine_similarity),
            "std_cosine_similarity": float(std_cosine_similarity),
            "mean_l2_distance": float(mean_l2_distance),
            "std_l2_distance": float(std_l2_distance),
            "num_valid_pairs": len(valid_pairs),
            "total_pairs": len(original_labels)
        },
        "cosine_similarities": [float(x) for x in cosine_similarities],
        "l2_distances": [float(x) for x in l2_distances]
    }


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('Gold as a Judge for Text Alignment')
    
    parser.add_argument('--data_folder', type=str, required=True,
                        help='Path to the synthetic data folder containing data.jsonl')
    

    
    parser.add_argument('--data', type=str, required=True,
                        choices=['imdb', 'sst', 'subj', 'emotion'],
                        help='Type of data to judge (imdb, sst, subj, emotion)')
    
    # Model type is automatically determined by data type:
    # - imdb, sst, subj: LogisticRegressionCV (binary classification)
    # - emotion: LogisticRegression with multi-class (6 emotion classes)
    
    parser.add_argument('--cv_folds', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')
    
    parser.add_argument('--n_jobs_cv', type=int, default=4,
                        help='Number of jobs for cross-validation (default: 4)')
    
    parser.add_argument('--output_file', type=str, default=None,
                        help='File to save the judgment results (JSON format, default: None)')
    
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Number of samples to randomly sample from data.jsonl (for testing, default: None)')
    
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed output for each judgment')
    
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Main function."""
    console = Console()
    console.print(vars(args))
    
    # Configure output directory at the very beginning
    if not args.output_file:
        # Determine model name based on data type
        if args.data == 'emotion':
            model_name = 'multiclass-logistic-regression'
        else:
            model_name = 'logistic-regression'
        
        exp_number = get_next_experiment_number(args.data_folder, model_name)
        judgments_dir = os.path.join(args.data_folder, "gold_judgments", model_name)
        exp_dir = os.path.join(judgments_dir, f"exp{exp_number}")
        os.makedirs(exp_dir, exist_ok=True)
        output_dir = exp_dir
        console.print(f"📁 Auto-generated output directory: {output_dir}")
    else:
        # If custom output file specified, use its directory
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    # Setup file logging to output directory
    setup_logging(output_dir)
    logging.info(f"Logging to file: {os.path.join(output_dir, 'main.log')}")
    
    # Check if data folder exists
    if not os.path.exists(args.data_folder):
        logging.error(f"Data folder does not exist: {args.data_folder}")
        return
    
    # Load oracle embeddings and labels for training
    logging.info(f"Loading oracle embeddings for {args.data}")
    oracle_embeddings_dict = load_oracle_embeddings(args.data)
    
    # Get train embeddings and labels
    oracle_embeddings = oracle_embeddings_dict['train'][0]  # embeddings
    oracle_labels = oracle_embeddings_dict['train'][1]      # labels (binary)
    
    logging.info(f"Loaded {len(oracle_embeddings)} oracle training samples")
    
    # Load synthetic embeddings and labels
    logging.info(f"Loading synthetic embeddings from: {args.data_folder}")
    synthetic_embeddings_dict = load_synthetic_embeddings(args.data, args.data_folder)
    
    # Get synthetic embeddings and labels
    synthetic_embeddings = synthetic_embeddings_dict['embeddings']
    synthetic_labels = synthetic_embeddings_dict['labels']  # 2d soft labels
    
    logging.info(f"Loaded {len(synthetic_embeddings)} synthetic samples")
    
    # Random sampling if specified
    if args.sample_size and args.sample_size < len(synthetic_embeddings):
        original_count = len(synthetic_embeddings)
        random.seed(42)  # For reproducible sampling
        indices = random.sample(range(len(synthetic_embeddings)), args.sample_size)
        synthetic_embeddings = synthetic_embeddings[indices]
        synthetic_labels = synthetic_labels[indices]
        logging.info(f"Randomly sampled {args.sample_size} samples from {original_count} total samples")
    
    # Train gold judge
    gold_judge, validation_metrics = train_gold_judge(
        oracle_embeddings, oracle_labels, args.data, args.cv_folds, args.n_jobs_cv
    )
    
    # Make predictions on synthetic data
    logging.info("Making predictions on synthetic data...")
    predicted_scores = predict_with_gold_judge(gold_judge, synthetic_embeddings, args.data)
    
    # Extract original labels (convert from 2D to 1D if needed)
    if args.data == 'emotion':
        # For emotion, labels are already 6-dimensional vectors
        original_labels = synthetic_labels.tolist()
        # For emotion, we need to handle 6-dimensional predictions
        # The gold judge will predict 6-dimensional emotion vectors
        judged_scores = predicted_scores.tolist()
    else:
        if synthetic_labels.ndim == 2:
            # Convert from [P(class=0), P(class=1)] to single score
            original_labels = synthetic_labels[:, 1].tolist()
        else:
            original_labels = synthetic_labels.tolist()
        
        # Convert predicted scores to list
        judged_scores = predicted_scores.tolist()
    
    # Calculate alignment metrics
    if args.data == 'emotion':
        metrics = calculate_emotion_alignment_metrics(original_labels, judged_scores)
    else:
        metrics = calculate_alignment_metrics(original_labels, judged_scores, args.data)
    
    # Create comparison plot (skip for emotion as it's more complex)
    if args.data != 'emotion':
        plot_path = create_comparison_plot(original_labels, judged_scores, output_dir, model_name, args.data)
        if plot_path:
            console.print(f"📊 Comparison plot saved to: {plot_path}")
    else:
        console.print("📊 Skipping comparison plot for emotion (vector data)")
    
    # Display results
    console.print("\n" + "="*60)
    console.print(f"🎯 {args.data.upper()} GOLD JUDGE ALIGNMENT RESULTS")
    console.print("="*60)
    
    # Display validation metrics
    console.print(f"📊 Gold Judge Validation:")
    console.print(f"  CV Accuracy: {validation_metrics['cv_mean_accuracy']:.4f} ± {validation_metrics['cv_std_accuracy']:.4f}")
    if args.data != 'emotion':
        console.print(f"  Best C: {validation_metrics['best_c']:.4f}")
    console.print(f"  Model Type: {validation_metrics['model_type']}")
    
    if metrics and "overall_metrics" in metrics:
        overall = metrics["overall_metrics"]
        
        if args.data == 'emotion':
            console.print(f"\n📊 Alignment Metrics:")
            console.print(f"  Mean Cosine Similarity: {overall['mean_cosine_similarity']:.4f} (±{overall['std_cosine_similarity']:.4f})")
            console.print(f"  Mean L2 Distance: {overall['mean_l2_distance']:.4f} (±{overall['std_l2_distance']:.4f})")
            console.print(f"  Valid Pairs: {overall['num_valid_pairs']}/{overall['total_pairs']}")
        else:
            console.print(f"\n📊 Alignment Metrics:")
            console.print(f"  Pearson Correlation: {overall['pearson_correlation']:.4f} (p={overall['pearson_p_value']:.4f})")
            console.print(f"  Spearman Correlation: {overall['spearman_correlation']:.4f} (p={overall['spearman_p_value']:.4f})")
            console.print(f"  Mean Absolute Error: {overall['mean_absolute_error']:.4f}")
            console.print(f"  Root Mean Square Error: {overall['root_mean_square_error']:.4f}")
            console.print(f"  Valid Pairs: {overall['num_valid_pairs']}/{overall['total_pairs']}")
            
            # Display bin-wise MAE summary for scalar tasks
            if "bin_wise_analysis" in metrics and "bins" in metrics["bin_wise_analysis"]:
                bins = metrics["bin_wise_analysis"]["bins"]
                console.print(f"\n📊 BIN-WISE MAE ANALYSIS:")
                console.print("-" * 60)
                
                for bin_info in bins:
                    if bin_info["count"] > 0:  # Only show bins with data
                        console.print(f"  Bin {bin_info['bin_index']:2d} [{bin_info['bin_range'][0]:.1f}-{bin_info['bin_range'][1]:.1f}]: "
                                    f"MAE={bin_info['mean_absolute_error']:.4f} (n={bin_info['count']})")
    
    # Show sample comparisons
    console.print(f"\n📋 SAMPLE COMPARISONS:")
    console.print("-" * 60)
    
    for i in range(min(5, len(original_labels))):
        orig_label = original_labels[i]
        judged_score = judged_scores[i]
        
        console.print(f"Sample {i+1}:")
        
        if args.data == 'emotion':
            console.print(f"  Original Label: {orig_label}")
            console.print(f"  Gold Judge Score: {judged_score}")
            # Calculate distance between vectors
            distance = calculate_l2_distance(orig_label, judged_score)
            cos_sim = calculate_cosine_similarity(orig_label, judged_score)
            console.print(f"  L2 Distance: {distance:.3f}")
            console.print(f"  Cosine Similarity: {cos_sim:.3f}")
        else:
            console.print(f"  Original Label: {orig_label:.3f}")
            console.print(f"  Gold Judge Score: {judged_score:.3f}")
            console.print(f"  Difference: {abs(orig_label - judged_score):.3f}")
        console.print()
    
    # Prepare metadata
    metadata = {
        "data_folder": args.data_folder,
        "data_type": args.data,
        "model": model_name,
        "cv_folds": args.cv_folds,
        "n_jobs_cv": args.n_jobs_cv,
        "sample_size": args.sample_size,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": len(synthetic_embeddings),
        "oracle_training_samples": len(oracle_embeddings),
        "validation_metrics": validation_metrics
    }
    
    # Save metadata
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    console.print(f"📋 Metadata saved to: {metadata_file}")
    
    # Save alignment metrics
    metrics_file = os.path.join(output_dir, "alignment_metrics.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    console.print(f"📊 Alignment metrics saved to: {metrics_file}")
    
    # Save results
    results_file = os.path.join(output_dir, "results.jsonl")
    with open(results_file, 'w', encoding='utf-8') as f:
        for i, (orig_label, judged_score) in enumerate(zip(original_labels, judged_scores)):
            result_entry = {
                "sample_index": i,
                "original_label": orig_label,
                "gold_judge_score": judged_score,
                "difference": abs(orig_label - judged_score)
            }
            f.write(json.dumps(result_entry) + '\n')
    console.print(f"📄 Results saved to: {results_file}")
    
    # If custom output file was specified, also save a combined version
    if args.output_file and args.output_file != results_file:
        combined_data = {
            "metadata": metadata,
            "alignment_metrics": metrics,
            "results": []
        }
        
        for i, (orig_label, judged_score) in enumerate(zip(original_labels, judged_scores)):
            combined_data["results"].append({
                "sample_index": i,
                "original_label": orig_label,
                "gold_judge_score": judged_score,
                "difference": abs(orig_label - judged_score)
            })
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2)
        console.print(f"💾 Combined results also saved to: {args.output_file}")


if __name__ == '__main__':
    load_dotenv()  # load environment variables from the .env file
    args = parse_arguments()  # parse command line arguments
    try:
        main(args)  # run main function
    except KeyboardInterrupt:
        logging.info("Ctrl+C pressed!")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)
