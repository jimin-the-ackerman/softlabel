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
import asyncio
import logging
import glob
import random
from typing import List, Dict, Any, Optional, Union

from rich.logging import RichHandler
from rich.console import Console
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from pydantic_core._pydantic_core import ValidationError
from pydantic import BaseModel, Field

from softprompt.utils import get_model_provider

# For plotting
import matplotlib.pyplot as plt
import numpy as np

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


class SentimentJudgment(BaseModel):
    """Schema for sentiment judgment output."""
    judge_score: float = Field(
        description="Sentiment score between 0 and 1, where 0 is negative and 1 is positive",
        ge=0.0,
        le=1.0
    )
    judge_confidence: float = Field(
        description="Confidence in the judgment between 0 and 1",
        ge=0.0,
        le=1.0
    )
    judge_reasoning: str = Field(
        description="Step-by-step reasoning for the sentiment score"
    )


class SubjectivityJudgment(BaseModel):
    """Schema for subjectivity judgment output."""
    judge_score: float = Field(
        description="Subjectivity score between 0 and 1, where 0 is objective and 1 is subjective",
        ge=0.0,
        le=1.0
    )
    judge_confidence: float = Field(
        description="Confidence in the judgment between 0 and 1",
        ge=0.0,
        le=1.0
    )
    judge_reasoning: str = Field(
        description="Step-by-step reasoning for the subjectivity score"
    )


class EmotionJudgment(BaseModel):
    """Schema for emotion judgment output."""
    judge_score: list[float] = Field(
        description="Emotion distribution across six emotions (sadness, joy, love, anger, fear, surprise) where each value is between 0 and 1 and all values sum to 1",
        min_items=6,
        max_items=6
    )
    judge_confidence: float = Field(
        description="Confidence in the judgment between 0 and 1",
        ge=0.0,
        le=1.0
    )
    judge_reasoning: str = Field(
        description="Step-by-step reasoning for the emotion distribution"
    )


class YahooJudgment(BaseModel):
    """Schema for Yahoo question category judgment output."""
    judge_score: list[float] = Field(
        description="Question category distribution across ten categories (Society & Culture, Science & Mathematics, Health, Education & Reference, Computers & Internet, Sports, Business & Finance, Entertainment & Music, Family & Relationships, Politics & Government) where each value is between 0 and 1 and all values sum to 1",
        min_items=10,
        max_items=10
    )
    judge_confidence: float = Field(
        description="Confidence in the judgment between 0 and 1",
        ge=0.0,
        le=1.0
    )
    judge_reasoning: str = Field(
        description="Step-by-step reasoning for the category distribution"
    )


class AGNewsJudgment(BaseModel):
    judge_score: list[float] = Field(
        description="Topic distribution across four categories (World, Sports, Business, Science/Technology) where each value is between 0 and 1 and all values sum to 1",
        min_items=4,
        max_items=4
    )
    judge_confidence: float = Field(
        description="Confidence in the judgment between 0 and 1",
        ge=0.0,
        le=1.0
    )
    judge_reasoning: str = Field(
        description="Step-by-step reasoning for the topic distribution"
    )


def get_judgment_schema(data_type: str):
    """Get the appropriate judgment schema based on data type."""
    if data_type in ['imdb', 'sst']:
        return SentimentJudgment
    elif data_type == 'subj':
        return SubjectivityJudgment
    elif data_type == 'emotion':
        return EmotionJudgment
    elif data_type == 'yahoo':
        return YahooJudgment
    elif data_type == 'agnews':
        return AGNewsJudgment
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


def get_next_experiment_number(data_folder: str, model: str) -> int:
    """Get the next available experiment number (exp0, exp1, exp2, ...) for a specific model."""
    judgments_dir = os.path.join(data_folder, "judgments", model)
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


def load_data_from_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    try:
        with open(filepath, "r", encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        logging.info(f"Loaded {len(data)} data points from {filepath}")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error in file {filepath}: {e}")
        return []
    except Exception as e:
        logging.error(f"Error reading file {filepath}: {e}")
        return []


def create_judgment_prompt(data_type: str) -> ChatPromptTemplate:
    """Create the prompt template for judgment based on data type."""
    if data_type in ['imdb', 'sst']:
        system_prompt = """You are an expert sentiment analyst. Your task is to evaluate the sentiment of given text and provide a score between 0 and 1, where:
- 0 represents very negative sentiment
- 0.5 represents neutral sentiment  
- 1 represents very positive sentiment

IMPORTANT: Provide your sentiment score with 3 decimal places of precision (e.g., 0.123, 0.456, 0.789). Do not round to simple fractions like 0.1, 0.2, etc.

Be thorough in your analysis and provide clear reasoning for your judgment."""

        human_prompt = """Please analyze the sentiment of the following text:

Text: {text}

Provide a sentiment score between 0 and 1 with 3 decimal places of precision (e.g., 0.123, 0.456, 0.789), along with your reasoning and confidence level."""
    
    elif data_type == 'subj':
        system_prompt = """You are an expert in analyzing text subjectivity. Your task is to evaluate the subjectivity of given text and provide a score between 0 and 1, where:
- 0 represents objective text (factual descriptions, neutral narration)
- 0.5 represents moderately subjective text
- 1 represents highly subjective text (personal opinions, emotional expressions, evaluative language)

IMPORTANT: Provide your subjectivity score with 3 decimal places of precision (e.g., 0.123, 0.456, 0.789). Do not round to simple fractions like 0.1, 0.2, etc.

Be thorough in your analysis and provide clear reasoning for your judgment."""

        human_prompt = """Please analyze the subjectivity of the following text:

Text: {text}

Provide a subjectivity score between 0 and 1 with 3 decimal places of precision (e.g., 0.123, 0.456, 0.789), along with your reasoning and confidence level."""
    
    elif data_type == 'emotion':
        system_prompt = """You are an expert in analyzing emotional content in text. Your task is to evaluate the emotional distribution of given text across six emotions: sadness, joy, love, anger, fear, and surprise.

For each emotion, provide a score between 0 and 1 indicating the strength of that emotion in the text. All six scores should sum to 1.0.

IMPORTANT: Provide each emotion score with 3 decimal places of precision (e.g., 0.123, 0.456, 0.789). Do not round to simple fractions.

Be thorough in your analysis and provide clear reasoning for your judgment."""

        human_prompt = """Please analyze the emotional content of the following text:

Text: {text}

Provide an emotion distribution across six emotions (sadness, joy, love, anger, fear, surprise) where each value is between 0 and 1 and all values sum to 1, along with your reasoning and confidence level."""
    
    elif data_type == 'yahoo':
        system_prompt = """You are an expert in analyzing question categories. Your task is to evaluate the category distribution of given questions and provide a score between 0 and 1 for each category, where:
- 0 represents no relevance to the category
- 1 represents very relevant to the category

IMPORTANT: Provide each category score with 3 decimal places of precision (e.g., 0.123, 0.456, 0.789). Do not round to simple fractions.

Be thorough in your analysis and provide clear reasoning for your judgment."""

        human_prompt = """Please analyze the category distribution of the following question:

Question: {text}

Provide a category distribution across ten categories (Society & Culture, Science & Mathematics, Health, Education & Reference, Computers & Internet, Sports, Business & Finance, Entertainment & Music, Family & Relationships, Politics & Government) where each value is between 0 and 1 and all values sum to 1, along with your reasoning and confidence level."""
    
    elif data_type == 'agnews':
        system_prompt = """You are an expert in analyzing news topics. Your task is to evaluate the topic distribution of given news articles and provide a score between 0 and 1 for each category, where:
- 0 represents no relevance to the category
- 1 represents very relevant to the category

IMPORTANT: Provide each category score with 3 decimal places of precision (e.g., 0.123, 0.456, 0.789). Do not round to simple fractions.

Be thorough in your analysis and provide clear reasoning for your judgment."""

        human_prompt = """Please analyze the topic distribution of the following news article:

Article: {text}

Provide a topic distribution across four categories (World, Sports, Business, Science/Technology) where each value is between 0 and 1 and all values sum to 1, along with your reasoning and confidence level."""
    
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])


def _create_error_result(error_msg: str, data_type: str = 'sentiment') -> Dict[str, Any]:
    """Create a standardized error result."""
    if data_type == 'emotion':
        judge_score = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]  # Uniform distribution
    elif data_type == 'yahoo':
        judge_score = [1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10]  # Uniform distribution
    elif data_type == 'agnews':
        judge_score = [1/4, 1/4, 1/4, 1/4] # Uniform distribution for AGNews
    else:
        judge_score = 0.5  # Neutral for sentiment/subjectivity
    
    return {
        "error": error_msg,
        "judge_score": judge_score,
        "judge_reasoning": "Failed to get judgment",
        "judge_confidence": 0.0
    }


def _write_result_entry(results_file: str, original_idx: int, text: str, original_label, 
                       original_reasoning: str, result_dict: Dict[str, Any], data_type: str) -> None:
    """Write a single result entry to the results file."""
    result_entry = {
        "text_index": original_idx,
        "original_label": original_label,
        "judge_score": result_dict.get('judge_score', 0.5 if data_type != 'emotion' else [1/6]*6),
        "judge_confidence": result_dict.get('judge_confidence', 0.0),
        "text": text,
        "original_reasoning": original_reasoning,
        "judge_reasoning": result_dict.get('judge_reasoning', '')
    }
    with open(results_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result_entry) + '\n')
        f.flush()


async def judge_texts_batch(
    texts: List[str],
    original_labels: Union[List[float], List[List[float]]],
    original_reasonings: List[str],
    original_indices: List[int],
    data_type: str,
    model: str = "gemini-2.0-flash",
    temperature: float = 0.1,
    api_key: Optional[str] = None,
    max_tokens: Optional[int] = None,
    batch_size: int = 10,
    results_file: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Judge texts for multiple tasks in batches.
    
    Args:
        texts: List of texts to analyze
        original_labels: List of original labels (float for sentiment/subjectivity, List[float] for emotion)
        original_reasonings: List of original reasoning
        original_indices: List of original indices
        data_type: Type of data (imdb, sst, subj, emotion)
        model: LLM model to use
        temperature: Sampling temperature (0.0 to 1.0)
        api_key: API key for the model provider
        max_tokens: Maximum tokens for the response
        batch_size: Number of texts to process in parallel
        
    Returns:
        List of dictionaries containing the judgment results
    """
    
    # Set API key if provided
    if api_key is not None:
        if model.startswith('gemini'):
            os.environ['GOOGLE_API_KEY'] = api_key
        elif model.startswith(('gpt', 'o1', 'o3', 'o4')):
            os.environ['OPENAI_API_KEY'] = api_key
        elif model.startswith('claude'):
            os.environ['ANTHROPIC_API_KEY'] = api_key
    
    # Initialize the chat model
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        llm = init_chat_model(
            model=model,
            model_provider=get_model_provider(model),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=60.0,
            max_retries=3,
        )
    
    # Get the appropriate schema and create structured output
    judgment_schema = get_judgment_schema(data_type)
    structured_llm = llm.with_structured_output(judgment_schema)
    
    # Create the prompt template
    template = create_judgment_prompt(data_type)
    
    # Create the chain
    chain = template | structured_llm
    
    results = []
    
    # Process texts in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_start_idx = i
        logging.info(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} ({len(batch_texts)} texts)")
        
        try:
            # Create inputs for the batch
            chain_inputs = [{"text": text} for text in batch_texts]
            
            # Get judgments for the batch
            batch_results = await chain.abatch(
                chain_inputs, 
                config={"max_concurrency": batch_size}
            )
            
            # Process batch results
            for j, result in enumerate(batch_results):
                if result is not None:
                    result_dict = result.model_dump()
                else:
                    result_dict = _create_error_result("No response from model")
                
                results.append(result_dict)
                
                # Write result incrementally if results_file is provided
                if results_file:
                    text_idx = batch_start_idx + j
                    _write_result_entry(results_file, original_indices[text_idx], texts[text_idx], 
                                      original_labels[text_idx], original_reasonings[text_idx], result_dict, data_type)
                    
        except (OutputParserException, ValidationError) as e:
            logging.warning(f"Error in batch {i//batch_size + 1}: {e}")
            
            # Add error results for the failed batch
            for j in range(len(batch_texts)):
                error_result = _create_error_result(str(e), data_type)
                results.append(error_result)
                
                # Write error result incrementally
                if results_file:
                    text_idx = batch_start_idx + j
                    _write_result_entry(results_file, original_indices[text_idx], texts[text_idx], 
                                      original_labels[text_idx], original_reasonings[text_idx], error_result, data_type)
        
        # Add a small delay between batches to avoid rate limits
        if i + batch_size < len(texts):
            await asyncio.sleep(0.5)
    
    return results


def calculate_bin_wise_mae(original_labels: List[float], judged_scores: List[float], 
                          bin_edges: List[float] = None) -> Dict[str, Any]:
    """Calculate bin-wise mean absolute error analysis."""
    import numpy as np
    
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
                          output_dir: str, model: str, data_type: str) -> str:
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
        ylabel = 'LLM Judged Sentiment Scores'
        title_task = 'Sentiment'
    elif data_type == 'subj':
        xlabel = 'Original Subjectivity Labels'
        ylabel = 'LLM Judged Subjectivity Scores'
        title_task = 'Subjectivity'
    elif data_type == 'emotion':
        xlabel = 'Original Emotion Labels'
        ylabel = 'LLM Judged Emotion Scores'
        title_task = 'Emotion'
    elif data_type == 'yahoo':
        xlabel = 'Original Yahoo Labels'
        ylabel = 'LLM Judged Yahoo Scores'
        title_task = 'Yahoo'
    elif data_type == 'agnews':
        xlabel = 'Original AGNews Labels'
        ylabel = 'LLM Judged AGNews Scores'
        title_task = 'AGNews'
    else:
        xlabel = 'Original Labels'
        ylabel = 'LLM Judged Scores'
        title_task = data_type.upper()
    
    # Customize the plot
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f'{title_task} Alignment: Original vs LLM Judgments\nModel: {model}, Correlation: {correlation:.3f}', 
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
    if data_type == 'emotion':
        return calculate_emotion_alignment_metrics(original_labels, judged_scores)
    elif data_type == 'yahoo':
        return calculate_yahoo_alignment_metrics(original_labels, judged_scores)
    elif data_type == 'agnews':
        return calculate_emotion_alignment_metrics(original_labels, judged_scores) # AGNews uses the same emotion-like distribution
    else:
        return calculate_scalar_alignment_metrics(original_labels, judged_scores)

def calculate_scalar_alignment_metrics(original_labels: List[float], judged_scores: List[float]) -> Dict[str, Any]:
    """Calculate alignment metrics for scalar (sentiment/subjectivity) tasks."""
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
    
    # Calculate metrics
    import numpy as np
    from scipy.stats import pearsonr, spearmanr
    
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
    valid_pairs = [(orig, judged) for orig, judged in zip(original_labels, judged_scores)]
    
    # valid_pairs = [(orig, judged) for orig, judged in zip(original_labels, judged_scores) 
    #                if isinstance(judged, list) and len(judged) == 6 and all(isinstance(x, (int, float)) for x in judged)]
    
    # if not valid_pairs:
    #     logging.error("No valid pairs for emotion alignment calculation")
    #     return {}
    
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

def calculate_yahoo_alignment_metrics(original_labels: List[List[float]], judged_scores: List[List[float]]) -> Dict[str, Any]:
    """Calculate alignment metrics for Yahoo question category classification using cosine similarity."""
    if len(original_labels) != len(judged_scores):
        logging.error("Mismatch in number of labels and judgments")
        return {}
    
    # Filter out error cases
    valid_pairs = [(orig, judged) for orig, judged in zip(original_labels, judged_scores) 
                   if isinstance(judged, list) and len(judged) == 10 and all(isinstance(x, (int, float)) for x in judged)]
    
    if not valid_pairs:
        logging.error("No valid pairs for Yahoo alignment calculation")
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
    parser = argparse.ArgumentParser('LLM as a Judge for Text Alignment')
    
    parser.add_argument('--data_folder', type=str, required=True,
                        help='Path to the data folder containing data.jsonl')
    
    parser.add_argument('--data', type=str, required=True,
                        choices=['imdb', 'sst', 'subj', 'emotion', 'agnews', 'yahoo'],
                        help='Type of data to judge (imdb, sst, subj)')
    
    parser.add_argument('--model', type=str, default='gemini-2.0-flash',
                        choices=['gpt-4o-mini', 'gpt-4o', 'gpt-4', 'gpt-3.5-turbo',
                                 'gemini-2.0-flash', 'gemini-1.5-flash',
                                 'claude-3-haiku', 'claude-3-sonnet', 'claude-3-opus'],
                        help='LLM model to use for judgment (default: gemini-2.0-flash)')
    
    parser.add_argument('--api_key', type=str, default=None,
                        help='API key for the model provider')
    
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Sampling temperature (larger than 0.0)')
    
    parser.add_argument('--max_tokens', type=int, default=None,
                        help='Maximum tokens for the response')
    
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Number of texts to process in parallel (default: 50)')
    
    parser.add_argument('--output_file', type=str, default=None,
                        help='File to save the judgment results (JSON format, default: None)')
    
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Number of samples to randomly sample from data.jsonl (for testing, default: None)')
    
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed output for each judgment')
    
    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    """Main function."""
    console = Console()
    console.print(vars(args))
    
    # Configure output directory at the very beginning
    if not args.output_file:
        exp_number = get_next_experiment_number(args.data_folder, args.model)
        judgments_dir = os.path.join(args.data_folder, "judgments", args.model)
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
    
    # Find data.jsonl file
    data_file = os.path.join(args.data_folder, 'data.jsonl')
    if not os.path.exists(data_file):
        logging.error(f"data.jsonl not found in: {args.data_folder}")
        return
    
    # Load data
    data = load_data_from_jsonl(data_file)
    if not data:
        logging.error("No data loaded")
        return
    
    # Random sampling if specified
    if args.sample_size and args.sample_size < len(data):
        original_count = len(data)
        random.seed(42)  # For reproducible sampling
        data = random.sample(data, args.sample_size)
        logging.info(f"Randomly sampled {args.sample_size} samples from {original_count} total samples")
    
    # Extract texts, labels, and original reasoning with indices
    texts = []
    original_labels = []
    original_reasonings = []
    original_indices = []
    
    for i, item in enumerate(data):
        # Handle different label types
        if args.data == 'emotion':
            # For emotion, label should be a list of 6 floats
            if isinstance(item['label'], list):
                original_labels.append(item['label'])
                texts.append(item['text'])
            else:
                logging.warning(f"Expected list for emotion label, got {type(item['label'])}")
                continue
        elif args.data == 'yahoo':
            # For Yahoo, label should be a list of 10 floats
            if isinstance(item['label'], list):
                original_labels.append(item['label'])
                texts.append([f"{item['title']} {item['content']}"])  # FIXME
            else:
                logging.warning(f"Expected list for Yahoo label, got {type(item['label'])}")
                continue
        elif args.data == 'agnews':
            # For AGNews, label should be a list of 4 floats
            if isinstance(item['label'], list):
                original_labels.append(item['label'])
                texts.append(item['text'])
            else:
                logging.warning(f"Expected list for AGNews label, got {type(item['label'])}")
                continue
        else:
            # For sentiment/subjectivity, label should be a float
            original_labels.append(float(item['label']))
            texts.append(item['text'])
            
        original_reasonings.append(item.get('reasoning', ''))
        original_indices.append(i)
    
    if not texts:
        logging.error("No valid texts found in data")
        return
    
    logging.info(f"Processing {len(texts)} texts with {args.model} for {args.data} task")
    logging.info(f"Batch size: {args.batch_size}")
    
    # Prepare results file for incremental writing
    results_file = os.path.join(output_dir, "results.jsonl")
    console.print(f"📄 Results will be saved incrementally to: {results_file}")
    
    # Get judgments with incremental saving
    judgments = await judge_texts_batch(
        texts=texts,
        original_labels=original_labels,
        original_reasonings=original_reasonings,
        original_indices=original_indices,
        data_type=args.data,
        model=args.model,
        temperature=args.temperature,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        results_file=results_file
    )
    
    # Extract judged scores
    judged_scores = []
    for judgment in judgments:
        if "error" in judgment:
            if args.data == 'emotion':
                judged_scores.append([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])  # Uniform distribution for errors
            elif args.data == 'yahoo':
                judged_scores.append([1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10])  # Uniform distribution for errors
            elif args.data == 'agnews':
                judged_scores.append([1/4, 1/4, 1/4, 1/4]) # Uniform distribution for errors
            else:
                judged_scores.append(0.5)  # Neutral for errors
        else:
            judged_scores.append(judgment['judge_score'])
    
    # Calculate alignment metrics
    metrics = calculate_alignment_metrics(original_labels, judged_scores, args.data)
    
    # Create comparison plot (skip for vector data as it's more complex)
    if args.data not in ['emotion', 'yahoo', 'agnews']:
        plot_path = create_comparison_plot(original_labels, judged_scores, output_dir, args.model, args.data)
        if plot_path:
            console.print(f"📊 Comparison plot saved to: {plot_path}")
    else:
        console.print("📊 Skipping comparison plot for vector data (emotion/yahoo/agnews)")
    
    # Display results
    console.print("\n" + "="*60)
    console.print(f"🎯 {args.data.upper()} ALIGNMENT RESULTS")
    console.print("="*60)
    
    if metrics and "overall_metrics" in metrics:
        overall = metrics["overall_metrics"]
        
        if args.data == 'emotion':
            console.print(f"📊 Mean Cosine Similarity: {overall['mean_cosine_similarity']:.4f} (±{overall['std_cosine_similarity']:.4f})")
            console.print(f"📊 Mean L2 Distance: {overall['mean_l2_distance']:.4f} (±{overall['std_l2_distance']:.4f})")
            console.print(f"📊 Valid Pairs: {overall['num_valid_pairs']}/{overall['total_pairs']}")
        elif args.data == 'yahoo':
            console.print(f"📊 Mean Cosine Similarity: {overall['mean_cosine_similarity']:.4f} (±{overall['std_cosine_similarity']:.4f})")
            console.print(f"📊 Mean L2 Distance: {overall['mean_l2_distance']:.4f} (±{overall['std_l2_distance']:.4f})")
            console.print(f"📊 Valid Pairs: {overall['num_valid_pairs']}/{overall['total_pairs']}")
        elif args.data == 'agnews':
            console.print(f"📊 Mean Cosine Similarity: {overall['mean_cosine_similarity']:.4f} (±{overall['std_cosine_similarity']:.4f})")
            console.print(f"📊 Mean L2 Distance: {overall['mean_l2_distance']:.4f} (±{overall['std_l2_distance']:.4f})")
            console.print(f"📊 Valid Pairs: {overall['num_valid_pairs']}/{overall['total_pairs']}")
        else:
            console.print(f"📊 Pearson Correlation: {overall['pearson_correlation']:.4f} (p={overall['pearson_p_value']:.4f})")
            console.print(f"📊 Spearman Correlation: {overall['spearman_correlation']:.4f} (p={overall['spearman_p_value']:.4f})")
            console.print(f"📊 Mean Absolute Error: {overall['mean_absolute_error']:.4f}")
            console.print(f"📊 Root Mean Square Error: {overall['root_mean_square_error']:.4f}")
            console.print(f"📊 Valid Pairs: {overall['num_valid_pairs']}/{overall['total_pairs']}")
            
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
    
    for i in range(min(5, len(texts))):
        text = texts[i][:80] + "..." if len(texts[i]) > 80 else texts[i]
        orig_label = original_labels[i]
        judged_score = judged_scores[i]
        
        console.print(f"Text {i+1}: {text}")
        
        if args.data == 'emotion':
            console.print(f"  Original Label: {orig_label}")
            console.print(f"  Judged Score: {judged_score}")
            # Calculate distance between vectors
            distance = calculate_l2_distance(orig_label, judged_score)
            cos_sim = calculate_cosine_similarity(orig_label, judged_score)
            console.print(f"  L2 Distance: {distance:.3f}")
            console.print(f"  Cosine Similarity: {cos_sim:.3f}")
        elif args.data == 'yahoo':
            console.print(f"  Original Label: {orig_label}")
            console.print(f"  Judged Score: {judged_score}")
            # Calculate distance between vectors
            distance = calculate_l2_distance(orig_label, judged_score)
            cos_sim = calculate_cosine_similarity(orig_label, judged_score)
            console.print(f"  L2 Distance: {distance:.3f}")
            console.print(f"  Cosine Similarity: {cos_sim:.3f}")
        elif args.data == 'agnews':
            console.print(f"  Original Label: {orig_label}")
            console.print(f"  Judged Score: {judged_score}")
            # Calculate distance between vectors
            distance = calculate_l2_distance(orig_label, judged_score)
            cos_sim = calculate_cosine_similarity(orig_label, judged_score)
            console.print(f"  L2 Distance: {distance:.3f}")
            console.print(f"  Cosine Similarity: {cos_sim:.3f}")
        else:
            console.print(f"  Original Label: {orig_label:.3f}")
            console.print(f"  Judged Score: {judged_score:.3f}")
            console.print(f"  Difference: {abs(orig_label - judged_score):.3f}")
        
        if args.verbose and i < len(judgments) and "error" not in judgments[i]:
            console.print(f"  Reasoning: {judgments[i]['judge_reasoning'][:100]}...")
            console.print(f"  Confidence: {judgments[i]['judge_confidence']:.3f}")
        console.print()
    
    # Prepare metadata
    metadata = {
        "data_folder": args.data_folder,
        "data_type": args.data,
        "model": args.model,
        "model_provider": get_model_provider(args.model),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "batch_size": args.batch_size,
        "api_key_provided": args.api_key is not None,
        "sample_size": args.sample_size,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_texts": len(texts),
        "successful_judgments": len([j for j in judgments if "error" not in j]),
        "failed_judgments": len([j for j in judgments if "error" in j])
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
    
    console.print(f"📄 All results saved to: {results_file}")
    
    # If custom output file was specified, also save a combined version
    if args.output_file and args.output_file != results_file:
        combined_data = {
            "metadata": metadata,
            "alignment_metrics": metrics,
            "results": []
        }
        
        for i, (text, orig_label, judgment, judged_score) in enumerate(zip(texts, original_labels, judgments, judged_scores)):
            combined_data["results"].append({
                "text_index": i,
                "text": text,
                "original_label": orig_label,
                "judge_score": judged_score,
                "judge_reasoning": judgment.get('judge_reasoning', ''),
                "judge_confidence": judgment.get('judge_confidence', 0.0),
                "judgment_details": judgment
            })
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2)
        console.print(f"💾 Combined results also saved to: {args.output_file}")


if __name__ == '__main__':
    load_dotenv()  # load environment variables from the .env file
    args = parse_arguments()  # parse command line arguments
    try:
        asyncio.run(main(args))  # run main function
    except KeyboardInterrupt:
        logging.info("Ctrl+C pressed!")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1) 