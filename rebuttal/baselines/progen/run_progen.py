#!/usr/bin/env python3
"""
ProGen: Progressive Zero-shot Dataset Generation via In-context Feedback

Main entry point for the ProGen framework as described in the EMNLP 2022 Findings paper:
"PROGEN: Progressive Zero-shot Dataset Generation via In-context Feedback" by Jiacheng Ye, et al.

This script implements the iterative feedback loop where a task-specific model (TAM) 
is trained on the current dataset and used to guide the generation of the next batch of data.
"""

import os
import sys
import argparse
import logging
import random
import asyncio
from pathlib import Path
from dotenv import load_dotenv

import torch
import numpy as np

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

# Add the progen folder to the path
progen_folder = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, progen_folder)

# Local imports
from rebuttal.baselines.progen.framework import ProGenFramework
from rebuttal.baselines.progen.utils import get_default_output_dir, save_config, setup_logging


data_to_default_size = {
    'imdb': 50_000,
    'sst': 20_000,
    'subj': 16_000,
    'emotion': 200_000,
}


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="ProGen: Progressive Zero-shot Dataset Generation")
    
    # Data arguments
    parser.add_argument("--data", type=str, default="imdb", choices=["imdb", "sst", "subj", "emotion"],
                       help="Dataset name")
    parser.add_argument("--data_root", type=str, default="./data",
                       help="Root directory for data")
    
    # LLM arguments
    parser.add_argument("--llm_model", type=str, default="gemini-2.0-flash",
                        choices=["gemini-2.0-flash", "gemini-2.5-flash", "gpt-4o",
                                "gpt-4o", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "o4-mini", "o4",
                                "claude-3.5-haiku"],
                       help="LLM model name")
    parser.add_argument("--llm_template_type", type=str, default="hard",
                       choices=["hard", "soft"],
                       help="Type of LLM template to use (hard or soft)")
    parser.add_argument("--score_distribution", type=str, default="uniform",
                       choices=["uniform", "beta"],
                       help="Distribution type for soft template scores")
    parser.add_argument("--margin", type=float, default=0.0,
                       help="Margin for binary score sampling (avoid [0.5-margin, 0.5+margin])")
    parser.add_argument("--cot", action="store_true", default=False,
                       help="Use Chain of Thought instructions in prompts")
    
    # ProGen framework parameters
    parser.add_argument("--feedback_strategy", type=str, default="random",
                       choices=["influence", "random"],
                       help="Strategy for selecting helpful examples")
    parser.add_argument("--sample_size", type=int, default=None,
                       help="Final size of the generated training set")
    parser.add_argument("--batch_size", type=int, default=50,
                       help="Number of samples to generate per iteration")
    parser.add_argument("--feedback_interval", type=int, default=1,
                       help="How often to update helpful examples (in iterations)")
    parser.add_argument("--save_interval", type=int, default=-1,
                       help="How often to save intermediate results (in iterations)")
    parser.add_argument("--num_iterations", type=int, default=None,
                       help="Total number of feedback loops (auto-calculated if None)")
    parser.add_argument("--influence_sample_size", type=int, default=100,
                       help="Number of training samples to score for influence")
    parser.add_argument("--num_in_context_examples", type=int, default=8,
                       help="Number of helpful examples to use for feedback")
    parser.add_argument("--feedback_application_prob", type=float, default=0.5,
                       help="Probability of applying feedback in a given iteration")
    
    parser.add_argument("--d_alpha", type=float, default=0.5, help="For multiclass")
    parser.add_argument("--b_alpha", type=float, default=0.1, help="For binary")

    # TAM parameters
    parser.add_argument("--tam_model", type=str, default="distilbert-base-uncased",
                       help="Task-specific model architecture")
    parser.add_argument("--tam_learning_rate", type=float, default=2e-5,
                       help="TAM learning rate")
    parser.add_argument("--tam_batch_size", type=int, default=16,
                       help="TAM batch size")
    parser.add_argument("--tam_num_epochs", type=int, default=1,
                       help="TAM epochs per iteration")
    parser.add_argument("--tam_weight_decay", type=float, default=0.01,
                       help="TAM weight decay")
    
    # Influence calculation parameters
    parser.add_argument("--shvp_recursion_depth", type=int, default=5000,
                       help="Number of steps for iterative HVP approximation")
    parser.add_argument("--shvp_damping", type=float, default=0.01,
                       help="Damping factor for sHVP calculation")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (auto-generated if None)")
    parser.add_argument("--save_validation_data", action="store_true", default=False,
                       help="Save validation data to CSV file")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set defaults
    if args.num_iterations is None:
        args.num_iterations = args.sample_size // args.batch_size
    
    if args.sample_size is None:
        args.sample_size = data_to_default_size[args.data]

    if args.output_dir is None:
        args.output_dir = get_default_output_dir(args.data, args.llm_model)
    
    return args


def main():
    """Main function."""
    # Load environment variables from .env file
    load_dotenv()
    
    args = parse_arguments()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=False)

    # Setup logging
    setup_logging(args.output_dir)
    
    # Save configuration
    save_config(args, Path(args.output_dir))
    
    # Create and run ProGen framework
    progen = ProGenFramework(args)
    
    # Run the async framework
    asyncio.run(progen.run())


if __name__ == "__main__":
    main() 