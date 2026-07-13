#!/usr/bin/env python3
"""
    Fine-tuning script for language models, supporting training with soft labels.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from transformers import TrainerCallback, TrainerState, TrainerControl
from sklearn.metrics import accuracy_score, f1_score
from peft import LoraConfig, get_peft_model, TaskType
from rich.logging import RichHandler

# Add the project root to the path
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
        )
    )

from rebuttal.loaders import IMDb, SUBJ, SST, Emotion, AGNews, Yahoo
from softprompt.metrics.binary import expected_calibration_error as binary_expected_calibration_error
from softprompt.metrics.multiclass import expected_calibration_error as multiclass_expected_calibration_error


def get_default_output_dir(data: str, model_id: str) -> str:
    """Generate default output directory name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model_id.replace('/', '_').replace('-', '_')
    return f"./results_finetune/{data}/{model_name}/{timestamp}"


def save_config(args: argparse.Namespace, output_dir: Path):
    """Save configuration arguments to a JSON file."""
    config = vars(args)
    config_file = output_dir / "config.json"
    
    # Convert Path objects to strings for JSON serialization
    config_serializable = {}
    for key, value in config.items():
        if isinstance(value, Path):
            config_serializable[key] = str(value)
        else:
            config_serializable[key] = value
    
    with open(config_file, 'w') as f:
        json.dump(config_serializable, f, indent=2, default=str)
    
    logging.info(f"Configuration saved to: {config_file}")


def save_evaluation_results(evaluation_results: dict, output_dir: Path):
    """Save evaluation results to a JSON file."""
    results_file = output_dir / "evaluation_results.json"
    
    # Convert numpy types to Python types for JSON serialization
    results_serializable = {}
    for key, value in evaluation_results.items():
        if isinstance(value, (np.integer, np.floating)):
            results_serializable[key] = value.item()
        elif isinstance(value, np.ndarray):
            results_serializable[key] = value.tolist()
        else:
            results_serializable[key] = value
    
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2, default=str)
    
    logging.info(f"Evaluation results saved to: {results_file}")


def setup_logging(output_dir: str = None):
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


class CustomLoggingCallback(TrainerCallback):
    """
    A custom callback that pretty-prints training progress to the console.
    """

    def __init__(self, logging_steps):
        super().__init__()
        self.train_begin_message_shown = False
        self.logging_steps = logging_steps
        self.last_log_step = -1


    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Called at the beginning of training.
        """
        if state.is_world_process_zero and not self.train_begin_message_shown:
            logging.info("="*80)
            logging.info("🚀 Starting Custom Training Loop 🚀")
            # The model is inside a PeftModel wrapper, so we access the base model
            model = kwargs.get('model')
            if hasattr(model, 'base_model'):
                model_name = model.base_model.model.__class__.__name__
            else:
                model_name = model.__class__.__name__
            logging.info(f"Model: {model_name}")
            logging.info(f"Total optimization steps: {state.max_steps}")
            logging.info(f"Number of epochs: {args.num_train_epochs}")
            logging.info(f"Logging steps: {self.logging_steps}")
            logging.info("="*80)
            self.train_begin_message_shown = True


    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Called when logs are available. Print logs in a single line format.
        """

        if state.is_world_process_zero and logs is not None:
            # Check if this is an evaluation log
            eval_metrics = {k: v for k, v in logs.items() if k.startswith('eval_')}
            
            if eval_metrics:
                # Handle evaluation logs with multi-line format
                logging.info("-" * 50)
                logging.info(f"Evaluation at step {state.global_step}:")
                
                for key, value in eval_metrics.items():
                    clean_key = key.replace('eval_', '').replace('_', ' ').title()
                    if isinstance(value, float):
                        logging.info(f"  - {clean_key}: {value:.4f}")
                    else:
                        logging.info(f"  - {clean_key}: {value}")
                logging.info("-" * 50)
            else:
                # Handle training logs in single line format
                log_parts = [f"Step: {state.global_step:>4}", f"Epoch: {state.epoch:.2f}"]
                
                # Add loss if available
                if 'loss' in logs:
                    log_parts.append(f"Loss: {logs['loss']:.3f}")
                
                # Add learning rate if available
                if 'learning_rate' in logs:
                    log_parts.append(f"LR: {logs['learning_rate']:.2e}")
                
                # Add gradient norm if available
                if 'grad_norm' in logs:
                    log_parts.append(f"GradNorm: {logs['grad_norm']:.3f}")
                
                # Add progress
                progress = state.global_step / state.max_steps
                log_parts.append(f"Progress: {progress:.1%}")
                
                log_message = " | ".join(log_parts)
                logging.info(log_message)


    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Called at the end of each epoch.
        """
        if state.is_world_process_zero and state.epoch > 0:
            logging.info(f"🎉 Epoch {int(state.epoch)} completed! 🎉")

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Called at the end of training.
        """
        if state.is_world_process_zero:
            logging.info("="*80)
            logging.info("✅ Custom Training Loop Finished ✅")
            if state.log_history:
                 final_log = state.log_history[-1]
                 if 'loss' in final_log:
                     logging.info(f"Final Loss: {final_log['loss']:.4f}")
                 # Log best metric
                 if state.best_metric is not None:
                     logging.info(f"Best {args.metric_for_best_model}: {state.best_metric:.4f} at step {state.best_global_step}")
            logging.info("="*80)


class CustomTrainer(Trainer):
    """Custom trainer that handles both soft and hard labels during training.
    
    Args:
        use_soft_labels (bool): If True, uses soft_labels for training with KL divergence loss.
                               If False, uses labels for training with cross-entropy loss.
    """
    
    def __init__(self, use_soft_labels=False, *args, **kwargs):
        self.use_soft_labels = use_soft_labels
        super().__init__(*args, **kwargs)

    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # For soft label training, use soft_labels; otherwise use labels

        if self.use_soft_labels:
            if self.model.training:
                # training on soft labels
                labels = inputs.pop("soft_labels")
                _ = inputs.pop("labels", None)
            else:
                # evaluating with hard labels
                labels = inputs.pop("labels")
                _ = inputs.pop("soft_labels", None)
        else:
            if self.model.training:
                # training on hard labels
                labels = inputs.pop("labels")
                _ = inputs.pop("soft_labels", None)
            else:
                # evaluating with hard labels
                labels = inputs.pop("labels")
                _ = inputs.pop("soft_labels", None)

        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.model.training:
            if self.use_soft_labels:
                loss_function = nn.KLDivLoss(reduction='batchmean')
                loss = loss_function(
                    F.log_softmax(logits, dim=-1), labels
                )
            else:
                loss_function = nn.CrossEntropyLoss()
                loss = loss_function(
                    logits.view(-1, self.model.config.num_labels),
                    labels.view(-1)  # assuming labels are integer class IDs
                )
        else:
            loss_function = nn.CrossEntropyLoss()
            loss = loss_function(
                logits.view(-1, self.model.config.num_labels),
                labels.view(-1)  # assuming labels are integer class IDs
            )
        
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):  # TODO: add type hint
    """Compute evaluation metrics."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    
    # Convert logits to probabilities for calibration error
    probabilities = F.softmax(torch.tensor(logits), dim=1).numpy()
    
    # Determine if this is a binary or multiclass task
    num_classes = probabilities.shape[1]
    
    if num_classes == 2:
        # Binary classification: use binary ECE with positive class probability
        positive_probs = probabilities[:, 1]
        ece = binary_expected_calibration_error(labels, positive_probs)
    else:
        # Multiclass classification: use multiclass ECE with full probability matrix
        ece = multiclass_expected_calibration_error(labels, probabilities)
    
    return {
        'eval_accuracy': accuracy_score(labels, predictions),
        'eval_f1_macro': f1_score(labels, predictions, average='macro'),
        'eval_ece': ece
    }


def get_parameter_stats(model) -> tuple[int, int]:
    """Get parameter statistics for the model."""
    total_params = 0
    trainable_params = 0
    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    return total_params, trainable_params


def tokenizer_function(examples, tokenizer: AutoTokenizer, max_length: int = 512):
    """Tokenize text examples."""
    return tokenizer(
        examples['text'], padding='do_not_pad',
        truncation=True, max_length=max_length,
    )


def main():

    parser = argparse.ArgumentParser(description="Fine-tune language models with soft labels")
    
    # Data arguments
    parser.add_argument("--data", type=str, default="imdb",
                        choices=["imdb", "subj", "sst", "emotion", "agnews", "yahoo"],
                       help="Dataset to use for training")
    parser.add_argument("--synthetic_data_folder", type=str, 
                       default=None,
                       help="Path to synthetic dataset")
    
    # Model arguments
    parser.add_argument("--model_id", type=str, default="distilbert-base-uncased",
                       help="Model identifier from Hugging Face Hub")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length for tokenization")
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", default=True,
                       help="Whether to use LoRA fine-tuning")
    parser.add_argument("--lora_rank", type=int, default=4,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=None,
                       help="LoRA alpha (defaults to 2 * rank)")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout rate")
    parser.add_argument("--target_modules", type=str, nargs="+", default=None,
                       help="Target modules for LoRA (auto-detected if not specified)")
    parser.add_argument("--freeze_pre_classifier", action='store_true', default=False)

    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128,
                       help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=None,
                       help="Evaluation batch size (defaults to batch_size)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Warmup ratio")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                       choices=["linear", "cosine", "constant"],
                       help="Learning rate scheduler type")
    parser.add_argument("--weight_decay", type=float, default=0.001,
                       help="Weight decay")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for checkpoints (defaults to ../results_finetune/data/model_timestamp)")
    parser.add_argument("--logging_dir", type=str, default=None,
                       help="Logging directory (defaults to output_dir/logs)")
    parser.add_argument("--save_steps", type=int, default=100,
                       help="Save steps")
    parser.add_argument("--eval_steps", type=int, default=100,
                       help="Evaluation steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Logging steps")
    parser.add_argument("--save_total_limit", type=int, default=2,
                       help="Maximum number of checkpoints to save")
    
    # Other arguments
    parser.add_argument("--fp16", action="store_true", default=True,  # FIXME:
                       help="Whether to use fp16 training")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--use_soft_labels", action="store_true",
                       help="Whether to use soft labels for training (False for hard labels). Margin filtering works regardless of this setting.")
    parser.add_argument("--synthetic_data_limit", type=int, default=None,
                       help="Limit number of synthetic examples (defaults to min(original_train_size, synthetic_size))")
    parser.add_argument("--margin", type=float, default=0.0,
                       help="Margin for filtering soft labels. Examples with P(class=1) in [0.5-margin, 0.5+margin] are filtered out")
    
    args = parser.parse_args()
    
    # Set defaults
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    if args.lora_alpha is None:
        args.lora_alpha = args.lora_rank * 2
    
    # Set random seed (TODO: this is important because
    # it controls the subsampling process taken to match the
    # size of the gold training data.)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup paths
    root_dir = Path(__file__).parent.parent.parent
    output_dir = Path(args.output_dir) if args.output_dir else Path(get_default_output_dir(args.data, args.model_id))
    logging_dir = Path(args.logging_dir) if args.logging_dir else Path(output_dir / "logs")
    
    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(str(output_dir))
    
    # Save configuration
    save_config(args, output_dir)
    
    logging.info(f"Root directory: {root_dir}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Logging directory: {logging_dir}")
    
    # Load data
    logging.info(f"Loading {args.data} dataset...")
    if args.data == 'imdb':
        tokenized_datasets = IMDb.load_oracle_dataset(root="./data/imdb")
    elif args.data == 'subj':
        tokenized_datasets = SUBJ.load_oracle_dataset()
    elif args.data == 'sst':
        tokenized_datasets = SST.load_oracle_dataset(binary=True)
    elif args.data == 'emotion':
        tokenized_datasets = Emotion.load_oracle_dataset(undersample=True)
    elif args.data == 'agnews':
        tokenized_datasets = AGNews.load_oracle_dataset()
    elif args.data == 'yahoo':
        tokenized_datasets = Yahoo.load_oracle_dataset()  # TODO:
    else:
        raise ValueError(f"Unknown dataset: {args.data}")
    
    logging.info(f"Tokenized datasets: {tokenized_datasets}")
    
    # Load synthetic data if requested
    if args.synthetic_data_folder is not None:
        synthetic_data_folder = Path(args.synthetic_data_folder)
        if not synthetic_data_folder.exists():
            raise ValueError(f"synthetic_data_folder does not exist: {synthetic_data_folder}")
        
        logging.info(f"Loading synthetic data from {args.synthetic_data_folder}...")
        if args.data == 'imdb':
            synthetic_ds = IMDb.load_synthetic_dataset(root=args.synthetic_data_folder)
        elif args.data == 'subj':
            synthetic_ds = SUBJ.load_synthetic_dataset(root=args.synthetic_data_folder)
        elif args.data == 'sst':
            synthetic_ds = SST.load_synthetic_dataset(root=args.synthetic_data_folder)
        elif args.data == 'emotion':
            synthetic_ds = Emotion.load_synthetic_dataset(root=args.synthetic_data_folder)
        elif args.data == 'agnews':
            synthetic_ds = AGNews.load_synthetic_dataset(root=args.synthetic_data_folder)
        elif args.data == 'yahoo':
            synthetic_ds = Yahoo.load_synthetic_dataset(root=args.synthetic_data_folder)
        else:
            raise ValueError(f"Unknown dataset for synthetic data: {args.data}")
        
        # Get original training data size for comparison
        original_train_size = len(tokenized_datasets['train'])
        synthetic_size = len(synthetic_ds)
        
        # Apply margin filtering if margin > 0
        if args.margin > 0:
            logging.info(f"Applying margin filtering with margin={args.margin}...")
            
            def filter_by_margin(example):  # TODO: add type hint
                # Filter based on soft_labels (original probabilities)
                if 'soft_labels' in example:
                    soft_label = example['soft_labels']
                    
                    # For binary tasks, filter based on P(class=1)
                    if isinstance(soft_label, (list, np.ndarray)) and len(soft_label) == 2:
                        p_class1 = soft_label[1]
                        # Keep examples outside the margin range [0.5-margin, 0.5+margin]
                        # i.e., keep examples where p_class1 < 0.5-margin OR p_class1 > 0.5+margin
                        return p_class1 < (0.5 - args.margin) or p_class1 > (0.5 + args.margin)
                    elif isinstance(soft_label, (list, np.ndarray)) and len(soft_label) > 2:
                        # For multiclass, filter based on max probability
                        max_prob = max(soft_label)
                        # Keep examples where max probability is outside the margin range
                        # This is a simplified approach - you might want to adjust based on your needs
                        return max_prob < (1.0/len(soft_label) - args.margin) or \
                            max_prob > (1.0/len(soft_label) + args.margin)
                    else:
                        # For single probability values (binary case)
                        p_class1 = soft_label
                        return p_class1 < (0.5 - args.margin) or p_class1 > (0.5 + args.margin)
                return True
            
            # Apply filtering
            synthetic_ds = synthetic_ds.filter(filter_by_margin)
            filtered_size = len(synthetic_ds)
            logging.info(f"Margin filtering: {synthetic_size} -> {filtered_size} examples (removed {synthetic_size - filtered_size})")
            synthetic_size = filtered_size
        
        # Calculate synthetic_data_limit if not provided
        if args.synthetic_data_limit is None:
            args.synthetic_data_limit = min(original_train_size, synthetic_size)
            logging.info(f"Auto-calculated synthetic_data_limit: {args.synthetic_data_limit} (min of {original_train_size} original, {synthetic_size} synthetic)")
        
        # Apply synthetic data limit
        if args.synthetic_data_limit and args.synthetic_data_limit < synthetic_size:
            synthetic_ds = synthetic_ds.shuffle(seed=args.seed).select(range(args.synthetic_data_limit))
            logging.info(f"Applied synthetic_data_limit: {synthetic_size} -> {args.synthetic_data_limit} examples")
        
        logging.info(f"Synthetic dataset: {synthetic_ds}")
    else:
        synthetic_ds = None
        logging.info("No synthetic data path provided. Using oracle dataset for training.")
    
    # Initialize tokenizer
    logging.info(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    # Tokenize datasets
    def tokenize(examples):
        return tokenizer_function(examples, tokenizer, args.max_length)


    if synthetic_ds is not None:
        logging.info("Tokenizing synthetic dataset...")
        synthetic_ds = synthetic_ds.map(tokenize, batched=True, batch_size=50, num_proc=8)
        synthetic_ds = synthetic_ds.remove_columns('text')
        synthetic_ds.set_format('torch')
    
    logging.info("Tokenizing gold datasets...")
    tokenized_datasets = tokenized_datasets.map(tokenize, batched=True, batch_size=50, num_proc=8)
    tokenized_datasets = tokenized_datasets.remove_columns('text')
    tokenized_datasets.set_format('torch')
    
    # Initialize model
    logging.info(f"Loading model: {args.model_id}")
    
    # Set number of labels based on dataset
    if args.data == 'emotion':
        num_labels = 6  # emotion has 6 classes
    elif args.data == 'agnews':
        num_labels = 4
    elif args.data == '20newsgroups':
        num_labels = 20
    elif args.data == 'yahoo':
        num_labels = 10
    else:
        num_labels = 2  # imdb, subj, sst are binary classification
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id, num_labels=num_labels
    )
    
    # Apply LoRA if requested
    if args.use_lora:
        logging.info("Applying LoRA configuration...")
        
        # Auto-detect target modules if not specified
        if args.target_modules is None:
            if args.model_id == 'distilbert-base-uncased':
                target_modules = ['q_lin', 'v_lin']
            elif args.model_id == 'allenai/longformer-base-4096':
                target_modules = ['query', 'value']
            elif args.model_id.startswith('answerdotai/ModernBERT'):
                target_modules = ['Wqkv']
            else:
                raise ValueError(f"Unknown model type for LoRA target modules: {args.model_id}")
        else:
            target_modules = args.target_modules
            
        logging.info(f"Target modules: {target_modules}")
        
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias='none',
            task_type=TaskType.SEQ_CLS
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    if args.freeze_pre_classifier:
        # Freeze specific pre-classifier layers based on model type
        if args.model_id.startswith('answerdotai/ModernBERT'):
            for param in model.base_model.head.parameters():
                param.requires_grad = False
        elif args.model_id.startswith('distilbert'):
            for param in model.base_model.pre_classifier.parameters():
                param.requires_grad = False
        elif args.model_id.startswith('allenai/longformer-base'):
            pass
        logging.info(f"Freezed the pre-classifer weights of {args.model_id}")
        
    # Print parameter statistics
    total_params, trainable_params = get_parameter_stats(model)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        logging_dir=str(logging_dir),
        logging_strategy="steps", # Enable logging to get loss values
        logging_steps=args.logging_steps,
        eval_strategy='steps',
        eval_steps=args.eval_steps,
        save_strategy='steps',
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        metric_for_best_model='f1_macro' if args.data == 'emotion' else 'accuracy',
        load_best_model_at_end=True,
        fp16=args.fp16,
        fp16_full_eval=args.fp16,
        report_to=[],  # Disable all default callbacks but keep DefaultFlowCallback for on_log
        disable_tqdm=False,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    
    # Setup data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Setup trainer
    if args.use_soft_labels:
        logging.info("Minimizing KL divergence against soft label targets.")
    trainer = CustomTrainer(
        use_soft_labels=args.use_soft_labels,
        model=model,
        args=training_args,
        train_dataset=synthetic_ds if synthetic_ds is not None else tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[CustomLoggingCallback(args.logging_steps)],
    )
    
    # Keep DefaultFlowCallback for on_log but remove PrinterCallback
    trainer.callback_handler.callbacks = [cb for cb in trainer.callback_handler.callbacks 
                                         if isinstance(cb, CustomLoggingCallback) or 
                                         cb.__class__.__name__ == 'DefaultFlowCallback']
    
    # Start training
    logging.info("Starting training...")
    trainer.train()
    
    # Evaluate final model
    logging.info("Evaluating final model...")
    evaluation_results = trainer.evaluate()
    logging.info("Final Evaluation Results:")
    logging.info(json.dumps(evaluation_results, indent=2, default=str))
    
    # Save evaluation results
    save_evaluation_results(evaluation_results, output_dir)
    
    logging.info("Training completed!")
    if args.synthetic_data_folder:
        logging.info(f"Just in case you forgot, synthetic data was loaded from: {args.synthetic_data_folder}")
    logging.info(f"Did we use soft labels for training? {args.use_soft_labels}.")


if __name__ == "__main__":
    main() 