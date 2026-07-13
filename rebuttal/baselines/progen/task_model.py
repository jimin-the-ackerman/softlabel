"""
Task-specific model (TAM) for ProGen framework.
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import List

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from datasets import Dataset

from rebuttal.baselines.progen.dataset_config import get_dataset_handler


class TaskSpecificModel:
    """Task-specific model (TAM) for ProGen."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", 
                 learning_rate: float = 2e-5, batch_size: int = 16, 
                 num_epochs: int = 1, weight_decay: float = 0.01, data: str = "imdb"):
        """Initialize the task-specific model."""
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.data = data
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Get dataset handler to determine number of labels
        dataset_handler = get_dataset_handler(data)
        self.num_labels = dataset_handler.get_num_classes()
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=self.num_labels
        )
        
        # Move model to device
        self.model.to(self.device)
        
        logging.info(f"TAM initialized: {model_name}")
        logging.info(f"Number of labels: {self.num_labels}")
        logging.info(f"Device: {self.device}")
    
    def train(self, dataset_df: pd.DataFrame, model_save_path: str):
        """
        Fine-tune the model on the given dataset.
        
        Args:
            dataset_df: DataFrame with 'text' and 'labels' columns
            model_save_path: Path to save the trained model
        """
        logging.info(f"Training TAM on {len(dataset_df)} samples...")

        # Validate required columns
        required_columns = ['text', 'labels']
        missing_columns = [col for col in required_columns if col not in dataset_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}. Available columns: {list(dataset_df.columns)}")

        # Clean and validate the dataset
        dataset_df = dataset_df.copy()
        
        # Remove empty or invalid texts
        dataset_df = dataset_df.dropna(subset=['text'])
        dataset_df = dataset_df[dataset_df['text'].str.strip() != '']
        
        # Ensure labels are integers
        dataset_df['labels'] = dataset_df['labels'].astype(int)
        
        # Validate label range
        min_label = dataset_df['labels'].min()
        max_label = dataset_df['labels'].max()
        if min_label < 0 or max_label >= self.num_labels:
            raise ValueError(f"Labels must be in range [0, {self.num_labels-1}], got [{min_label}, {max_label}]")
        
        logging.info(f"Cleaned dataset: {len(dataset_df)} samples")
        logging.info(f"Label distribution: {dataset_df['labels'].value_counts().to_dict()}")

        # Prepare Hugging Face Dataset
        hf_dataset = Dataset.from_pandas(dataset_df)
        
        def tokenize_function(examples):
            # Ensure text is string and handle any None values
            texts = [str(text) if text is not None else "" for text in examples["text"]]
            
            return self.tokenizer(
                texts, 
                padding="do_not_pad", 
                truncation=True,
                max_length=512,
                return_tensors=None  # Important: don't return tensors here
            )
        
        tokenized_dataset = hf_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=model_save_path,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            logging_dir=os.path.join(model_save_path, 'logs'),
            logging_steps=100,
            save_strategy="epoch",
            remove_unused_columns=True,  # Changed to True
            report_to=[],  # Disable wandb/tensorboard
            dataloader_pin_memory=False,  # Add this to avoid memory issues
        )
        
        # Setup data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Initialize and run trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        try:
            trainer.train()
            logging.info(f"TAM saved to {model_save_path}")
        except Exception as e:
            logging.error(f"Training failed: {e}")
            logging.error(f"Dataset info: {len(dataset_df)} samples, labels: {dataset_df['labels'].unique()}")
            raise

    def load_model(self, model_path: str):
        """Load a trained model from path."""
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        logging.info(f"TAM loaded from {model_path}") 