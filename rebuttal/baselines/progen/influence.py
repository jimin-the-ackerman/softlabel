"""
Influence function calculation module for ProGen framework.
"""

import logging
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import List

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from rebuttal.baselines.progen.dataset_config import get_dataset_handler


class InfluenceCalculator:
    """Influence function calculation module."""
    
    def __init__(self, recursion_depth: int = 5000, damping: float = 0.01, data: str = "imdb"):
        """Initialize the influence calculator."""
        self.recursion_depth = recursion_depth
        self.damping = damping
        self.data = data
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Get dataset handler to determine number of labels
        dataset_handler = get_dataset_handler(data)
        self.num_labels = dataset_handler.get_num_classes()
    
    def reverse_cross_entropy_loss(self, logits: torch.Tensor, 
                                 targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Reverse Cross-Entropy (RCE) loss.
        
        Args:
            logits: Model logits (N, C)
            targets: One-hot encoded targets (N, C)
            
        Returns:
            RCE loss tensor
        """
        probs = F.softmax(logits, dim=1)
        # RCE: -sum_c(probs_c * log(targets_c))
        # Add small epsilon to avoid log(0)
        epsilon = 1e-12
        targets_safe = torch.clamp(targets, epsilon, 1.0 - epsilon)
        rce_loss = -torch.sum(probs * torch.log(targets_safe), dim=1)
        return rce_loss
    
    def compute_influence_scores(self, model_path: str, train_df: pd.DataFrame, 
                               val_df: pd.DataFrame, sample_size: int = 5000) -> List[float]:
        """
        Compute influence scores for training samples.
        
        Args:
            model_path: Path to the trained model
            train_df: Training dataset DataFrame
            val_df: Validation dataset DataFrame
            sample_size: Number of training samples to score
    
        Returns:
            List of influence scores (same length as train_df)
        """
        logging.info(f"Computing influence scores for {sample_size} samples...")
        
        # Load the model
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(self.device)
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Prepare validation data
        val_texts = val_df['text'].tolist()
        val_labels = val_df['labels'].tolist()
        
        # Convert labels to one-hot encoding
        val_labels_onehot = torch.zeros(len(val_labels), self.num_labels)
        val_labels_onehot[range(len(val_labels)), val_labels] = 1.0
        val_labels_onehot = val_labels_onehot.to(self.device)
        
        # Compute validation set gradient (using RCE loss)
        val_grad = self._compute_validation_gradient(model, tokenizer, val_texts, val_labels_onehot)
        
        # Sample training data for influence calculation
        if sample_size < len(train_df):
            sampled_indices = random.sample(range(len(train_df)), sample_size)
            train_df_sampled = train_df.iloc[sampled_indices].reset_index(drop=True)
        else:
            train_df_sampled = train_df
            sampled_indices = list(range(len(train_df)))
        
        # Initialize influence scores
        influence_scores = np.full(len(train_df), -np.inf)  # -inf for unscored samples
        
        # Compute influence scores for sampled data
        for i, (idx, row) in enumerate(zip(sampled_indices, train_df_sampled.itertuples())):
            if i % 100 == 0:
                logging.info(f"Computing influence for sample {i+1}/{len(train_df_sampled)}")
            
            # Compute training sample gradient (using standard CE loss)
            train_grad = self._compute_training_sample_gradient(
                model, tokenizer, row.text, row.labels
            )
            
            # Compute inverse Hessian-vector product
            ihvp = self._compute_inverse_hessian_vector_product(
                model, tokenizer, train_df_sampled, val_grad
            )
            
            # Compute influence score
            influence_score = -torch.dot(val_grad, ihvp).item()
            influence_scores[idx] = influence_score
        
        logging.info("Influence score computation completed")
        return influence_scores.tolist()
    
    def _compute_validation_gradient(self, model: nn.Module, tokenizer: AutoTokenizer,
                                   val_texts: List[str], val_labels_onehot: torch.Tensor) -> torch.Tensor:
        """Compute gradient of validation set using RCE loss."""
        model.zero_grad()
        
        # Tokenize validation texts
        val_inputs = tokenizer(val_texts, padding=True, truncation=True, 
                              max_length=512, return_tensors="pt")
        val_inputs = {k: v.to(self.device) for k, v in val_inputs.items()}
        
        # Forward pass
        val_logits = model(**val_inputs).logits
        
        # Compute RCE loss
        val_loss = self.reverse_cross_entropy_loss(val_logits, val_labels_onehot).mean()
        
        # Backward pass
        val_loss.backward()
        
        # Collect gradients
        val_grad = []
        for param in model.parameters():
            if param.grad is not None:
                val_grad.append(param.grad.view(-1))
        
        val_grad = torch.cat(val_grad)
        return val_grad
    
    def _compute_training_sample_gradient(self, model: nn.Module, tokenizer: AutoTokenizer,
                                        text: str, label: int) -> torch.Tensor:
        """Compute gradient of a single training sample using CE loss."""
        model.zero_grad()
        
        # Tokenize training text
        train_inputs = tokenizer([text], padding=True, truncation=True, 
                                max_length=512, return_tensors="pt")
        train_inputs = {k: v.to(self.device) for k, v in train_inputs.items()}
        
        # Forward pass
        train_logits = model(**train_inputs).logits
        
        # Compute CE loss
        train_loss = F.cross_entropy(train_logits, torch.tensor([label]).to(self.device))
        
        # Backward pass
        train_loss.backward()
        
        # Collect gradients
        train_grad = []
        for param in model.parameters():
            if param.grad is not None:
                train_grad.append(param.grad.view(-1))
        
        train_grad = torch.cat(train_grad)
        return train_grad
    
    def _compute_inverse_hessian_vector_product(self, model: nn.Module, tokenizer: AutoTokenizer,
                                              train_df: pd.DataFrame, v: torch.Tensor) -> torch.Tensor:
        """
        Compute inverse Hessian-vector product using iterative approximation.
        
        Args:
            model: The trained model
            tokenizer: Tokenizer for the model
            train_df: Training dataset
            v: Vector to multiply with inverse Hessian
            
        Returns:
            Approximation of H^(-1) * v
        """
        h = v.clone()
        
        for t in range(self.recursion_depth):
            # Sample a random batch from training data
            batch_indices = random.sample(range(len(train_df)), min(32, len(train_df)))
            batch_df = train_df.iloc[batch_indices]
            
            # Compute Hessian-vector product on this batch
            hv = self._compute_hessian_vector_product(model, tokenizer, batch_df, h)
            
            # Update h: h_{t+1} = v + (1 - λ)h_t - H_t(h_t)
            h = v + (1 - self.damping) * h - hv
            
            if t % 1000 == 0:
                logging.info(f"HVP iteration {t+1}/{self.recursion_depth}")
        
        return h
    
    def _compute_hessian_vector_product(self, model: nn.Module, tokenizer: AutoTokenizer,
                                      batch_df: pd.DataFrame, v: torch.Tensor) -> torch.Tensor:
        """Compute Hessian-vector product on a batch of data."""
        model.zero_grad()
        
        # Tokenize batch
        batch_texts = batch_df['text'].tolist()
        batch_labels = batch_df['labels'].tolist()
        
        batch_inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                                max_length=512, return_tensors="pt")
        batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
        
        # Forward pass
        batch_logits = model(**batch_inputs).logits
        
        # Compute loss
        batch_loss = F.cross_entropy(batch_logits, torch.tensor(batch_labels).to(self.device))
        
        # First backward pass
        batch_loss.backward()
        
        # Collect gradients
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        
        grads = torch.cat(grads)
        
        # Second backward pass for HVP
        model.zero_grad()
        hvp = torch.autograd.grad(grads, model.parameters(), grad_outputs=v, retain_graph=True)
        
        # Collect HVP
        hvp_flat = []
        for grad in hvp:
            if grad is not None:
                hvp_flat.append(grad.view(-1))
        
        return torch.cat(hvp_flat) 