"""
Dataset configuration and handlers for ProGen framework.
"""

import random
import numpy as np
from typing import Dict, List, Any, Union
from rebuttal.baselines.progen.templates import (
    get_hard_system_prompt, get_soft_system_prompt,
    get_hard_human_prompt, get_soft_human_prompt
)


# Dataset configurations
DATASET_CONFIGS = {
    "imdb": {
        "type": "binary",
        "labels": ["negative", "positive"],
        "label_mapping": {"negative": 0, "positive": 1},
        "template_var": "sentiment",
        "num_classes": 2
    },
    "sst": {
        "type": "binary",
        "labels": ["negative", "positive"],
        "label_mapping": {"negative": 0, "positive": 1},
        "template_var": "sentiment",
        "num_classes": 2
    },
    "subj": {
        "type": "binary",
        "labels": ["objective", "subjective"],
        "label_mapping": {"objective": 0, "subjective": 1},
        "template_var": "subjectivity",
        "num_classes": 2
    },
    "emotion": {
        "type": "multiclass",
        "labels": ["sadness", "joy", "love", "anger", "fear", "surprise"],
        "label_mapping": {label: i for i, label in enumerate(["sadness", "joy", "love", "anger", "fear", "surprise"])},
        "template_var": "emotion",
        "num_classes": 6
    }
}


class BaseTemplateFactory:
    """Base class for creating templates with shared functionality."""
    
    def create_system_prompt(self, dataset: str, template_type: str, cot: bool = False) -> str:
        """Create system prompt for a dataset and template type."""
        if template_type == "hard":
            return get_hard_system_prompt(dataset, cot)
        else:
            return get_soft_system_prompt(dataset, cot)
    
    def create_human_prompt(self, dataset: str, template_type: str) -> str:
        """Create human prompt for a dataset and template type."""
        config = DATASET_CONFIGS[dataset]
        var_name = config["template_var"]
        
        if template_type == "hard":
            return get_hard_human_prompt(dataset, var_name)
        else:
            return get_soft_human_prompt(dataset, var_name)


class BaseScoreSampler:
    """Base class for score sampling with shared functionality."""
    
    @staticmethod
    def sample_uniform_binary(is_positive: bool, margin: float = 0.2, **kwargs) -> float:
        """
        Sample uniform binary score with margin to avoid ambiguous middle region.
        
        Args:
            is_positive: True for positive class, False for negative class
            margin: Margin from 0.5 to avoid (e.g., 0.2 means avoid [0.3, 0.7])
            **kwargs: Additional parameters
            
        Returns:
            A float score in the appropriate range
        """
        if is_positive:
            # Sample from [0.5 + margin, 1.0]
            return random.uniform(0.5 + margin, 1.0)
        else:
            # Sample from [0.0, 0.5 - margin]
            return random.uniform(0.0, 0.5 - margin)
    
    @staticmethod
    def sample_beta_binary(is_positive: bool, margin: float = 0.2, **kwargs) -> float:
        """
        Sample beta binary score with margin to avoid ambiguous middle region.
        
        Args:
            is_positive: True for positive class, False for negative class
            margin: Margin from 0.5 to avoid (e.g., 0.2 means avoid [0.3, 0.7])
            **kwargs: Additional parameters for beta distribution
            
        Returns:
            A float score in the appropriate range
        """
        if is_positive:
            # Beta distribution for high scores (0.5 + margin to 1.0)
            alpha_high = kwargs.get('alpha_high', 5.0)
            beta_high = kwargs.get('beta_high', 1.5)
            # Transform beta(0,1) to beta(0.5 + margin, 1.0)
            beta_sample = random.betavariate(alpha_high, beta_high)
            return 0.5 + margin + (1.0 - (0.5 + margin)) * beta_sample
        else:
            # Beta distribution for low scores (0.0 to 0.5 - margin)
            alpha_low = kwargs.get('alpha_low', 1.5)
            beta_low = kwargs.get('beta_low', 5.0)
            # Transform beta(0,1) to beta(0.0, 0.5 - margin)
            beta_sample = random.betavariate(alpha_low, beta_low)
            return (0.5 - margin) * beta_sample


class BinaryDatasetHandler:
    """Handler for binary classification datasets (IMDb, SST, SUBJ)."""
    
    def __init__(self, dataset: str):
        self.dataset = dataset
        self.config = DATASET_CONFIGS[dataset]
        self.template_factory = BaseTemplateFactory()
    
    def get_labels(self) -> List[str]:
        """Get dataset labels."""
        return self.config["labels"]
    
    def get_num_classes(self) -> int:
        """Get number of classes."""
        return self.config["num_classes"]
    
    def create_templates(self, template_type: str, cot: bool = False) -> Dict[str, Any]:
        """Create templates for the dataset."""
        from langchain_core.prompts import ChatPromptTemplate
        
        system_prompt = self.template_factory.create_system_prompt(self.dataset, template_type, cot)
        human_prompt = self.template_factory.create_human_prompt(self.dataset, template_type)
        
        template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
        
        return {self.dataset: template}
    
    def prepare_template_inputs(self, label_or_score, examples_block: str, template_type: str, **kwargs) -> Dict[str, Any]:
        """Prepare template inputs for binary dataset."""
        template_inputs = {'examples_block': examples_block}
        
        if template_type == "hard":
            # Add the original template variable (e.g., 'sentiment', 'subjectivity')
            template_inputs[self.config["template_var"]] = label_or_score
            # Don't provide label - let LLM determine the numeric label
        else:  # soft
            # For binary soft templates, we only need the score
            template_inputs['score'] = label_or_score
            # Don't provide label - let LLM determine the numeric label
        
        return template_inputs
    

class MulticlassDatasetHandler:
    """Handler for multiclass classification datasets (Emotion)."""
    _supported = ('emotion',)
    def __init__(self, dataset: str):
        assert dataset in self._supported
        self.dataset = dataset
        self.config = DATASET_CONFIGS[dataset]
        self.template_factory = BaseTemplateFactory()
    
    def get_labels(self) -> List[str]:
        """Get dataset labels."""
        return self.config["labels"]
    
    def get_num_classes(self) -> int:
        """Get number of classes."""
        return self.config["num_classes"]
    
    def create_templates(self, template_type: str, cot: bool = False) -> Dict[str, Any]:
        """Create templates for the dataset."""
        from langchain_core.prompts import ChatPromptTemplate
        
        system_prompt = self.template_factory.create_system_prompt(self.dataset, template_type, cot)
        human_prompt = self.template_factory.create_human_prompt(self.dataset, template_type)
        
        template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
        
        return {self.dataset: template}
    
    def prepare_template_inputs(self, label_or_score, examples_block: str, template_type: str, **kwargs) -> Dict[str, Any]:
        """Prepare template inputs for multiclass dataset."""
        template_inputs = {'examples_block': examples_block}
        
        if template_type == "hard":
            # Add the original template variable (e.g., 'emotion')
            template_inputs[self.config["template_var"]] = label_or_score
            # Don't provide label - let LLM determine the numeric label
        else:  # soft
            # For emotion soft templates, parse the comma-separated string
            if isinstance(label_or_score, str) and ',' in label_or_score:
                # Parse comma-separated string to list of floats
                probabilities = [float(x.strip()) for x in label_or_score.split(",")]
                template_inputs['probabilities'] = probabilities
            else:
                # Fallback to single score (for backward compatibility)
                template_inputs['score'] = label_or_score
            # Don't provide label - let LLM determine the numeric label
        
        return template_inputs
    
    def sample_score(self, label: str, distribution: str = "uniform", **kwargs) -> float:
        """Sample intensity score for multiclass dataset."""
        # For emotion, we use high intensity (0.8-1.0) for the target emotion
        if distribution == "uniform":
            return random.uniform(0.8, 1.0)
        elif distribution == "beta":
            alpha_high = kwargs.get('alpha_high', 5.0)
            beta_high = kwargs.get('beta_high', 1.5)
            return random.betavariate(alpha_high, beta_high)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
    
    def sample_dirichlet_vector(self, alpha: float = 1.0) -> str:
        """Sample 6-dimensional probability vector from Dirichlet distribution."""
        vector = np.random.dirichlet([alpha] * 6)
        return ",".join([f"{x:.3f}" for x in vector])
    
    def get_label_index(self, label: str) -> int:
        """Get index for a given label."""
        return self.config["label_mapping"].get(label, 1)  # Default to joy (index 1)
    


def get_dataset_handler(dataset: str) -> Union[BinaryDatasetHandler, MulticlassDatasetHandler]:
    """Factory function to get appropriate dataset handler."""
    config = DATASET_CONFIGS[dataset]
    
    if config["type"] == "binary":
        return BinaryDatasetHandler(dataset)
    else:  # multiclass
        return MulticlassDatasetHandler(dataset) 
