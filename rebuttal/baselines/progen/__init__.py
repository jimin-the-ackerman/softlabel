"""
ProGen: Progressive Zero-shot Dataset Generation via In-context Feedback

A modular implementation of the ProGen framework for synthetic dataset generation.
"""

from rebuttal.baselines.progen.framework import ProGenFramework
from rebuttal.baselines.progen.llm_generators import LLMGenerator, SoftLLMGenerator
from rebuttal.baselines.progen.task_model import TaskSpecificModel
from rebuttal.baselines.progen.influence import InfluenceCalculator
from rebuttal.baselines.progen.schemas import get_output_schema
from rebuttal.baselines.progen.dataset_config import get_dataset_handler
from rebuttal.baselines.progen.utils import get_default_output_dir, save_config, setup_logging

__all__ = [
    'ProGenFramework',
    'LLMGenerator',
    'SoftLLMGenerator',
    'TaskSpecificModel',
    'InfluenceCalculator',
    'get_output_schema',
    'get_dataset_handler',
    'get_default_output_dir',
    'save_config',
    'setup_logging'
] 