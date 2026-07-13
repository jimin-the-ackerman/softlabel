"""
LLM generators for ProGen framework.
"""

import os
import logging
import asyncio
import random
from typing import List, Union, Dict

from tqdm import tqdm

# LangChain imports
from langchain.chat_models import init_chat_model

# Local imports
from rebuttal.baselines.progen.schemas import get_output_schema
from rebuttal.baselines.progen.dataset_config import get_dataset_handler


def get_model_provider(model: str) -> str:
    if model.startswith(('gpt', 'o1', 'o3', 'o4')):
        return 'openai'
    elif model.startswith(('gemini', 'gemma')):
        return 'google_genai'
    elif model.startswith('claude'):
        return 'anthropic'
    else:
        raise NotImplementedError("Work in progress...")  # TODO:


class LLMGenerator:
    """LLM generation module for ProGen using LangChain."""
    
    def __init__(self, model_name: str = "gemini-2.0-flash", api_key: str = None, data: str = "imdb",
                template_type: str = "hard", cot: bool = False):
        """Initialize the LLM generator."""
        self.model_name = model_name
        self.data = data
        self.template_type = template_type  # important attribute
        self.cot = cot  # Chain of Thought flag
        
        # Setup API key
        if api_key:
            if model_name.startswith('gemini'):
                os.environ['GOOGLE_API_KEY'] = api_key
                logging.info(f"Using provided Google API key: {api_key[:5]}...")
            else:
                raise NotImplementedError(f"API key setup not implemented for model: {model_name}")
        
        # Initialize the LangChain chat model
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                self.llm = init_chat_model(
                    model=model_name,
                    model_provider=get_model_provider(model_name),
                    temperature=1.0,
                    max_tokens=None,
                    timeout=60.,
                    max_retries=5,
                )
            logging.info(f"LangChain chat model initialized: {model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LangChain chat model: {e}")
        
        # Initialize dataset handler
        self.dataset_handler = get_dataset_handler(data)
        
        # Create templates for the dataset
        self.templates = self.dataset_handler.create_templates(template_type, cot)
        
        # Setup structured output
        self.schema = get_output_schema(data, cot)
        self.structured_llm = self.llm.with_structured_output(self.schema)
        
        logging.info(f"{template_type.title()} templates created for dataset: {data}")
        if cot:
            logging.info("Chain of Thought (CoT) enabled")
        logging.info(f"Output schema: {self.schema.__name__}")
    
    def _create_examples_block(self, examples: List[str]) -> str:
        """Create a block of in-context examples for ProGen."""
        if not examples:
            return ""
        
        # Add instructions for learning from examples
        selected_instruction = "IMPORTANT: Analyze these examples carefully before generating your response!\n\n"
        selected_instruction += "What to learn from the examples:\n"
        selected_instruction += "- Writing style, vocabulary, and tone\n"
        selected_instruction += "- Structure and approach to the task\n"
        selected_instruction += "- How they achieve the target sentiment/subjectivity/emotion\n"
        selected_instruction += "- Quality and authenticity of expression\n\n"
        selected_instruction += "Apply similar techniques and maintain comparable quality in your response."
        
        # ProGen format: F-5 format (no labels in examples)
        examples_block = f"Here are {len(examples)} examples to learn from:\n\n"
        for i, example in enumerate(examples):
            examples_block += f"Example {i+1}: {example}\n\n"
        examples_block += selected_instruction
        
        return examples_block
    
    def _get_default_label(self):
        """Get default label based on dataset type."""
        if self.data == "emotion":
            return [0.0] * 6  # 6-dimensional zero vector
        else:
            return -1  # Binary default
    
    async def generate_batch(self,
        inputs: List[Union[str, float, List[float]]],
        in_context_examples: List[str] = None) -> Dict[str, List]:
        """
        Generate a batch of synthetic data using the LLM.
        
        Args:
            inputs: List of inputs for each sample:
                   - For hard templates: List[str] (e.g., ['positive', 'negative', 'subjective', 'objective', 'joy', etc.])
                   - For soft templates: List[float] or List[List[float]] (e.g., [0.8, 0.2, 0.9, ...])
            in_context_examples: List of in-context example texts for learning
            
        Returns:
            Dictionary with keys 'text', 'label', and 'reasoning' (if CoT enabled), each containing a list of values
        """
        num_samples = len(inputs)
        
        # Create examples block if in-context examples are provided
        examples_block = self._create_examples_block(in_context_examples) if in_context_examples else ""
        
        # Get the template for this dataset
        template = self.templates[self.data]
        
        logging.info(f"Generating {num_samples} samples with {self.template_type} template...")
        if in_context_examples:
            logging.info(f"Using {len(in_context_examples)} in-context examples")
        
        # Create chain inputs for batch processing
        chain_inputs = []
        
        # Prepare template inputs for all samples
        chain_inputs = []
        for i, input_val in enumerate(inputs):
            template_inputs = self.dataset_handler.prepare_template_inputs(
                input_val, examples_block, self.template_type
            )
            chain_inputs.append(template_inputs)
            
            if i == 0:  # Log the first sample for debugging
                logging.debug(f"Sample {i}: input={input_val}")
        
        def extract_response_data(response):
            """Extract data from a single response."""
            if not response or not hasattr(response, 'text'):
                return "", self._get_default_label(), ""
            
            text = response.text.strip() if response.text else ""
            label = getattr(response, 'label', None) or self._get_default_label()
            reasoning = getattr(response, 'reasoning', "") if self.cot else ""
            
            return text, label, reasoning
        
        def process_responses(responses):
            """Process a list of responses and extract data."""
            texts, labels, reasonings = [], [], []
            
            for response in responses:
                text, label, reasoning = extract_response_data(response)
                texts.append(text)
                labels.append(label)
                if self.cot:
                    reasonings.append(reasoning)
            
            return texts, labels, reasonings
        
        try:
            # Try batch processing first
            chain = template | self.structured_llm
            responses = await chain.abatch(
                chain_inputs, 
                config={"max_concurrency": min(num_samples, 10)}
            )
            texts, labels, reasonings = process_responses(responses)
            
        except Exception as e:
            logging.warning(f"Error in batch generation: {e}")
            # Fallback to individual generation
            texts, labels, reasonings = [], [], []
            
            for i in tqdm(range(num_samples), desc="Generating (fallback)"):
                try:
                    messages = template.format_messages(**chain_inputs[i])
                    response = await self.structured_llm.ainvoke(messages)
                    text, label, reasoning = extract_response_data(response)
                    
                    texts.append(text)
                    labels.append(label)
                    if self.cot:
                        reasonings.append(reasoning)
                        
                except Exception as e:
                    logging.warning(f"Error generating sample {i}: {e}")
                    text, label, reasoning = extract_response_data(None)
                    texts.append(text)
                    labels.append(label)
                    if self.cot:
                        reasonings.append(reasoning)
                
                await asyncio.sleep(0.1)  # Rate limiting
        
        # Prepare return dictionary
        result = {
            'text': texts,
            'label': labels
        }
        
        if self.cot:
            result['reasoning'] = reasonings
            
        return result


class SoftLLMGenerator(LLMGenerator):
    """LLM generation module for ProGen using LangChain with soft templates."""
    
    def __init__(self, model_name: str = "gemini-2.0-flash", api_key: str = None, data: str = "imdb", cot: bool = False):
        """Initialize the soft LLM generator."""
        super().__init__(model_name, api_key, data, template_type="soft", cot=cot) 