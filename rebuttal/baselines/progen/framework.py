"""
Main ProGen framework implementation.
"""

import os
import logging
import random
import json
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict

# Local imports
from rebuttal.baselines.progen.llm_generators import LLMGenerator, SoftLLMGenerator
from rebuttal.baselines.progen.task_model import TaskSpecificModel
from rebuttal.baselines.progen.influence import InfluenceCalculator
from rebuttal.baselines.progen.dataset_config import get_dataset_handler, BaseScoreSampler
from rebuttal.loaders import IMDb, SUBJ, SST, Emotion


class ProGenFramework:
    """Main ProGen framework implementation."""
    
    def __init__(self, args):
        """Initialize the ProGen framework."""
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components based on template type
        if args.llm_template_type == "hard":
            self.llm = LLMGenerator(
                model_name=args.llm_model,
                data=args.data,
                cot=args.cot
            )
        else:  # soft
            self.llm = SoftLLMGenerator(
                model_name=args.llm_model,
                data=args.data,
                cot=args.cot
            )
        
        self.tam = TaskSpecificModel(
            model_name=args.tam_model,
            learning_rate=args.tam_learning_rate,
            batch_size=args.tam_batch_size,
            num_epochs=args.tam_num_epochs,
            weight_decay=args.tam_weight_decay,
            data=args.data
        )
        
        if args.feedback_strategy == 'influence':
            self.influence_calculator = InfluenceCalculator(
                recursion_depth=args.shvp_recursion_depth,
                damping=args.shvp_damping,
                data=args.data
            )
    
        # Initialize datasets
        self.training_df = pd.DataFrame(columns=['text', 'labels'])
        self.validation_df = None  # TODO
        self.helpful_examples = None
        
        logging.info("ProGen framework initialized")
    
    def load_oracle_dataset(self):
        """Load the oracle dataset for validation."""
        logging.info(f"Loading oracle dataset for {self.args.data}...")
        
        # Setup data root path
        data_root = Path(self.args.data_root).resolve()
        
        # Load oracle dataset based on the data type
        if self.args.data == 'imdb':
            oracle_datasets = IMDb.load_oracle_dataset(root=data_root / "imdb")
        elif self.args.data == 'subj':
            oracle_datasets = SUBJ.load_oracle_dataset()
        elif self.args.data == 'sst':
            oracle_datasets = SST.load_oracle_dataset(binary=True)
        elif self.args.data == 'emotion':
            oracle_datasets = Emotion.load_oracle_dataset(undersample=True)
        else:
            raise ValueError(f"Unknown dataset: {self.args.data}")
        
        logging.info(f"Oracle datasets: {oracle_datasets}")
        
        # Convert test set to DataFrame for validation
        test_dataset = oracle_datasets['test']
        self.validation_df = pd.DataFrame({
            'text': test_dataset['text'],
            'labels': test_dataset['labels']
        })
        
        # Save validation set if requested
        if self.args.save_validation_data:
            val_file = self.output_dir / "validation_set.csv"
            self.validation_df.to_csv(val_file, index=False)
            logging.info(f"Validation set loaded and saved to {val_file}")
        else:
            logging.info("Validation set loaded (not saved to file)")
        logging.info(f"Validation set size: {len(self.validation_df)}")
    
    async def generate_data_batch(self, iteration: int) -> pd.DataFrame:
        """Generate a new batch of data."""
        logging.info(f"Generating data batch for iteration {iteration + 1}...")
        
        # Determine if we should use in-context examples
        use_in_context = (self.helpful_examples is not None and 
                         random.random() < self.args.feedback_application_prob)
        
        if use_in_context:
            # Sample in-context examples from the helpful examples pool
            in_context_examples = self.sample_in_context_examples(self.helpful_examples)
            logging.info(f"Using {len(in_context_examples)} in-context examples")
            helpful_texts = in_context_examples
        else:
            logging.info("No in-context examples used (random generation)")
            helpful_texts = None
        
        # Get dataset handler to determine labels and generation strategy
        dataset_handler = get_dataset_handler(self.args.data)
        label_pairs = dataset_handler.get_labels()
        
        # Check if we're using soft templates
        is_soft_template = isinstance(self.llm, SoftLLMGenerator)
        
        # Generate inputs based on template type
        if self.args.data == "emotion":
            # For emotion dataset, generate samples for all 6 emotions
            samples_per_label = self.args.batch_size // len(label_pairs)
            
            if is_soft_template:
                # Create Dirichlet probability vectors for all samples
                inputs = []
                for _ in range(self.args.batch_size):
                    dirichlet_vector = dataset_handler.sample_dirichlet_vector(alpha=self.args.d_alpha)  # TODO: 0.1
                    inputs.append(dirichlet_vector)
            else:
                # Create labels list for all samples
                inputs = []
                for i, label in enumerate(label_pairs):
                    inputs.extend([label] * samples_per_label)
                
                # Add remaining samples to make up the total
                remaining = self.args.batch_size - len(inputs)
                if remaining > 0:
                    extra_label = random.choice(label_pairs)
                    inputs.extend([extra_label] * remaining)
            
            # Generate all samples at once
            generation_results = await self.llm.generate_batch(inputs, helpful_texts)
            
        else:
            # For binary classification datasets
            pos_count = self.args.batch_size // 2
            neg_count = self.args.batch_size - pos_count
            if is_soft_template:
                # Get score function
                score_gen_fn = BaseScoreSampler.sample_uniform_binary
                # Generate scores
                pos_scores = [score_gen_fn(True, margin=self.args.margin) for _ in range(pos_count)]
                neg_scores = [score_gen_fn(False, margin=self.args.margin) for _ in range(neg_count)]
                inputs = pos_scores + neg_scores  # [0.32, 0.14, ..., 0.87]
            else:
                # Get string inputs
                inputs = [label_pairs[1]] * pos_count + [label_pairs[0]] * neg_count
            # Generate
            generation_results = await self.llm.generate_batch(inputs, helpful_texts)
                    
        # Store valid texts and labels only
        valid_data: List[Dict] = []
        
        # Extract data from the dictionary returned by generate_batch
        texts = generation_results['text']
        labels = generation_results['label']
        reasonings = generation_results.get('reasoning', [])  # Only present if CoT is enabled
        
        for i, (text, label, input_val) in enumerate(zip(texts, labels, inputs)):
            if text and isinstance(text, str) and text.strip():
                data_row = {
                    'text': text.strip(), 
                    'label': label,  # Use the label from the response
                    'input_val': input_val  # Keep the original input for reference
                }
                
                # Add reasoning if CoT is enabled
                if self.args.cot and i < len(reasonings):
                    data_row['reasoning'] = reasonings[i]
                    
                valid_data.append(data_row)
        logging.info(f"Generated batch with {len(valid_data):,} valid samples")
        
        # Write results to data.jsonl immediately to prevent data loss
        data_file = self.output_dir / "data.jsonl"
        with open(data_file, 'a', encoding='utf-8') as f:
            for data_row in valid_data:
                # Prepare the row for saving (only text and label for compatibility)
                formatted_row = {
                    'label': data_row['label'], 
                    'text': data_row['text']
                }
                
                # Add reasoning if present
                if 'reasoning' in data_row:
                    formatted_row['reasoning'] = data_row['reasoning']
                    
                json.dump(formatted_row, f, ensure_ascii=False)
                f.write('\n')
        logging.info(f"Results written to {data_file}")
        
        # Create training DataFrame with 'labels' for Hugging Face compatibility
        # Only include text and labels for training (exclude reasoning and input_val)
        training_data = [{'text': row['text'], 'labels': row['label']} for row in valid_data]
        batch_df = pd.DataFrame(training_data)
        
        return batch_df
    
    def select_helpful_examples_pool(self, iteration: int) -> List[str]:
        """Select a pool of helpful examples for the next iteration."""
        if self.args.feedback_strategy == 'random':
            # Random selection from training data
            if len(self.training_df) >= self.args.influence_sample_size:
                helpful_examples = random.sample(
                    self.training_df['text'].tolist(), 
                    k=self.args.influence_sample_size
                )
            else:
                # If not enough training data, use all available
                helpful_examples = self.training_df['text'].tolist()
            logging.info(f"Selected {len(helpful_examples)} random examples for helpful pool")
            
        elif self.args.feedback_strategy == 'influence':
            # Train TAM for influence-based selection
            model_save_path = self.output_dir / "tam_checkpoints" / f"iter_{iteration}"
            self.tam.train(self.training_df, str(model_save_path))
            logging.info(f"TAM trained for influence-based selection")
            
            # Compute influence scores
            influence_scores = self.influence_calculator.compute_influence_scores(
                str(model_save_path),
                self.training_df,
                self.validation_df,
                self.args.influence_sample_size
            )
            
            # Add influence scores to training DataFrame
            training_df_with_scores = self.training_df.copy()
            training_df_with_scores['influence'] = influence_scores
            
            # Select top examples by influence score
            helpful_df = \
                training_df_with_scores.nlargest(
                    self.args.influence_sample_size, 'influence'
                )
            helpful_examples = helpful_df['text'].tolist()
            
            logging.info(f"Selected {len(helpful_examples):,} examples by influence for helpful pool")
        
        return helpful_examples
    
    def sample_in_context_examples(self, helpful_examples: List[str]) -> List[str]:
        """Sample in-context examples from the helpful examples pool."""
        if len(helpful_examples) >= self.args.num_in_context_examples:
            return random.sample(helpful_examples, k=self.args.num_in_context_examples)
        else:
            # If pool is smaller than requested, return all available
            return helpful_examples
    
    async def run(self):
        """Run the main ProGen loop."""
        logging.info("=" * 80)
        logging.info("🚀 Starting ProGen Framework 🚀")
        logging.info(f"Total iterations: {self.args.num_iterations}")
        logging.info(f"Batch size: {self.args.batch_size}")
        logging.info(f"Feedback strategy: {self.args.feedback_strategy}")
        logging.info(f"Feedback interval: {self.args.feedback_interval} iterations")
        logging.info("=" * 80)
        
        # Load oracle dataset for validation (FIXME)
        self.load_oracle_dataset()
        
        # Main iterative loop
        for iteration in range(self.args.num_iterations):
            logging.info(f"\n--- ProGen Iteration {iteration + 1}/{self.args.num_iterations} ---")
            
            # Generate new data batch
            new_batch_df = await self.generate_data_batch(iteration)
            
            # Add to training set
            self.training_df = pd.concat([self.training_df, new_batch_df], ignore_index=True)
            logging.info(f"Training set size: {len(self.training_df)}")
            
            # Select helpful examples based on feedback interval
            if (iteration + 1) % self.args.feedback_interval == 0:
                logging.info(f"Updating helpful examples (FI: {self.args.feedback_interval})")
                self.helpful_examples = self.select_helpful_examples_pool(iteration)
            
            # Save intermediate results periodically
            if self.args.save_interval > 0:  # -1; never save
                if (iteration + 1) % self.args.save_interval == 0:
                    intermediate_dir = self.output_dir / "training_sets"
                    intermediate_dir.mkdir(exist_ok=True)
                    intermediate_file = intermediate_dir / f"iter_{iteration}.csv"
                    self.training_df.to_csv(intermediate_file, index=False)
                    logging.info(f"Intermediate results saved to {intermediate_file}")

        # Save final dataset
        final_file = self.output_dir / f"progen_{self.args.data}_dataset.csv"
        self.training_df.to_csv(final_file, index=False)
        logging.info(f"Final dataset saved to {final_file}")
        logging.info(f"Total generated samples: {len(self.training_df)}")
        
        logging.info("=" * 80)
        logging.info("✅ ProGen Framework Completed ✅")
        logging.info("=" * 80) 