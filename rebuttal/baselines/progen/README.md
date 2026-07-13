# ProGen: Progressive Zero-shot Dataset Generation

A modular implementation of the ProGen framework for synthetic dataset generation via in-context feedback.

## 📁 Module Structure

```
progen/
├── __init__.py              # Package initialization
├── run_progen.py            # Main entry point
├── framework.py             # Main ProGen framework orchestration
├── llm_generators.py        # LLM generation modules (hard/soft templates)
├── task_model.py            # Task-specific model (TAM) implementation
├── influence.py             # Influence function calculation
├── schemas.py               # Structured output schemas
├── dataset_config.py        # Dataset configurations and handlers
├── utils.py                 # Utility functions
└── README.md               # This file
```

## 🚀 Quick Start

### Basic Usage

```bash
# Run with default settings (IMDb, hard templates, influence strategy)
python -m softprompt.baselines.progen.run_progen

# Run with custom settings
python -m softprompt.baselines.progen.run_progen \
    --data emotion \
    --llm_template_type soft \
    --score_distribution uniform \
    --total_dataset_size 1000 \
    --batch_size 100
```

### Using the Shell Script

```bash
# Run with shell script
./shellscripts/run_progen.sh --data imdb --llm_template_type hard

# Run with API key
./shellscripts/run_progen.sh --data sst --api_key YOUR_API_KEY
```

## 📋 Key Components

### 1. **Framework** (`framework.py`)
- Main orchestration class `ProGenFramework`
- Manages the iterative feedback loop
- Coordinates all components

### 2. **LLM Generators** (`llm_generators.py`)
- `LLMGenerator`: Hard template generation
- `SoftLLMGenerator`: Soft template generation with continuous scores
- Both use structured outputs for reliable generation

### 3. **Task-Specific Model** (`task_model.py`)
- `TaskSpecificModel`: Fine-tunes on generated data
- Uses HuggingFace `Trainer` API
- Supports all dataset types

### 4. **Influence Calculator** (`influence.py`)
- `InfluenceCalculator`: Computes influence scores
- Uses Reverse Cross-Entropy (RCE) loss
- Implements iterative Hessian-vector product approximation

### 5. **Dataset Configuration** (`dataset_config.py`)
- Dataset-specific handlers and templates
- `BinaryDatasetHandler`: For IMDb, SST, SUBJ
- `MulticlassDatasetHandler`: For Emotion dataset
- Template factories for hard/soft prompts

### 6. **Structured Outputs** (`schemas.py`)
- Pydantic schemas for reliable LLM responses
- Dataset-specific schemas (IMDB, SST, SUBJ, Emotion)
- Ensures consistent output format

## 🎯 Supported Datasets

| Dataset | Type | Labels | Template Variable |
|---------|------|--------|-------------------|
| IMDb | Binary | negative, positive | sentiment |
| SST | Binary | negative, positive | sentiment |
| SUBJ | Binary | objective, subjective | subjectivity |
| Emotion | Multiclass | sadness, joy, love, anger, fear, surprise | emotion |

## 🔧 Configuration Options

### Data Parameters
- `--data`: Dataset name (imdb, sst, subj, emotion)
- `--data_root`: Root directory for data

### LLM Parameters
- `--llm_model`: LLM model name (default: gemini-2.0-flash)
- `--api_key`: API key for LLM service
- `--llm_template_type`: Template type (hard, soft)
- `--score_distribution`: Score distribution (uniform, beta)

### Framework Parameters
- `--feedback_strategy`: Strategy (influence, random)
- `--total_dataset_size`: Final dataset size
- `--batch_size`: Samples per iteration
- `--feedback_interval`: Feedback update frequency
- `--num_in_context_examples`: Number of helpful examples

### TAM Parameters
- `--tam_model`: Model architecture
- `--tam_learning_rate`: Learning rate
- `--tam_batch_size`: Batch size
- `--tam_num_epochs`: Epochs per iteration

## 🔄 Feedback Strategies

### 1. **Influence Strategy**
- Uses influence functions to select helpful examples
- Computes influence scores using RCE loss
- Selects top examples by influence score

### 2. **Random Strategy**
- Randomly selects examples from training set
- Faster but less targeted

## 📊 Template Types

### 1. **Hard Templates**
- Uses discrete labels (positive/negative, etc.)
- Direct label specification
- Simpler prompt structure

### 2. **Soft Templates**
- Uses continuous scores (0-1 scale)
- Supports uniform and beta distributions
- More nuanced control over generation

## 🏗️ Architecture Benefits

### Modularity
- Each component is self-contained
- Easy to modify or extend individual parts
- Clear separation of concerns

### Reusability
- Components can be used independently
- Easy to test individual modules
- Configurable for different use cases

### Maintainability
- Clean, organized code structure
- Comprehensive documentation
- Type hints throughout

### Extensibility
- Easy to add new datasets
- Simple to implement new feedback strategies
- Flexible template system

## 🐛 Debugging

### VS Code Debug Configurations
Multiple debug configurations are available in `.vscode/launch.json`:

- **Hard Templates (IMDb)**: Basic hard template generation
- **Soft Templates (IMDb)**: Soft template with uniform distribution
- **Emotion Dataset**: Multiclass emotion generation
- **Influence Strategy**: Influence-based feedback
- **Beta Distribution**: Soft templates with beta distribution
- **Large Batch, Rare Feedback**: Different batch/feedback settings

### Logging
- Rich console logging with progress bars
- File logging to `progen.log`
- Detailed configuration logging

## 📈 Output Structure

```
results_progen/
└── {dataset}/
    └── {timestamp}/
        ├── config.json              # Configuration parameters
        ├── progen.log               # Execution log
        ├── validation_set.csv       # Oracle validation set
        ├── training_set_iter_{i}.csv # Intermediate results
        ├── tam_iter_{i}/            # Trained models per iteration
        └── progen_{dataset}_dataset.csv # Final generated dataset
```

## 🔗 Dependencies

- **LangChain**: LLM integration and structured outputs
- **Transformers**: Task-specific model training
- **PyTorch**: Deep learning framework
- **Pandas**: Data manipulation
- **Rich**: Enhanced logging
- **Pydantic**: Structured output validation

## 📝 Usage Examples

### Basic IMDb Generation
```python
from rebuttal.baselines.progen import ProGenFramework

# Create framework
progen = ProGenFramework(args)

# Run the framework
await progen.run()
```

### Custom LLM Generator
```python
from rebuttal.baselines.progen import LLMGenerator

# Create hard template generator
generator = LLMGenerator(
    model_name="gemini-2.0-flash",
    data="imdb"
)

# Generate batch
texts = await generator.generate_batch(
    labels=["positive", "negative"],
    helpful_examples=["example1", "example2"]
)
```

### Influence Calculation
```python
from rebuttal.baselines.progen import InfluenceCalculator

# Create influence calculator
calculator = InfluenceCalculator(data="imdb")

# Compute influence scores
scores = calculator.compute_influence_scores(
    model_path="path/to/model",
    train_df=train_data,
    val_df=val_data
)
```

## 🤝 Contributing

When adding new features:

1. **New Datasets**: Add configuration to `dataset_config.py`
2. **New Templates**: Extend template factories
3. **New Schemas**: Add Pydantic models to `schemas.py`
4. **New Strategies**: Implement in `framework.py`

## 📚 References

- **ProGen Paper**: "PROGEN: Progressive Zero-shot Dataset Generation via In-context Feedback" (EMNLP 2022)
- **Influence Functions**: Koh & Liang (2017) - "Understanding Black-box Predictions via Influence Functions"
- **LangChain**: Framework for LLM applications
- **HuggingFace**: Transformers library for NLP models 