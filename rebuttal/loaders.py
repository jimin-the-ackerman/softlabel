import os
import json

import numpy as np
import pandas as pd

from pathlib import Path
from typing import Tuple, Optional, Dict, Required, Union

from datasets import Dataset, DatasetDict, Value, concatenate_datasets


DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"

# path relative to root project directory
ORACLE_DATA_DIR = {
    'imdb': 'data/imdb/',
    'sst': 'data/sst/',
    'subj': 'data/subj/',
    'emotion': 'data/emotion/',
    'agnews': 'data/agnews/',
    'yahoo': 'data/yahoo/'
}


def load_oracle_embeddings(data: Required[str], model: Optional[str] = DEFAULT_EMBEDDING_MODEL):
    """
    Load oracle embeddings for a given dataset.

    Args:
        data: The name of the dataset. (e.g., 'imdb', 'sst', 'subj', 'emotion', 'agnews')
        model: The name of the embedding model. (e.g., 'openai/text-embedding-3-small')

    Returns:
        A dictionary containing the oracle embeddings for the given dataset.
    """
    
    directory = Path(ORACLE_DATA_DIR[data]).resolve()
    assert directory.exists(), f"Directory {directory} does not exist."

    if data == 'imdb':
        return IMDb.load_oracle_embeddings(directory, model)
    elif data == 'sst':
        return SST.load_oracle_embeddings(directory, model)
    elif data == 'subj':
        return SUBJ.load_oracle_embeddings(directory, model)
    elif data == 'emotion':
        return Emotion.load_oracle_embeddings(directory, model)
    elif data == 'agnews':
        return AGNews.load_oracle_embeddings(directory, model)
    elif data == 'yahoo':
        return Yahoo.load_oracle_embeddings(directory, model)
    else:
        raise ValueError


def load_synthetic_embeddings(data: Required[str], root: Required[str], model: Optional[str] = DEFAULT_EMBEDDING_MODEL):
    """
    Load synthetic embeddings for a given dataset.

    Args:
        data: The name of the dataset. (e.g., 'imdb', 'sst', 'subj', 'emotion', 'agnews')
        root: The root directory containing the synthetic data and embeddings.
        model: The name of the embedding model. (e.g., 'openai/text-embedding-3-small')

    Returns:
        A dictionary containing the synthetic embeddings for the given dataset.
    """
    
    if data == 'imdb':
        return IMDb.load_synthetic_embeddings(root, model)
    elif data == 'sst':
        return SST.load_synthetic_embeddings(root, model)
    elif data == 'subj':
        return SUBJ.load_synthetic_embeddings(root, model)
    elif data == 'emotion':
        return Emotion.load_synthetic_embeddings(root, model)
    elif data == 'agnews':
        return AGNews.load_synthetic_embeddings(root, model)
    elif data == 'yahoo':
        return Yahoo.load_synthetic_embeddings(root, model)
    else:
        raise ValueError(f"Unknown dataset: {data}")



class _DatasetBase:
    def __init__(self):
        pass


class IMDb(_DatasetBase):
    _label_mapper = {'negative': 0, 'positive': 1}
    
    @classmethod
    def load_oracle_dataset(cls, root: Union[str, Path]) -> DatasetDict:
        root = Path(root).resolve()
        train_df, test_df = pd.read_csv(root / "train.csv"), pd.read_csv(root / "test.csv")
        train_dataset = Dataset.from_dict({
            'text': train_df['review'].tolist(),
            'labels': train_df['sentiment'].map(cls._label_mapper).tolist()
        })
        test_dataset = Dataset.from_dict({
            'text': test_df['review'].tolist(),
            'labels': test_df['sentiment'].map(cls._label_mapper).tolist()
        })
        return DatasetDict(
            {
                'train': train_dataset,
                'test': test_dataset
            }
        )
        
    @staticmethod
    def load_oracle_embeddings(root: Required[str],
                               model: Optional[str] = DEFAULT_EMBEDDING_MODEL
                               ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        
        directory = Path(root).resolve() / "embeddings" / model
        assert directory.exists(), f"Directory {directory} does not exist."
        
        X_train = np.load(directory / "train.features.npy")  # (N, D)
        y_train = np.load(directory / "train.labels.npy")
        X_test = np.load(directory / "test.features.npy")
        y_test = np.load(directory / "test.labels.npy")
        
        return {
            'train': (X_train, y_train),
            'test': (X_test, y_test),
        }
    
    @staticmethod
    def load_synthetic_dataset(root: Union[str, Path]):

        root = Path(root).resolve()
        assert root.exists(), f"Directory {root} does not exist."

        # Load data
        with open(root / "data.jsonl", "r") as f:
            data = [json.loads(line) for line in f]
        
        # Extract soft labels (original probabilities)
        soft_labels = [d['label'] for d in data]
        
        # Convert to 2D format for binary classification: [P(class=0), P(class=1)]
        soft_labels_2d = [[1-l, l] for l in soft_labels]
        
        # Convert to binary hard labels for training
        hard_labels = [round(l) for l in soft_labels]
        
        return Dataset.from_dict({
                'text': [d['text'] for d in data],
                'labels': hard_labels,      # Hard labels for training
                'soft_labels': soft_labels_2d  # Soft labels for filtering (2D format)
            })

    @staticmethod
    def load_synthetic_embeddings(root: Required[str],  # e.g., results/imdb/gemini-2.0-flash/soft/
                                  model: Optional[str] = DEFAULT_EMBEDDING_MODEL,
                                  text: bool = True
                                 ) -> Dict[str, np.ndarray]:

        directory = Path(root).resolve()
        assert directory.exists(), f"Directory {directory} does not exist."

        # Load data (label and text)
        with open(directory / "data.jsonl", "r") as f:
            data = [json.loads(line) for line in f]
        
        # Text (x)
        texts = [d['text'] for d in data]

        # Labels (y)
        labels = [d['label'] for d in data]  # e.g., [0.3, 0.7, ..., 0.1]
        
        # Convert to 2D format for binary classification: [P(class=0), P(class=1)]
        labels_2d = [[1-l, l] for l in labels]
        labels = np.array(labels_2d)  # shape: [n_samples, 2]
        assert len(labels) == len(texts)

        # Embeddings (z)
        embeddings = np.load(directory / f"embeddings/{model}/data.npy")
        assert len(labels) == embeddings.shape[0]

        result = {
            'labels': labels,
            'embeddings': embeddings,
        }
        
        if text:
            result['text'] = texts
            
        return result


class SST(_DatasetBase):

    @classmethod
    def load_oracle_dataset(cls, binary: bool = True) -> DatasetDict:
        from datasets import load_dataset
        ds = load_dataset("stanfordnlp/sst", trust_remote_code=True)
        ds = ds.remove_columns(['tokens', 'tree'])
        ds = ds.rename_column('sentence', 'text')
        ds = ds.rename_column('label', 'labels')

        if binary:
            ds = ds.filter(lambda example: example['labels'] != -1)
            ds = ds.map(lambda example: {'labels': int(example['labels'] > 0.5)})
            # Ensure labels are integers, not floats
            ds = ds.cast_column('labels', Value('int64'))

        return ds
        
    @staticmethod
    def load_oracle_embeddings(root: Required[str],
                               model: Optional[str] = DEFAULT_EMBEDDING_MODEL
                               ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        
        directory = Path(root).resolve() / "embeddings" / model
        assert directory.exists(), f"Directory {directory} does not exist."
        
        X_train = np.load(directory / "train.features.npy")
        y_train = np.load(directory / "train.labels.npy")
        X_valid = np.load(directory / "validation.features.npy")
        y_valid = np.load(directory / "validation.labels.npy")
        X_test = np.load(directory / "test.features.npy")
        y_test = np.load(directory / "test.labels.npy")
        
        return {
            'train': (X_train, y_train),
            'validation': (X_valid, y_valid),
            'test': (X_test, y_test),
        }
    
    @staticmethod
    def load_synthetic_dataset(root: Union[str, Path]):
        root = Path(root).resolve()
        assert root.exists(), f"Directory {root} does not exist."

        # Load data
        with open(root / "data.jsonl", "r") as f:
            data = [json.loads(line) for line in f]
        
        # Extract soft labels (original probabilities)
        soft_labels = [d['label'] for d in data]
        
        # Convert to 2D format for binary classification: [P(class=0), P(class=1)]
        soft_labels_2d = [[1-l, l] for l in soft_labels]
        
        # Convert to binary hard labels for training
        hard_labels = [round(l) for l in soft_labels]
        
        return Dataset.from_dict({
                'text': [d['text'] for d in data],
                'labels': hard_labels,      # Hard labels for training
                'soft_labels': soft_labels_2d  # Soft labels for filtering (2D format)
            })
    
    @staticmethod
    def load_synthetic_embeddings(root: Required[str],
                            model: Optional[str] = DEFAULT_EMBEDDING_MODEL,
                            text: bool = True
                            ) -> Dict[str, np.ndarray]:

        directory = Path(root).resolve()
        assert directory.exists(), f"Directory {directory} does not exist."

        # Load data (label and text)
        with open(directory / "data.jsonl", "r") as f:
            data = [json.loads(line) for line in f]
        
        # Text (x)
        texts = [d['text'] for d in data]

        # Labels (y)
        labels = [d['label'] for d in data]
        # Convert to 2D format for binary classification: [P(class=0), P(class=1)]
        labels_2d = [[1-l, l] for l in labels]
        labels = np.array(labels_2d)  # shape: [n_samples, 2]
        assert len(labels) == len(texts)

        # Embeddings (z)
        embeddings = np.load(directory / f"embeddings/{model}/data.npy")
        assert len(labels) == embeddings.shape[0]

        result = {
            'labels': labels,
            'embeddings': embeddings,
        }
        
        if text:
            result['text'] = texts
            
        return result


class SUBJ(_DatasetBase):

    @classmethod
    def load_oracle_dataset(cls) -> DatasetDict:
        from datasets import load_dataset
        ds = load_dataset("SetFit/subj", trust_remote_code=True)
        ds = ds.remove_columns('label_text')
        ds = ds.rename_column('label', 'labels')
        return ds
        
    @staticmethod
    def load_oracle_embeddings(root: Required[str],
                               model: Optional[str] = DEFAULT_EMBEDDING_MODEL
                               ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        
        directory = Path(root).resolve() / "embeddings" / model
        assert directory.exists(), f"Directory {directory} does not exist."
        
        X_train = np.load(directory / "train.features.npy")
        y_train = np.load(directory / "train.labels.npy")
        X_test = np.load(directory / "test.features.npy")
        y_test = np.load(directory / "test.labels.npy")
        
        return {
            'train': (X_train, y_train),
            'test': (X_test, y_test),
        }
    
    @staticmethod
    def load_synthetic_dataset(root: Union[str, Path]):
        root = Path(root).resolve()
        assert root.exists(), f"Directory {root} does not exist."

        # Load data
        with open(root / "data.jsonl", "r") as f:
            data = [json.loads(line) for line in f]
        
        # Extract soft labels (original probabilities)
        soft_labels = [d['label'] for d in data]
        
        # Convert to 2D format for binary classification: [P(class=0), P(class=1)]
        soft_labels_2d = [[1-l, l] for l in soft_labels]
        
        # Convert to binary hard labels for training
        hard_labels = [round(l) for l in soft_labels]
        
        return Dataset.from_dict({
                'text': [d['text'] for d in data],
                'labels': hard_labels,      # Hard labels for training
                'soft_labels': soft_labels_2d  # Soft labels for filtering (2D format)
            })
    
    @staticmethod
    def load_synthetic_embeddings(root: Required[str],
                            model: Optional[str] = DEFAULT_EMBEDDING_MODEL,
                            text: bool = True
                            ) -> Dict[str, np.ndarray]:

        directory = Path(root).resolve()
        assert directory.exists(), f"Directory {directory} does not exist."

        # Load data (label and text)
        with open(directory / "data.jsonl", "r") as f:
            data = [json.loads(line) for line in f]
        
        # Text (x)
        texts = [d['text'] for d in data]

        # Labels (y)
        labels = [d['label'] for d in data]
        # Convert to 2D format for binary classification: [P(class=0), P(class=1)]
        labels_2d = [[1-l, l] for l in labels]
        labels = np.array(labels_2d)  # shape: [n_samples, 2]
        assert len(labels) == len(texts)

        # Embeddings (z)
        embeddings = np.load(directory / f"embeddings/{model}/data.npy")
        assert len(labels) == embeddings.shape[0]

        result = {
            'labels': labels,
            'embeddings': embeddings,
        }
        
        if text:
            result['text'] = texts
            
        return result


class Emotion(_DatasetBase):
    @classmethod
    def load_oracle_dataset(cls, undersample: bool = False) -> DatasetDict:
        from datasets import load_dataset
        
        # Load split data (used for testing)
        ds_split = load_dataset("dair-ai/emotion", "split", trust_remote_code=True)
        ds_split = ds_split.rename_column('label', 'labels')
        test_data = concatenate_datasets([ds_split['train'], ds_split['validation'], ds_split['test']])
        
        # Load unsplit data (used for training, will be undersampled later)
        ds_unsplit = load_dataset("dair-ai/emotion", "unsplit", trust_remote_code=True)
        ds_unsplit = ds_unsplit.rename_column('label', 'labels')
        
        # Undersample if requested
        if undersample:
            # Get the unsplit dataset
            ds_train = ds_unsplit['train']
            
            # Find minority class count
            label_counts = ds_train['labels']
            unique_labels, counts = np.unique(label_counts, return_counts=True)
            minority_class_count = counts.min()
            
            # Undersample each class to minority class count
            undersampled_datasets = []
            for label in unique_labels:
                # Filter dataset for current class
                class_dataset = ds_train.filter(lambda x: x['labels'] == label)
                
                if len(class_dataset) > minority_class_count:
                    # Randomly sample minority_class_count examples
                    class_dataset = class_dataset.shuffle(seed=42).select(range(minority_class_count))
                
                undersampled_datasets.append(class_dataset)
            
            # Concatenate all undersampled datasets
            ds_unsplit_balanced = concatenate_datasets(undersampled_datasets)
            train_data = ds_unsplit_balanced
            
            return DatasetDict({
                'train': train_data,  # undersampled unsplit data for training
                'test': test_data     # all split data concatenated for testing
            })
        else:
            return DatasetDict({
                'train': ds_unsplit['train'],  # unsplit data for training
                'test': test_data              # all split data concatenated for testing
            })
    
    @staticmethod
    def _filter_valid_emotion_labels(labels, texts):
        """Filter out invalid emotion labels (those that don't have length 6)."""
        valid_mask = np.array([len(label) == 6 for label in labels])
        valid_idx = np.where(valid_mask)[0]
        filtered_labels = [l for i, l in enumerate(labels) if i in valid_idx]
        filtered_texts = [texts[i] for i in valid_idx]
        return filtered_labels, filtered_texts, valid_idx
    
    @staticmethod
    def load_oracle_embeddings(root: Required[str],
                               model: Optional[str] = DEFAULT_EMBEDDING_MODEL
                               ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        
        directory = Path(root).resolve() / "embeddings" / model
        assert directory.exists(), f"Directory {directory} does not exist."

        # 1. Split (used as testing data)
        X_train = np.load(directory / "train.features.npy")
        y_train = np.load(directory / "train.labels.npy")
        X_valid = np.load(directory / "validation.features.npy")
        y_valid = np.load(directory / "validation.labels.npy")
        X_test = np.load(directory / "test.features.npy")
        y_test = np.load(directory / "test.labels.npy")

        # 1.1. Concatenate
        X_split = np.concatenate((X_train, X_valid, X_test), axis=0)
        y_split = np.concatenate((y_train, y_valid, y_test), axis=0)
        
        # 2. Unsplit (used as training data)
        X_unsplit = np.load(directory / "unsplit.features.npy")
        y_unsplit = np.load(directory / "unsplit.labels.npy")

        return {
            'train': (X_unsplit, y_unsplit),
            'test': (X_split, y_split),
        }

    @staticmethod
    def load_synthetic_dataset(root: Union[str, Path]):
        root = Path(root).resolve()
        assert root.exists(), f"Directory {root} does not exist."

        # Load data
        with open(root / "data.jsonl", "r") as f:
            data = [json.loads(line) for line in f]
        
        texts = [d['text'] for d in data]
        labels = [d['label'] for d in data]
        
        # Filter valid emotion labels
        filtered_labels, filtered_texts, _ = Emotion._filter_valid_emotion_labels(labels, texts)
        
        # Convert to hard labels (argmax of soft labels)
        hard_labels = [np.argmax(label) for label in filtered_labels]
        
        # Normalize soft labels for consistency
        normalized_soft_labels = []
        for label in filtered_labels:
            normalized_label = np.array(label) / np.sum(label)
            normalized_soft_labels.append(normalized_label.tolist())
        
        return Dataset.from_dict({
                'text': filtered_texts,
                'labels': hard_labels,                 # Hard labels for training
                'soft_labels': normalized_soft_labels  # Soft labels for filtering
            })

    @staticmethod
    def load_synthetic_embeddings(root: Required[str],
                                  model: Optional[str] = DEFAULT_EMBEDDING_MODEL,
                                  text: bool = True
                                  ) -> Dict[str, np.ndarray]:

        directory = Path(root).resolve()
        assert directory.exists(), f"Directory {directory} does not exist."

        # Load data (label and text)
        with open(directory / "data.jsonl", "r") as f:
            data = [json.loads(line) for line in f]

        # Text (x)
        texts = [d['text'] for d in data]

        # Labels (y)
        labels = [d['label'] for d in data]
        
        # Filter valid emotion labels
        filtered_labels, filtered_texts, valid_idx = Emotion._filter_valid_emotion_labels(labels, texts)
        
        # Process labels
        labels = np.array(filtered_labels)                            # soft vectors
        labels = labels / labels.sum(axis=1, keepdims=True)  # normalized
        assert labels.sum(axis=1).all(), "Rows must sum up to 1."

        # Embeddings (z)
        embeddings = np.load(directory / f"embeddings/{model}/data.npy")
        embeddings = embeddings[valid_idx]
        assert labels.shape[0] == embeddings.shape[0]

        result = {
            'labels': labels,
            'embeddings': embeddings,
        }
        
        if text:
            result['text'] = filtered_texts
            
        return result


class Yahoo(_DatasetBase):
    huggingface_id: str = "yassiracharki/Yahoo_Answers_10_categories_for_NLP"
    @classmethod
    def load_oracle_dataset(cls) -> DatasetDict:
        from datasets import load_dataset
        
        # Load from Huggingface
        ds = load_dataset(cls.huggingface_id, trust_remote_code=True)
        train_df = ds['train'].to_pandas()
        test_df = ds['test'].to_pandas()

        # Work on dataframes
        train_df['question'] = \
            train_df['question_title'].fillna('') + ' ' + \
                train_df['question_content'].fillna('')
        
        test_df['question'] = \
            test_df['question_title'].fillna('') + ' ' + \
                test_df['question_content'].fillna('')

        # Load 'subsample_idx.txt'
        filepath = Path(ORACLE_DATA_DIR['yahoo']).resolve() / "subsample_idx.txt"
        subsample_idx = np.loadtxt(filepath, dtype=int)
             
        # Filter training data (using subsample_idx)
        # No filtering applied to testing data
        train_df = train_df.loc[subsample_idx, :].copy()

        train_dataset = Dataset.from_dict({
            'text': train_df['question'].tolist(),
            'labels': (train_df['class_index'] - 1).tolist()
        })

        test_dataset = Dataset.from_dict({
            'text': test_df['question'].tolist(),
            'labels': (test_df['class_index'] - 1).tolist()
        })

        return DatasetDict(
            {
                'train': train_dataset,
                'test': test_dataset,
            }
        )

    @staticmethod
    def load_oracle_embeddings(root: Required[str],
                               model: Optional[str] = DEFAULT_EMBEDDING_MODEL):

        directory = Path(root).resolve() / "embeddings" / model
        assert directory.exists(), f"Directory {directory} does not exist."

        X_train = np.load(directory / "train.features.npy")
        y_train = np.load(directory / "train.labels.npy")
        X_test = np.load(directory / "test.features.npy")
        y_test = np.load(directory / "test.labels.npy")

        return {
            'train': (X_train, y_train),
            'test': (X_test, y_test),
        }

    @staticmethod
    def load_synthetic_dataset(root: Union[str, Path]):
        root = Path(root).resolve()
        assert root.exists(), f"Directory {root} does not exist."

        # Load data
        with open(root / "data.jsonl", "r") as f:
            data = [json.loads(line) for line in f]
        
        # Texts
        def get_text():
            titles = [d['title'] for d in data]
            contents = [d['content'] for d in data]
            return [f"{t} {c}" for t, c in zip(titles, contents)]

        texts = get_text()
        labels = [d['label'] for d in data]  # soft

        filtered_labels, filtered_texts, _ = \
            Yahoo._filter_valid_yahoo_labels_and_text(labels, texts)
        
        # Convert to hard labels (argmax of soft labels)
        hard_labels = [np.argmax(label) for label in filtered_labels]

        # Normalize soft labels for consistency
        normalized_soft_labels = []
        for label in filtered_labels:
            normalized_label = np.array(label) / np.sum(label)
            normalized_soft_labels.append(normalized_label.tolist())

        return Dataset.from_dict({
            'text': filtered_texts,
            'labels': hard_labels,                 # for cross-entropy training
            'soft_labels': normalized_soft_labels  # for filtering or KL-div training
        })

    @staticmethod
    def load_synthetic_embeddings(root: Required[str],
                                  model: Optional[str] = DEFAULT_EMBEDDING_MODEL,
                                  text: str = True
                                  ) -> Dict[str, np.ndarray]:
        
        directory = Path(root).resolve()
        assert directory.exists(), f"Directory {directory} does not exist."

        # Load data (label and text)
        with open(directory / "data.jsonl", "r") as f:
            data = [json.loads(line) for line in f]
        
        titles = [d['title'] for d in data]
        contents = [d['content'] for d in data]
        texts = [f"{t} {c}" for t, c in zip(titles, contents)]
        labels = [d['label'] for d in data]

        # Filter valid rows
        filtered_labels, filtered_texts, valid_idx = \
            Yahoo._filter_valid_yahoo_labels_and_text(labels, texts)
        
        # Process labels
        labels = np.array(filtered_labels)
        labels = labels / labels.sum(axis=1, keepdims=True)
        assert labels.sum(axis=1).all(), "Rows must sum up to 1."

        # Embeddings
        embeddings = np.load(directory / f"embeddings/{model}/data.npy")
        embeddings = embeddings[valid_idx]
        assert labels.shape[0] == embeddings.shape[0]

        result = {
            'labels': labels,
            'embeddings': embeddings,
        }

        if text:
            result['text'] = filtered_texts
        
        return result


    @staticmethod
    def _filter_valid_yahoo_labels_and_text(labels, texts):
        """
        Filter out invalid labels (those that do not have a length of 10).
        Also filter out those with empty text.
        """
        label_valid_mask = np.array([len(label) == 10 for label in labels])
        text_valid_mask = np.array([text.strip() != "" for text in texts])
        valid_mask = label_valid_mask * text_valid_mask
        valid_idx = np.where(valid_mask)[0]
        filtered_labels = [labels[i] for i in valid_idx]
        filtered_texts = [texts[i] for i in valid_idx]
        return filtered_labels, filtered_texts, valid_idx


class AGNews(_DatasetBase):
    @classmethod
    def load_oracle_dataset(cls, undersample: bool = False) -> DatasetDict:
        from datasets import load_dataset
        ds = load_dataset("SetFit/ag_news", trust_remote_code=True)
        ds = ds.remove_columns('label_text')
        ds = ds.rename_column('label', 'labels')
        ds = ds.map(  # optional cleaning; not mandatory
            lambda example: {'text': AGNews.clean_text(example['text'])} 
            )
        return ds

    @staticmethod
    def load_oracle_embeddings(root: Required[str],
        model: Optional[str] = DEFAULT_EMBEDDING_MODEL) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        
        directory = Path(root).resolve() / "embeddings" / model
        assert directory.exists(), f"Directory {directory} does not exist."

        X_train = np.load(directory / "train.features.npy")
        y_train = np.load(directory / "train.labels.npy")
        X_test = np.load(directory / "test.features.npy")
        y_test = np.load(directory / "test.labels.npy")
        
        return {
            'train': (X_train, y_train),
            'test': (X_test, y_test),
        }

    @staticmethod
    def _filter_valid_agnews_labels_and_text(labels, texts):
        """
        Filter out invalid labels (those that do not have a length of 4).
        Also filter out those with empty text.
        """
        label_valid_mask = np.array([len(label) == 4 for label in labels])
        text_valid_mask = np.array([text.strip() != "" for text in texts])
        valid_mask = label_valid_mask * text_valid_mask
        valid_idx = np.where(valid_mask)[0]
        filtered_labels = [labels[i] for i in valid_idx]
        filtered_texts = [texts[i] for i in valid_idx]
        return filtered_labels, filtered_texts, valid_idx

    @staticmethod
    def load_synthetic_dataset(root: Union[str, Path]):
        root = Path(root).resolve()
        assert root.exists(), f"Directory {root} does not exist."
        
        # Load data
        with open(root / "data.jsonl", "r") as f:
            data = [json.loads(line) for line in f]
        
        texts = [d['text'] for d in data]
        labels = [d['label'] for d in data]

        # Filter valid rows
        filtered_labels, filtered_texts, _ = \
            AGNews._filter_valid_agnews_labels_and_text(labels, texts)
        
        # Convert to hard labels (argmax of soft labels)
        hard_labels = [np.argmax(label) for label in filtered_labels]

        # Normalize soft labels for consistency
        normalized_soft_labels = []
        for label in filtered_labels:
            normalized_label = np.array(label) / np.sum(label)
            normalized_soft_labels.append(normalized_label.tolist())
        
        return Dataset.from_dict({
                'text': filtered_texts,
                'labels': hard_labels,
                'soft_labels': normalized_soft_labels
            })

    @staticmethod
    def load_synthetic_embeddings(root: Required[str],
                                model: Optional[str] = DEFAULT_EMBEDDING_MODEL,
                                text: bool = True
                                ) -> Dict[str, np.ndarray]:
        
        directory = Path(root).resolve()
        assert directory.exists(), f"Directory {directory} does not exist."

        # Load data (label and text)
        with open(directory / "data.jsonl", "r") as f:
            data = [json.loads(line) for line in f]

        texts = [d['text'] for d in data]
        labels = [d['label'] for d in data]

        # Filter valid rows
        filtered_labels, filtered_texts, valid_idx = \
            AGNews._filter_valid_agnews_labels_and_text(labels, texts)

        # Process labels
        labels = np.array(filtered_labels)
        labels = labels / labels.sum(axis=1, keepdims=True)
        assert labels.sum(axis=1).all(), "Rows must sum up to 1."

        # Embeddings
        embeddings = np.load(directory, f"embeddings/{model}/data.npy")
        embeddings = embeddings[valid_idx]
        assert labels.shape[0] == embeddings.shape[0]

        result = {
            'labels': labels,
            'embeddings': embeddings,
        }

        if text:
            result['text'] = filtered_texts
        
        return result

    @staticmethod
    def clean_text(text: str) -> str:
        import re
        text = text.lower()
        text.replace("\n", " ")
        text = re.sub(r'""', '"', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub('\s+', ' ', text).strip()
        return text