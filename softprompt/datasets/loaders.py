import os
import json

import numpy as np

from pathlib import Path
from typing import Tuple, Union


DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"

# path relative to root project directory
DATA_TO_EMBEDDING_DIR = {
    'imdb': 'data/imdb/embeddings/',
    'sst': 'data/sst/embeddings/.',
    'subj': 'data/subj/embeddings/',
    'emotion': 'data/emotion/embeddings/',
    'agnews': 'data/agnews/embeddings/',
}


def load_oracle_data(data: str,
                     model: str = DEFAULT_EMBEDDING_MODEL):

    embedding_dir = Path(DATA_TO_EMBEDDING_DIR[data]).resolve()
    embedding_dir = embedding_dir / model

    if data == 'imdb':
        return IMDb.load_oracle_imdb(embedding_dir)
    elif data == 'sst':
        raise NotImplementedError
    elif data == 'subj':
        return SUBJ.load_oracle_subj(embedding_dir)
    elif data == 'emotion':
        return Emotion.load_oracle_emotion(embedding_dir)
    elif data == 'agnews':
        raise NotImplementedError
    else:
        raise ValueError


def load_synthetic_data(data: str,
                        directory: str,
                        model: str = DEFAULT_EMBEDDING_MODEL):

    embedding_dir = Path(directory).resolve()

    if data == 'imdb':
        return IMDb.load_synthetic_imdb(embedding_dir, model)
    elif data == 'sst':
        raise NotImplementedError
    elif data == 'subj':
        return SUBJ.load_synthetic_subj(embedding_dir, model)
    elif data == 'emotion':
        return Emotion.load_synthetic_emotion(embedding_dir, model)
    elif data == 'agnews':
        raise NotImplementedError
    else:
        raise ValueError


class Emotion:
    
    @staticmethod
    def load_oracle_emotion(directory: str) -> Tuple[np.ndarray]:

        _directory = Path(directory).resolve()

        X_train = np.load(_directory / "train.features.npy")
        y_train = np.load(_directory / "train.labels.npy")
        X_valid = np.load(_directory / "validation.features.npy")
        y_valid = np.load(_directory / "validation.labels.npy")
        X_test = np.load(_directory / "test.features.npy")
        y_test = np.load(_directory / "test.labels.npy")

        X_split = np.concatenate((X_train, X_valid, X_test), axis=0)
        y_split = np.concatenate((y_train, y_valid, y_test), axis=0)
        
        X_unsplit = np.load(_directory / "unsplit.features.npy")
        y_unsplit = np.load(_directory / "unsplit.labels.npy")
        
        return X_unsplit, y_unsplit, X_split, y_split
    
    @staticmethod
    def load_synthetic_emotion(directory: str,
                               model: str = DEFAULT_EMBEDDING_MODEL) -> Tuple[np.ndarray]:

        _directory = Path(directory).resolve()

        # load labels and text (but only labels are required)
        with open(_directory / "data.jsonl", "r") as f:
            data = [json.loads(line) for line in f]

        # labels (filter invalid labels)
        labels = [d['label'] for d in data]
        mask = np.array([len(l) == 6 for l in labels])
        valid_idx = np.where(mask)[0]
        labels = [l for i, l in enumerate(labels) if i in valid_idx]
        labels = np.array(labels)             # soft vectors 
        soft_labels = labels / labels.sum(axis=1, keepdims=True)  # normalized
        assert soft_labels.sum(axis=1).all()  # check row sum = 1

        # embeddings 
        embeddings = np.load(_directory / f"embeddings/{model}/data.npy")
        embeddings = embeddings[valid_idx]
        assert len(soft_labels) == embeddings.shape[0]

        return embeddings, soft_labels  # X, y


class SUBJ:

    @staticmethod
    def load_oracle_subj(directory: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        
        _directory = Path(directory).resolve()
        
        X_train = np.load(_directory / "train.features.npy")
        y_train = np.load(_directory / "train.labels.npy")
        X_test = np.load(_directory / "test.features.npy")
        y_test = np.load(_directory / "test.labels.npy")
        
        return X_train, y_train, X_test, y_test
    
    @staticmethod
    def load_synthetic_subj(directory: str,
                            model: str = DEFAULT_EMBEDDING_MODEL):

        _directory = Path(directory).resolve()

        # load labels and text (but only labels are required)
        with open(_directory / "data.jsonl", "r") as f:
            data = [json.loads(line) for line in f]
        
        # parse labels
        labels = [d['label'] for d in data]
        labels = np.array(labels)

        # p_i -> [1-p_i, p_i]; for consistency with multiclass
        soft_labels = np.stack([1-labels, labels], axis=1)
        assert soft_labels.sum(axis=1).all()

        # embeddings 
        embeddings = np.load(_directory / f"embeddings/{model}/data.npy")
        assert len(soft_labels) == embeddings.shape[0]

        return embeddings, soft_labels


class IMDb:

    @staticmethod
    def load_oracle_imdb(directory: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        
        _directory = Path(directory).resolve()
        
        X_train = np.load(_directory / "train.features.npy")
        y_train = np.load(_directory / "train.labels.npy")
        X_test = np.load(_directory / "test.features.npy")
        y_test = np.load(_directory / "test.labels.npy")
        
        return X_train, y_train, X_test, y_test
    
    @staticmethod
    def load_synthetic_imdb(directory: str,
                            model: str = DEFAULT_EMBEDDING_MODEL):

        _directory = Path(directory).resolve()

        # load labels and text (but only labels are required)
        with open(_directory / "data.jsonl", "r") as f:
            data = [json.loads(line) for line in f]
        
        # parse labels
        labels = [d['label'] for d in data]
        labels = np.array(labels)

        # p_i -> [1-p_i, p_i]; for consistency with multiclass
        soft_labels = np.stack([1-labels, labels], axis=1)
        assert soft_labels.sum(axis=1).all()

        # embeddings 
        embeddings = np.load(_directory / f"embeddings/{model}/data.npy")
        assert len(soft_labels) == embeddings.shape[0]

        return embeddings, soft_labels
