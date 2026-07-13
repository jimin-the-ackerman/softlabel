import nltk
import numpy as np

import gc

from typing import List, Tuple
from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt', quiet=True)  # Ensure NLTK data is available


def vocabulary_size(texts: List[str]) -> int:
    """
    Computes the total number of unique tokens
    (vocabulary size) across all texts.
    
    :param texts: List of strings (the corpus).
    :return: integer - size of the overall vocabulary.
    """
    # Tokenize all at once
    all_tokens = set()
    for txt in texts:
        # Basic word-level tokenization
        tokens = nltk.word_tokenize(txt)
        all_tokens.update(tokens)
    return len(all_tokens)


def average_pairwise_similarity(embeddings: np.ndarray) -> float:
    """
    Computes the average pairwise cosine similarity among all embeddings using sklearn.
    :param embeddings: np.array of shape (N, d), one row per sample.
    :return: average similarity (float)
    """
    if embeddings.shape[0] < 2:
        return 0.0
    
    # This directly computes the similarity matrix
    sim_matrix = cosine_similarity(embeddings)
    
    # We only consider i<j pairs (upper triangle), to avoid double counting
    # and also to exclude the diagonal (similarity of an embedding with itself is 1)
    upper_tri_indices = np.triu_indices(sim_matrix.shape[0], k=1)
    sims = sim_matrix[upper_tri_indices]
    
    if sims.size == 0: # Handles the case where N=1 (after the initial check for N<2)
        return 0.0
        
    return np.mean(sims)


def average_pairwise_similarity_by_class(
        embeddings: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Computes average pairwise cosine similarity 'intra-class' and 'inter-class'
    using vectorized operations. Handles single-label (1D or 2D (N,1) labels)
    and multi-label (2D (N,C) binary/boolean indicators) cases.
    Includes attempts at explicit memory management.

    For multi-label scenarios (labels are 2D with shape (N,C) where C > 1),
    items are considered "intra-class" if they share at least one common class.
    Labels for multi-label are expected to be binary (0/1) or boolean.
    
    :param embeddings: 2D NumPy array of shape (N, d) representing item embeddings.
    :param labels: NumPy array representing labels for each embedding.
                     - For single-label: 1D array of shape (N,) or 2D array of shape (N,1).
                     - For multi-label: 2D array of shape (N, C) where C > 1.
    :return: Tuple (avg_inter_class_similarity, avg_intra_class_similarity).
             Returns (0.0, 0.0) if N < 2.
    """
    N = embeddings.shape[0]
    
    labels_np = np.asarray(labels)

    if embeddings.shape[0] != labels_np.shape[0]:
        raise ValueError(
            f"Number of embeddings ({embeddings.shape[0]}) must match the first "
            f"dimension of labels ({labels_np.shape[0]})."
        )

    if N < 2:
        # Call gc.collect() even here if this function is part of a loop
        # where previous iterations might have allocated memory.
        gc.collect()
        return (0.0, 0.0)

    # 1. Compute similarity matrix
    sim_matrix = cosine_similarity(embeddings) # Potentially very large: (N, N)

    # 2. Create same_class_indicator_matrix
    same_class_indicator_matrix = None # Initialize
    shared_class_counts = None # Initialize for multi-label case

    if labels_np.ndim == 1 or (labels_np.ndim == 2 and labels_np.shape[1] == 1):
        labels_1d = labels_np.ravel() if labels_np.ndim == 2 else labels_np
        same_class_indicator_matrix = (labels_1d[:, np.newaxis] == labels_1d[np.newaxis, :]) # (N,N)
    elif labels_np.ndim == 2 and labels_np.shape[1] > 1:
        labels_int = labels_np.astype(int)
        shared_class_counts = np.dot(labels_int, labels_int.T) # (N,N)
        same_class_indicator_matrix = (shared_class_counts > 0) # (N,N)
        del labels_int # Delete intermediate array
    else:
        # Clean up potentially large arrays before raising error
        del sim_matrix 
        if shared_class_counts is not None: del shared_class_counts
        gc.collect()
        raise ValueError(
            f"Labels array has an unsupported shape: {labels_np.shape}. "
            f"Must be 1D (N,), 2D (N,1), or 2D (N,C with C>1)."
        )

    # 3. Create triu_mask
    triu_mask = np.triu(np.ones((N, N), dtype=bool), k=1) # (N,N)

    # 4. Calculate intra-class similarities
    intra_class_mask = same_class_indicator_matrix & triu_mask
    intra_class_similarities = sim_matrix[intra_class_mask] # Can be large slice
    
    avg_intra_sim = 0.0
    sum_intra = 0.0
    count_intra = 0
    
    if intra_class_similarities.size > 0:
        sum_intra = np.sum(intra_class_similarities)
        count_intra = intra_class_similarities.size
        avg_intra_sim = sum_intra / count_intra
    
    # Delete intermediate arrays that are no longer needed for inter-class calculation
    del intra_class_similarities
    del intra_class_mask
    # same_class_indicator_matrix is not strictly needed anymore if we only use sum_intra and count_intra
    # but it's used implicitly by the deduction method. Let's keep it for now, or be more explicit.
    # For maximum memory saving, we could calculate inter_class_mask directly:
    # inter_class_mask = (~same_class_indicator_matrix) & triu_mask
    # inter_class_similarities = sim_matrix[inter_class_mask]
    # However, the deduction method is often faster if sum_intra is already computed.

    # 5. Calculate inter-class similarities
    all_upper_triangle_similarities = sim_matrix[triu_mask] # Can be large slice
    total_sum_upper_triangle = np.sum(all_upper_triangle_similarities)
    total_count_upper_triangle = all_upper_triangle_similarities.size

    del all_upper_triangle_similarities # Delete slice

    sum_inter = total_sum_upper_triangle - sum_intra
    count_inter = total_count_upper_triangle - count_intra
    
    avg_inter_sim = 0.0
    if count_inter > 0:
        avg_inter_sim = sum_inter / count_inter
        
    # Explicitly delete large intermediate arrays before returning
    del sim_matrix
    del same_class_indicator_matrix
    if shared_class_counts is not None: del shared_class_counts
    del triu_mask
    
    # Force garbage collection
    gc.collect()
        
    return avg_inter_sim, avg_intra_sim


def distinct_n(generated_texts: List[str], n: int = 2) -> float:
    """
    Computes Distinct-n for a list of generated texts:
    ratio of unique n-grams to total n-grams across the entire corpus.
    Higher is better for diversity.
    
    :param generated_texts: list of generated text strings
    :param n: n-gram size (e.g. 1, 2, or 3)
    :return: Distinct-n metric (float)
    """
    if not generated_texts:
        return 0.0
    
    total_ngrams = 0
    unique_ngrams = set()
    
    for text in generated_texts:
        tokens = nltk.word_tokenize(text)
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i+n])
            unique_ngrams.add(ngram)
            total_ngrams += 1
    
    if total_ngrams == 0:
        return 0.0
    
    return len(unique_ngrams) / total_ngrams


def inter_sample_ngram_freq(texts: List[str], n=2):
    """
    Counts how many times each n-gram occurs across the entire corpus
    and in how many distinct samples it appears.
    
    :param texts: list of strings
    :param n: n-gram size
    :return: a dictionary mapping n-grams to (total_count, sample_count)
    """
    # We'll store (ngram -> total_count_in_corpus, sample_count_in_corpus)
    ngram_stats = defaultdict(lambda: [0, 0])  # [total_count, sample_count]
    
    for txt in texts:
        tokens = nltk.word_tokenize(txt)
        seen_ngrams = set()  # track which n-grams were seen in this sample
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngram_stats[ngram][0] += 1
            seen_ngrams.add(ngram)
        # after processing all n-grams in this sample, increment sample_count
        for ng in seen_ngrams:
            ngram_stats[ng][1] += 1

    return dict(ngram_stats)


# Example usage:
if __name__ == "__main__":
    
    sample_texts = [
        "I love this movie so much !",
        "What a fantastic film . I love it .",
        "A truly wonderful experience with great visuals !"
    ]
    vocab_sz = vocabulary_size(sample_texts)
    print("Vocabulary Size:", vocab_sz)

    # Suppose we have embeddings for each sample
    # and numeric labels for each sample's class
    embeddings = np.array([
        [0.1, 0.2, 0.3],
        [0.12, 0.19, 0.29],
        [0.8, 0.1, 0.0],
        [0.79, 0.09, 0.02]
    ])
    labels = [0, 0, 1, 1]
    
    avg_sim_all = average_pairwise_similarity(embeddings)
    print("Avg unconditional similarity:", avg_sim_all)
    
    intra, inter = average_pairwise_similarity_by_class(embeddings, labels)
    print("Intra-class similarity:", intra)
    print("Inter-class similarity:", inter)


    sample_texts = [
        "I love this movie so much",
        "What a fantastic movie I saw",
        "I love watching this film"
    ]
    ngram_dict = inter_sample_ngram_freq(sample_texts, n=2)
    
    # Suppose we want to see n-grams that appear in at least 2 distinct samples
    common_ngrams = {ng: val for ng, val in ngram_dict.items() if val[1] >= 2}
    print("N-grams appearing in >=2 samples:\n", common_ngrams)