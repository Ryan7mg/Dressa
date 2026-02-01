"""
utils.py - Utility functions for Dressa App

Handles:
- Loading precomputed embeddings from .pkl files
- Cosine similarity search
- Combining and randomizing results from multiple models
"""

import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
import random
import logging

logger = logging.getLogger(__name__)

# Default paths
EMBEDDINGS_DIR = Path(__file__).parent / "embeddings"
IMAGES_DIR = Path(__file__).parent / "dress_images"

# Model name to filename mapping
EMBEDDING_FILES = {
    'openai_clip': 'openai_clip_embeddings.pkl',
    'fashion_clip': 'fashion_clip_embeddings.pkl',
    'marqo_fashion_clip': 'marqo_fashion_clip_embeddings.pkl',
    'marqo_fashion_siglip': 'marqo_fashion_siglip_embeddings.pkl'
}

# Cache for loaded embeddings
_embeddings_cache: Dict[str, dict] = {}


def load_embeddings(
    model_name: str,
    embeddings_dir: Optional[Path] = None
) -> dict:
    """
    Load precomputed embeddings from .pkl file.

    Args:
        model_name: One of 'openai_clip', 'fashion_clip',
                   'marqo_fashion_clip', 'marqo_fashion_siglip'
        embeddings_dir: Directory containing .pkl files (default: ./embeddings/)

    Returns:
        Dict with keys:
            - 'embeddings': numpy array of shape (N, 512)
            - 'image_paths': list of N image paths
    """
    # Return cached if available
    if model_name in _embeddings_cache:
        return _embeddings_cache[model_name]

    if embeddings_dir is None:
        embeddings_dir = EMBEDDINGS_DIR

    filename = EMBEDDING_FILES.get(model_name)
    if not filename:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Choose from: {list(EMBEDDING_FILES.keys())}")

    filepath = Path(embeddings_dir) / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Embedding file not found: {filepath}")

    logger.info(f"Loading embeddings from {filepath}")

    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    # Validate structure
    if 'embeddings' not in data or 'image_paths' not in data:
        raise ValueError(f"Invalid embedding file format. "
                        f"Expected keys: 'embeddings', 'image_paths'. "
                        f"Got: {data.keys()}")

    # Ensure embeddings are normalized
    embeddings = data['embeddings']
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    data['embeddings'] = embeddings / norms

    # Cache for future use
    _embeddings_cache[model_name] = data

    logger.info(f"Loaded {len(data['image_paths'])} embeddings for {model_name}")

    return data


def search_similar(
    query_embedding: np.ndarray,
    model_name: str,
    top_k: int = 10,
    embeddings_data: Optional[dict] = None
) -> List[Tuple[str, float]]:
    """
    Find top-K most similar images using cosine similarity.

    Args:
        query_embedding: Normalized query embedding of shape (512,)
        model_name: Which model's embeddings to search
        top_k: Number of results to return
        embeddings_data: Pre-loaded embeddings (loads if None)

    Returns:
        List of (image_path, similarity_score) tuples, sorted by similarity
    """
    # Load embeddings if not provided
    if embeddings_data is None:
        embeddings_data = load_embeddings(model_name)

    corpus_embeddings = embeddings_data['embeddings']
    image_paths = embeddings_data['image_paths']

    # Ensure query is 2D for sklearn
    query_2d = query_embedding.reshape(1, -1)

    # Compute cosine similarities
    similarities = cosine_similarity(query_2d, corpus_embeddings)[0]

    # Get top-K indices (sorted by similarity, descending)
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Build results
    results = []
    for idx in top_indices:
        results.append((image_paths[idx], float(similarities[idx])))

    return results


def search_all_models(
    query_embeddings: Dict[str, np.ndarray],
    top_k: int = 10
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Search across all models with their respective query embeddings.

    Args:
        query_embeddings: Dict mapping model_name -> query_embedding
        top_k: Number of results per model

    Returns:
        Dict mapping model_name -> list of (image_path, score) tuples
    """
    results = {}
    for model_name, query_emb in query_embeddings.items():
        results[model_name] = search_similar(query_emb, model_name, top_k)
    return results


def union_and_randomize(
    results_dict: Dict[str, List[Tuple[str, float]]],
    seed: Optional[int] = None
) -> List[Dict]:
    """
    Combine results from all models, remove duplicates, and shuffle.

    Args:
        results_dict: Dict mapping model_name -> list of (image_path, score)
        seed: Random seed for reproducibility (optional)

    Returns:
        List of dicts with keys:
            - 'image_path': path to the image
            - 'models': list of model names that returned this image
            - 'scores': dict mapping model_name -> similarity score
            - 'best_score': highest similarity score across models
    """
    # Track which models found each image
    image_info = {}  # image_path -> {'models': [...], 'scores': {...}}

    for model_name, results in results_dict.items():
        for image_path, score in results:
            if image_path not in image_info:
                image_info[image_path] = {
                    'image_path': image_path,
                    'models': [],
                    'scores': {}
                }
            image_info[image_path]['models'].append(model_name)
            image_info[image_path]['scores'][model_name] = score

    # Convert to list and compute best_score
    combined = list(image_info.values())
    for item in combined:
        item['best_score'] = max(item['scores'].values())

    # Shuffle randomly
    if seed is not None:
        random.seed(seed)
    random.shuffle(combined)

    return combined


def union_and_randomize_with_provenance(
    results_dict: Dict[str, List[Tuple[str, float]]],
    query_image_id: str
) -> List[Dict]:
    """
    Combine results from all models with provenance (rank per model) and deterministic shuffle.

    Args:
        results_dict: Dict mapping model_name -> list of (image_path, score)
        query_image_id: Unique identifier for the query image (used for deterministic shuffle)

    Returns:
        List of dicts with keys:
            - 'image_path': path to the image
            - 'provenance': dict mapping model_name -> rank (1-indexed)
            - 'display_position': position in the shuffled display order
    """
    image_info = {}

    for model_name, results in results_dict.items():
        for rank, (image_path, score) in enumerate(results, start=1):
            if image_path not in image_info:
                image_info[image_path] = {
                    'image_path': image_path,
                    'provenance': {}
                }
            image_info[image_path]['provenance'][model_name] = rank

    combined = list(image_info.values())

    # Deterministic shuffle based on query image ID
    seed = hash(query_image_id) & 0xFFFFFFFF
    random.Random(seed).shuffle(combined)

    for pos, item in enumerate(combined):
        item['display_position'] = pos

    return combined


def get_image_full_path(
    relative_path: str,
    images_dir: Optional[Path] = None
) -> Path:
    """
    Convert relative image path to full path.

    Args:
        relative_path: Relative path from embeddings file
        images_dir: Base directory for images

    Returns:
        Full path to the image
    """
    if images_dir is None:
        images_dir = IMAGES_DIR

    # Handle various path formats
    path = Path(relative_path)
    if path.is_absolute() and path.exists():
        return path

    # Try as relative to images_dir
    full_path = images_dir / path.name
    if full_path.exists():
        return full_path

    # Try the original path
    full_path = images_dir / relative_path
    if full_path.exists():
        return full_path

    # Return best guess
    return images_dir / path.name


def append_to_embeddings(
    image_path: str,
    embedding: np.ndarray,
    model_name: str,
    embeddings_dir: Optional[Path] = None
):
    """
    Append a new image's embedding to the corpus.
    Used for dynamic corpus growth after user ratings.

    Args:
        image_path: Path to the image being added
        embedding: Normalized embedding vector (512,)
        model_name: Which model's embedding file to update
        embeddings_dir: Directory containing .pkl files
    """
    if embeddings_dir is None:
        embeddings_dir = EMBEDDINGS_DIR

    filename = EMBEDDING_FILES.get(model_name)
    if not filename:
        raise ValueError(f"Unknown model: {model_name}")

    filepath = Path(embeddings_dir) / filename

    # Load current embeddings
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    # Ensure embedding is normalized
    embedding = embedding / np.linalg.norm(embedding)

    # Append new embedding
    data['embeddings'] = np.vstack([data['embeddings'], embedding.reshape(1, -1)])
    data['image_paths'].append(image_path)

    # Save back
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

    # Update cache if loaded
    if model_name in _embeddings_cache:
        _embeddings_cache[model_name] = data

    logger.info(f"Added {image_path} to {model_name} corpus. "
               f"New size: {len(data['image_paths'])}")


def clear_embeddings_cache():
    """Clear the embeddings cache (useful for testing)."""
    global _embeddings_cache
    _embeddings_cache = {}


# Convenience function for testing
def test_utils():
    """Test utility functions."""
    print("Testing utils.py...")

    # Test loading embeddings
    for model_name in EMBEDDING_FILES.keys():
        try:
            data = load_embeddings(model_name)
            print(f"  {model_name}: {data['embeddings'].shape[0]} images, "
                  f"dim={data['embeddings'].shape[1]}")
        except FileNotFoundError as e:
            print(f"  {model_name}: File not found (expected for first run)")

    print("\nUtils tests complete!")


if __name__ == "__main__":
    test_utils()
