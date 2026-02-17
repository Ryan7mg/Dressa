"""
utils.py - Utility functions for Dressa App

Handles:
- Loading precomputed embeddings from .pkl or text files
- Cosine similarity search
- Combining and randomizing results from multiple models
"""

import pickle
import numpy as np
from pathlib import Path
import os
from typing import List, Tuple, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
import random
import logging

logger = logging.getLogger(__name__)

# Default paths - use resolve() for consistent absolute paths
APP_DIR = Path(__file__).parent.resolve()
EMBEDDINGS_DIR = Path(
    os.getenv("DRESSA_EMBEDDINGS_DIR", str(APP_DIR / "embeddings"))
).resolve()
IMAGES_DIR = Path(
    os.getenv("DRESSA_IMAGES_DIR", str(APP_DIR / "dress_images"))
).resolve()

# Model name to filename mapping
EMBEDDING_FILES = {
    'openai_clip': 'openai_clip_embeddings.pkl',
    'fashion_clip': 'fashion_clip_embeddings.pkl',
    'marqo_fashion_clip': 'marqo_fashion_clip_embeddings.pkl',
    'marqo_fashion_siglip': 'marqo_fashion_siglip_embeddings.pkl'
}

# Cache for loaded embeddings
_embeddings_cache: Dict[str, dict] = {}


def _text_embedding_paths(model_name: str, embeddings_dir: Path) -> Tuple[Path, Path]:
    """Return (embeddings_csv_path, image_paths_txt_path) for text embedding storage."""
    return (
        Path(embeddings_dir) / f"{model_name}_embeddings.csv",
        Path(embeddings_dir) / f"{model_name}_image_paths.txt",
    )


def load_embeddings(
    model_name: str,
    embeddings_dir: Optional[Path] = None
) -> dict:
    """
    Load precomputed embeddings from .pkl or text files.

    Args:
        model_name: One of 'openai_clip', 'fashion_clip',
                   'marqo_fashion_clip', 'marqo_fashion_siglip'
        embeddings_dir: Directory containing embedding files (default: ./embeddings/)

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
    if filepath.exists():
        logger.info(f"Loading embeddings from {filepath}")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    else:
        embeddings_csv, image_paths_txt = _text_embedding_paths(model_name, Path(embeddings_dir))
        if not embeddings_csv.exists() or not image_paths_txt.exists():
            raise FileNotFoundError(
                f"Embedding files not found. Checked: {filepath}, "
                f"{embeddings_csv}, {image_paths_txt}"
            )

        logger.info(f"Loading embeddings from text files: {embeddings_csv}, {image_paths_txt}")
        embeddings = np.loadtxt(embeddings_csv, delimiter=",", dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        image_paths = [
            line.strip()
            for line in image_paths_txt.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        data = {
            "embeddings": embeddings,
            "image_paths": image_paths,
        }

    # Validate structure
    if 'embeddings' not in data or 'image_paths' not in data:
        raise ValueError(f"Invalid embedding file format. "
                        f"Expected keys: 'embeddings', 'image_paths'. "
                        f"Got: {data.keys()}")

    if len(data['image_paths']) != len(data['embeddings']):
        raise ValueError(
            f"Embedding row count mismatch for {model_name}: "
            f"{len(data['embeddings'])} embeddings vs {len(data['image_paths'])} image paths"
        )

    # Ensure embeddings are normalized
    embeddings = np.asarray(data['embeddings'], dtype=np.float32)
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
    query_image_hash: str
) -> List[Dict]:
    """
    Combine results from all models with provenance (rank per model) and deterministic shuffle.

    Args:
        results_dict: Dict mapping model_name -> list of (image_path, score)
        query_image_hash: Hash of query image content (used for deterministic shuffle)

    Returns:
        List of dicts with keys:
            - 'image_path': path to the image
            - 'provenance': dict mapping model_name -> rank (1-indexed)
            - 'display_position': position in the shuffled display order
    """
    image_info = {}

    # Log per-model results
    logger.info("=" * 60)
    logger.info("PER-MODEL RETRIEVAL RESULTS:")
    logger.info("=" * 60)

    for model_name, results in results_dict.items():
        logger.info(f"\n[{model_name}] Top {len(results)} results:")
        for rank, (image_path, score) in enumerate(results, start=1):
            img_name = Path(image_path).name
            logger.info(f"  Rank {rank}: {img_name} (score: {score:.4f})")

            if image_path not in image_info:
                image_info[image_path] = {
                    'image_path': image_path,
                    'provenance': {}
                }
            image_info[image_path]['provenance'][model_name] = rank

    combined = list(image_info.values())

    # Log duplicates (images returned by multiple models)
    logger.info("\n" + "=" * 60)
    logger.info("DUPLICATE ANALYSIS (images returned by multiple models):")
    logger.info("=" * 60)
    duplicates = [item for item in combined if len(item['provenance']) > 1]
    if duplicates:
        for item in duplicates:
            img_name = Path(item['image_path']).name
            models_ranks = ", ".join([f"{m}@rank{r}" for m, r in item['provenance'].items()])
            logger.info(f"  {img_name}: {models_ranks}")
        logger.info(f"\nTotal duplicates: {len(duplicates)} / {len(combined)} unique images")
    else:
        logger.info("  No duplicates found")

    # Deterministic shuffle based on query image hash
    seed = hash(query_image_hash) & 0xFFFFFFFF
    logger.info(f"\nShuffle seed (from image hash): {seed}")
    random.Random(seed).shuffle(combined)

    for pos, item in enumerate(combined):
        item['display_position'] = pos

    # Log final order
    logger.info("\n" + "=" * 60)
    logger.info("FINAL DISPLAY ORDER (after shuffle):")
    logger.info("=" * 60)
    for item in combined[:10]:  # Show first 10
        img_name = Path(item['image_path']).name
        prov = ", ".join([f"{m}@{r}" for m, r in item['provenance'].items()])
        logger.info(f"  Pos {item['display_position']}: {img_name} [{prov}]")
    if len(combined) > 10:
        logger.info(f"  ... and {len(combined) - 10} more")
    logger.info("=" * 60 + "\n")

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
        return path.resolve()

    # Try as relative to images_dir
    full_path = images_dir / path.name
    if full_path.exists():
        return full_path.resolve()

    # Try the original path
    full_path = images_dir / relative_path
    if full_path.exists():
        return full_path.resolve()

    # Return best guess (resolved for consistency)
    return (images_dir / path.name).resolve()


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
        embeddings_dir: Directory containing embedding files
    """
    if embeddings_dir is None:
        embeddings_dir = EMBEDDINGS_DIR

    filename = EMBEDDING_FILES.get(model_name)
    if not filename:
        raise ValueError(f"Unknown model: {model_name}")

    filepath = Path(embeddings_dir) / filename
    embeddings_csv, image_paths_txt = _text_embedding_paths(model_name, Path(embeddings_dir))

    if filepath.exists():
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        storage_format = "pkl"
    elif embeddings_csv.exists() and image_paths_txt.exists():
        embeddings = np.loadtxt(embeddings_csv, delimiter=",", dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        image_paths = [
            line.strip()
            for line in image_paths_txt.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        data = {
            "embeddings": embeddings,
            "image_paths": image_paths,
        }
        storage_format = "text"
    else:
        raise FileNotFoundError(
            f"No embedding storage found for {model_name}. "
            f"Checked {filepath}, {embeddings_csv}, {image_paths_txt}"
        )

    # Ensure embedding is normalized
    embedding = embedding / np.linalg.norm(embedding)

    # Append new embedding
    data['embeddings'] = np.vstack([data['embeddings'], embedding.reshape(1, -1)])
    data['image_paths'].append(image_path)

    # Save back
    if storage_format == "pkl":
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    else:
        np.savetxt(embeddings_csv, data['embeddings'], delimiter=",", fmt="%.8f")
        image_paths_txt.write_text("\n".join(data['image_paths']) + "\n", encoding="utf-8")

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
