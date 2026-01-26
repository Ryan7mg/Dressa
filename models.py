"""
models.py - Model Manager for Dressa App

Handles loading and inference for 4 CLIP models:
1. OpenAI CLIP (ViT-B/32) - General baseline
2. FashionCLIP - Fashion-specialized (transformers API)
3. Marqo-FashionCLIP - Fashion-tuned CLIP
4. Marqo-FashionSigLIP - Best performing (46.7% Recall@5)

All embeddings are L2-normalized for cosine similarity.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """Manages loading and inference for all 4 CLIP models."""

    # Model configurations
    MODEL_CONFIGS = {
        'openai_clip': {
            'type': 'openclip',
            'model_name': 'ViT-B-32',
            'pretrained': 'openai',
            'embedding_dim': 512
        },
        'fashion_clip': {
            'type': 'transformers',
            'model_name': 'patrickjohncyh/fashion-clip',
            'embedding_dim': 512
        },
        'marqo_fashion_clip': {
            'type': 'openclip',
            'model_name': 'ViT-B-16',
            'pretrained': 'hf-hub:Marqo/marqo-fashionCLIP',
            'embedding_dim': 512
        },
        'marqo_fashion_siglip': {
            'type': 'openclip',
            'model_name': 'ViT-B-16-SigLIP',
            'pretrained': 'hf-hub:Marqo/marqo-fashionSigLIP',
            'embedding_dim': 512
        }
    }

    def __init__(self, device: Optional[str] = None):
        """
        Initialize ModelManager.

        Args:
            device: 'cuda', 'mps', or 'cpu'. Auto-detects if None.
        """
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        # Storage for loaded models
        self.models = {}
        self.preprocessors = {}
        self.tokenizers = {}

    def load_all_models(self):
        """Load all 4 CLIP models. Call this at app startup."""
        logger.info("Loading all models...")

        for model_name in self.MODEL_CONFIGS.keys():
            self.load_model(model_name)

        logger.info("All models loaded successfully!")

    def load_model(self, model_name: str):
        """
        Load a single model by name.

        Args:
            model_name: One of 'openai_clip', 'fashion_clip',
                       'marqo_fashion_clip', 'marqo_fashion_siglip'
        """
        if model_name in self.models:
            logger.info(f"{model_name} already loaded")
            return

        config = self.MODEL_CONFIGS.get(model_name)
        if not config:
            raise ValueError(f"Unknown model: {model_name}")

        logger.info(f"Loading {model_name}...")

        if config['type'] == 'openclip':
            self._load_openclip_model(model_name, config)
        elif config['type'] == 'transformers':
            self._load_transformers_model(model_name, config)

        logger.info(f"{model_name} loaded successfully")

    def _load_openclip_model(self, model_name: str, config: dict):
        """Load an OpenCLIP-based model."""
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            config['model_name'],
            pretrained=config['pretrained']
        )
        model = model.to(self.device)
        model.eval()

        self.models[model_name] = model
        self.preprocessors[model_name] = preprocess

    def _load_transformers_model(self, model_name: str, config: dict):
        """Load a HuggingFace transformers model (FashionCLIP)."""
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained(config['model_name'])
        processor = CLIPProcessor.from_pretrained(config['model_name'])

        model = model.to(self.device)
        model.eval()

        self.models[model_name] = model
        self.preprocessors[model_name] = processor

    def encode_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        model_name: str
    ) -> np.ndarray:
        """
        Encode an image to a normalized embedding vector.

        Args:
            image: File path, PIL Image, or numpy array
            model_name: Which model to use for encoding

        Returns:
            Normalized embedding vector of shape (512,)
        """
        # Ensure model is loaded
        if model_name not in self.models:
            self.load_model(model_name)

        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
        else:
            image = image.convert('RGB')

        config = self.MODEL_CONFIGS[model_name]

        with torch.no_grad():
            if config['type'] == 'openclip':
                embedding = self._encode_openclip(image, model_name)
            else:
                embedding = self._encode_transformers(image, model_name)

        # Normalize embedding (critical for cosine similarity!)
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def _encode_openclip(self, image: Image.Image, model_name: str) -> np.ndarray:
        """Encode image using OpenCLIP model."""
        model = self.models[model_name]
        preprocess = self.preprocessors[model_name]

        # Preprocess and move to device
        image_tensor = preprocess(image).unsqueeze(0).to(self.device)

        # Encode
        image_features = model.encode_image(image_tensor)

        # Convert to numpy
        return image_features.cpu().numpy().flatten()

    def _encode_transformers(self, image: Image.Image, model_name: str) -> np.ndarray:
        """Encode image using transformers model (FashionCLIP)."""
        model = self.models[model_name]
        processor = self.preprocessors[model_name]

        # Process image
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Encode
        image_features = model.get_image_features(**inputs)

        # Convert to numpy
        return image_features.cpu().numpy().flatten()

    def encode_image_all_models(
        self,
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> dict:
        """
        Encode an image with all 4 models.

        Args:
            image: File path, PIL Image, or numpy array

        Returns:
            Dict mapping model_name -> normalized embedding
        """
        embeddings = {}
        for model_name in self.MODEL_CONFIGS.keys():
            embeddings[model_name] = self.encode_image(image, model_name)
        return embeddings

    def get_model_names(self) -> list:
        """Return list of available model names."""
        return list(self.MODEL_CONFIGS.keys())

    def is_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded."""
        return model_name in self.models


# Convenience function for quick testing
def test_models():
    """Test that all models can be loaded and encode images."""
    import tempfile

    # Create a dummy test image
    test_image = Image.new('RGB', (224, 224), color='red')

    manager = ModelManager()
    manager.load_all_models()

    print("\nTesting encoding with all models:")
    for model_name in manager.get_model_names():
        embedding = manager.encode_image(test_image, model_name)
        norm = np.linalg.norm(embedding)
        print(f"  {model_name}: shape={embedding.shape}, norm={norm:.4f}")

    print("\nAll models working correctly!")


if __name__ == "__main__":
    test_models()
