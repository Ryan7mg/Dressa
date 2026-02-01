"""
app.py - Gradio Web Interface for Dressa User Study

Main application that:
1. Allows users to upload dress photos
2. Searches 4 CLIP models for similar dresses
3. Displays results in a randomized gallery
4. Collects binary ratings (Similar / Not Similar)
5. Adds highly-rated uploads to corpus (dynamic growth)
"""

import gradio as gr
import numpy as np
import hashlib
from PIL import Image
from pathlib import Path
from datetime import datetime
import shutil
import logging
import os

from models import ModelManager
from utils import (
    load_embeddings, search_similar, search_all_models,
    union_and_randomize, union_and_randomize_with_provenance,
    append_to_embeddings, get_image_full_path,
    EMBEDDING_FILES
)
from database import Database

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
APP_DIR = Path(__file__).parent
UPLOADS_DIR = APP_DIR / "uploads"
IMAGES_DIR = APP_DIR / "dress_images"

# Ensure uploads directory exists
UPLOADS_DIR.mkdir(exist_ok=True)

# Global instances (loaded at startup)
model_manager = None
db = None

# Constants
TOP_K = 5  # Results per model
CORPUS_THRESHOLD = 5  # Ratings needed before adding to corpus


def init_app():
    """Initialize models and database at startup."""
    global model_manager, db

    logger.info("Initializing Dressa App...")

    # Initialize database
    db = Database()

    # Initialize model manager (models loaded lazily on first use)
    model_manager = ModelManager()

    # Pre-load embeddings into memory
    logger.info("Pre-loading embeddings...")
    for model_name in EMBEDDING_FILES.keys():
        try:
            load_embeddings(model_name)
        except FileNotFoundError:
            logger.warning(f"Embeddings not found for {model_name}")

    logger.info("Dressa App initialized!")


def save_uploaded_image(image: np.ndarray, user_id: str) -> str:
    """Save uploaded image and return path."""
    # Create user directory
    user_dir = UPLOADS_DIR / user_id
    user_dir.mkdir(exist_ok=True)

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"upload_{timestamp}.jpg"
    filepath = user_dir / filename

    # Save image
    Image.fromarray(image).save(filepath, "JPEG", quality=95)

    return str(filepath)


def compute_image_hash(image: np.ndarray) -> str:
    """Compute a deterministic hash from image content."""
    # Use a downsampled version for speed, but still unique per image
    small = Image.fromarray(image).resize((64, 64)).tobytes()
    return hashlib.md5(small).hexdigest()


def search_similar_dresses(
    image: np.ndarray,
    user_id: str,
    upload_id: str
) -> list:
    """
    Search for similar dresses using all 4 models.

    Returns list of result dicts for the gallery with provenance.
    """
    global model_manager

    # Ensure models are loaded
    if not model_manager.is_loaded('openai_clip'):
        logger.info("Loading models for first search...")
        model_manager.load_all_models()

    # Compute image hash for deterministic shuffle
    image_hash = compute_image_hash(image)
    logger.info(f"Image hash: {image_hash}")

    # Convert to PIL
    pil_image = Image.fromarray(image)

    # Get embeddings from all models
    logger.info("Extracting embeddings...")
    query_embeddings = model_manager.encode_image_all_models(pil_image)

    # Search each model
    logger.info("Searching corpus...")
    results_dict = search_all_models(query_embeddings, top_k=TOP_K)

    # Union and randomize with provenance (deterministic based on image content)
    combined_results = union_and_randomize_with_provenance(results_dict, image_hash)

    logger.info(f"Found {len(combined_results)} unique results")

    return combined_results


def get_gallery_images(results: list) -> list:
    """Convert results to gallery format."""
    gallery_images = []

    for result in results:
        image_path = result['image_path']

        # Try to find the actual file
        full_path = get_image_full_path(image_path)

        if full_path.exists():
            gallery_images.append(str(full_path))
        else:
            # Try alternative path constructions
            alt_path = IMAGES_DIR / Path(image_path).name
            if alt_path.exists():
                gallery_images.append(str(alt_path))
            else:
                logger.warning(f"Image not found: {image_path}")

    return gallery_images


def add_to_corpus(upload_id: str, filepath: str):
    """Add an upload to the corpus after threshold ratings."""
    global model_manager

    logger.info(f"Adding {filepath} to corpus...")

    # Load image
    image = Image.open(filepath)

    # Get embeddings from all models
    embeddings = model_manager.encode_image_all_models(image)

    # Append to each model's embeddings file
    for model_name, embedding in embeddings.items():
        append_to_embeddings(filepath, embedding, model_name)

    # Mark as added in database
    db.mark_added_to_corpus(upload_id)

    logger.info(f"Added to corpus: {filepath}")


# ==================== Gradio Interface ====================

def create_app():
    """Create the Gradio app interface."""

    # Session state stored in gr.State
    with gr.Blocks(title="Dressa - Dress Similarity Study", theme=gr.themes.Soft()) as app:

        # State variables
        user_id_state = gr.State(value=None)
        upload_id_state = gr.State(value=None)
        current_results_state = gr.State(value=[])
        selected_index_state = gr.State(value=None)
        rated_images_state = gr.State(value=set())

        # Header
        gr.Markdown("""
        # Dressa - Find Similar Dresses

        **How it works:**
        1. Upload a photo of a dress from your wardrobe
        2. Browse similar dresses found by our AI
        3. Rate each result: **Similar** or **Not Similar**

        Your ratings help us improve fashion search for everyone!
        """)

        # Main layout
        with gr.Row():
            # Left column: Upload
            with gr.Column(scale=1):
                gr.Markdown("### Upload Your Dress")
                upload_image = gr.Image(
                    label="Upload a dress photo",
                    type="numpy",
                    sources=["upload", "webcam"]
                )
                search_btn = gr.Button("Find Similar Dresses", variant="primary")
                status_text = gr.Markdown("")

            # Right column: Results
            with gr.Column(scale=2):
                gr.Markdown("### Similar Dresses")
                progress_text = gr.Markdown("Upload an image to start searching...")

                gallery = gr.Gallery(
                    label="Click an image to rate it",
                    columns=5,
                    rows=3,
                    height="auto",
                    object_fit="contain",
                    allow_preview=True
                )

        # Rating section (appears after clicking image)
        with gr.Row(visible=False) as rating_row:
            with gr.Column():
                selected_image = gr.Image(
                    label="Selected Image",
                    height=300,
                    interactive=False
                )
            with gr.Column():
                gr.Markdown("### Is this dress similar to yours?")
                with gr.Row():
                    similar_btn = gr.Button("Similar", variant="primary", size="lg")
                    not_similar_btn = gr.Button("Not Similar", variant="secondary", size="lg")
                rating_status = gr.Markdown("")

        # ==================== Event Handlers ====================

        def on_search(image, user_id, upload_id):
            """Handle search button click."""
            if image is None:
                return (
                    user_id, upload_id, [], [], set(),
                    "Please upload an image first.",
                    "Upload an image to start searching...",
                    gr.update(visible=False)
                )

            # Create user if needed
            if user_id is None:
                user_id = db.create_user()
                logger.info(f"New user: {user_id}")

            # Save uploaded image
            filepath = save_uploaded_image(image, user_id)
            upload_id = db.create_upload(user_id, filepath)
            logger.info(f"New upload: {upload_id}")

            # Search for similar dresses
            status = "Searching... (this may take a moment on first run)"
            results = search_similar_dresses(image, user_id, upload_id)

            # Get gallery images
            gallery_images = get_gallery_images(results)

            if not gallery_images:
                return (
                    user_id, upload_id, results, [], set(),
                    "Search complete, but no images found. Check embeddings files.",
                    "No results found.",
                    gr.update(visible=False)
                )

            progress = f"Found **{len(gallery_images)}** similar dresses. Click any image to rate it."

            return (
                user_id, upload_id, results, gallery_images, set(),
                "Search complete!",
                progress,
                gr.update(visible=False)
            )

        def on_gallery_select(evt: gr.SelectData, results, rated_images):
            """Handle gallery image selection."""
            if not results or evt.index >= len(results):
                return None, None, gr.update(visible=False), ""

            selected_result = results[evt.index]
            image_path = get_image_full_path(selected_result['image_path'])

            # Check if already rated
            if str(image_path) in rated_images:
                return (
                    evt.index, str(image_path),
                    gr.update(visible=True),
                    "You already rated this image."
                )

            return (
                evt.index, str(image_path),
                gr.update(visible=True),
                ""
            )

        def on_rate(
            rating: str,
            user_id, upload_id, results, selected_index, rated_images
        ):
            """Handle rating submission."""
            if selected_index is None or not results:
                return rated_images, "Please select an image first.", gr.update()

            result = results[selected_index]

            # Save single evaluation rating with provenance
            db.save_evaluation_rating(
                user_id=user_id,
                upload_id=upload_id,
                result_image_id=result['image_path'],
                rating=rating,
                provenance=result['provenance'],
                display_position=result['display_position']
            )

            # Update rated images set
            full_path = str(get_image_full_path(result['image_path']))
            new_rated = rated_images.copy()
            new_rated.add(full_path)

            # Check if we should add to corpus
            upload = db.get_upload(upload_id)
            if upload and upload['num_ratings'] >= CORPUS_THRESHOLD:
                if not upload['added_to_corpus']:
                    try:
                        add_to_corpus(upload_id, upload['filepath'])
                        logger.info(f"Added upload {upload_id} to corpus!")
                    except Exception as e:
                        logger.error(f"Failed to add to corpus: {e}")

            # Count progress
            total = len(results)
            rated_count = len(new_rated)

            status = f"Rated as **{rating.replace('_', ' ')}**! ({rated_count}/{total} rated)"

            return new_rated, status, gr.update(visible=False)

        # Wire up events
        search_btn.click(
            fn=on_search,
            inputs=[upload_image, user_id_state, upload_id_state],
            outputs=[
                user_id_state, upload_id_state, current_results_state,
                gallery, rated_images_state, status_text, progress_text,
                rating_row
            ]
        )

        gallery.select(
            fn=on_gallery_select,
            inputs=[current_results_state, rated_images_state],
            outputs=[selected_index_state, selected_image, rating_row, rating_status]
        )

        similar_btn.click(
            fn=lambda *args: on_rate("similar", *args),
            inputs=[
                user_id_state, upload_id_state, current_results_state,
                selected_index_state, rated_images_state
            ],
            outputs=[rated_images_state, rating_status, rating_row]
        )

        not_similar_btn.click(
            fn=lambda *args: on_rate("not_similar", *args),
            inputs=[
                user_id_state, upload_id_state, current_results_state,
                selected_index_state, rated_images_state
            ],
            outputs=[rated_images_state, rating_status, rating_row]
        )

    return app


def main():
    """Launch the Gradio app."""
    init_app()
    app = create_app()

    logger.info("Starting Gradio server...")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set True for public URL
        show_error=True
    )


if __name__ == "__main__":
    main()
