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
import base64
from PIL import Image
from pathlib import Path
from datetime import datetime
import shutil
import logging
import os
import io
import time

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

# Paths - use resolve() for consistent absolute paths
APP_DIR = Path(__file__).parent.resolve()
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
    upload_id: str,
    progress=None
) -> list:
    """
    Search for similar dresses using all 4 models.

    Returns list of result dicts for the gallery with provenance.
    """
    global model_manager

    def _progress(value: float, desc: str):
        if progress is not None:
            progress(value, desc=desc)

    # Ensure models are loaded
    if not model_manager.is_loaded('openai_clip'):
        logger.info("Loading models for first search...")
        _progress(0.1, "Loading models (first time can take a few minutes)")
        load_start = time.perf_counter()
        model_manager.load_all_models()
        load_elapsed = time.perf_counter() - load_start
        logger.info(f"Model load total time: {load_elapsed:.2f}s")

    # Compute image hash for deterministic shuffle
    image_hash = compute_image_hash(image)
    logger.info(f"Image hash: {image_hash}")

    # Convert to PIL
    pil_image = Image.fromarray(image)

    # Get embeddings from all models
    logger.info("Extracting embeddings...")
    _progress(0.5, "Encoding image with 4 models")
    encode_start = time.perf_counter()
    query_embeddings = model_manager.encode_image_all_models(pil_image)
    encode_elapsed = time.perf_counter() - encode_start
    logger.info(f"Encode time (all models): {encode_elapsed:.2f}s")

    # Search each model
    logger.info("Searching corpus...")
    _progress(0.7, "Searching embeddings")
    search_start = time.perf_counter()
    results_dict = search_all_models(query_embeddings, top_k=TOP_K)
    search_elapsed = time.perf_counter() - search_start
    logger.info(f"Search time (all models): {search_elapsed:.2f}s")

    # Union and randomize with provenance (deterministic based on image content)
    _progress(0.85, "Preparing results")
    union_start = time.perf_counter()
    combined_results = union_and_randomize_with_provenance(results_dict, image_hash)
    union_elapsed = time.perf_counter() - union_start
    logger.info(f"Union/shuffle time: {union_elapsed:.2f}s")

    logger.info(f"Found {len(combined_results)} unique results")

    return combined_results


def _is_under_dir(path: Path, base_dir: Path) -> bool:
    """Return True if path is within base_dir (after resolving)."""
    try:
        path.resolve().relative_to(base_dir.resolve())
        return True
    except ValueError:
        return False


def filter_results_for_gallery(results: list) -> tuple[list, list]:
    """
    Filter search results to only images that live in dress_images.

    Returns:
        filtered_results: results aligned with gallery order
        gallery_images: list of resolved image paths
    """
    filtered_results = []
    gallery_images = []

    for result in results:
        image_path = result['image_path']

        # Skip user uploads - they should not appear in search results
        if '/uploads/' in str(image_path).replace('\\', '/'):
            logger.info(f"Skipping upload result: {image_path}")
            continue

        # Resolve to an actual file
        full_path = get_image_full_path(image_path)

        if not full_path.exists():
            # Try alternative path constructions
            alt_path = IMAGES_DIR / Path(image_path).name
            if alt_path.exists():
                full_path = alt_path
            else:
                logger.warning(f"Image not found: {image_path}")
                continue

        # Ensure images are only served from the corpus folder
        if not _is_under_dir(full_path, IMAGES_DIR):
            logger.warning(f"Skipping non-corpus image: {full_path}")
            continue

        filtered_results.append(result)
        gallery_images.append(str(full_path))

    return filtered_results, gallery_images


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

# JavaScript for toggle selection functionality (passed to launch() for Gradio 6.0+)
TOGGLE_JS = """
function toggleSelection(index) {
    const item = document.querySelector(`[data-index="${index}"]`);
    if (!item) return;
    item.classList.toggle('selected');

    // Get all selected indices
    const selected = [...document.querySelectorAll('.result-item.selected')]
        .map(el => parseInt(el.dataset.index));

    console.log('Selected indices:', selected);

    // Find the textbox by elem_id - Gradio wraps it in a div with the ID
    let input = null;
    const container = document.getElementById('selected-indices-input');
    if (container) {
        input = container.querySelector('textarea') || container.querySelector('input');
        console.log('Found container, input:', input);
    }

    // Fallback: search all textareas
    if (!input) {
        document.querySelectorAll('textarea').forEach(el => {
            console.log('Textarea value:', el.value);
            if (el.value !== undefined && el.value.startsWith('[')) {
                input = el;
            }
        });
    }

    if (input) {
        console.log('Updating input from', input.value, 'to', JSON.stringify(selected));
        input.value = JSON.stringify(selected);
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
        // Also try focus/blur to trigger Gradio
        input.focus();
        input.blur();
    } else {
        console.log('Could not find input element');
    }

    // Find submit button by text content
    let submitBtn = null;
    document.querySelectorAll('button').forEach(btn => {
        if (btn.textContent && btn.textContent.includes('Submit')) {
            submitBtn = btn;
        }
    });

    if (submitBtn) {
        console.log('Found submit button, updating text');
        submitBtn.textContent = `Submit (${selected.length} selected)`;
    } else {
        console.log('Could not find submit button');
    }
}
window.toggleSelection = toggleSelection;
"""

def create_app():
    """Create the Gradio app interface."""

    # Session state stored in gr.State
    with gr.Blocks(title="Dressa - Dress Similarity Study") as app:

        # State variables
        user_id_state = gr.State(value=None)
        upload_id_state = gr.State(value=None)
        current_results_state = gr.State(value=[])
        selected_indices_state = gr.State(value=[])  # List of selected image indices
        gallery_images_state = gr.State(value=[])  # Store gallery image paths

        # Header
        gr.Markdown("""
        # Dressa - Find Similar Dresses

        **How it works:**
        1. Upload a photo of a dress from your wardrobe
        2. Browse similar dresses found by our AI
        3. Tap dresses that look similar to yours

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

                # Instructions for selection
                selection_instructions = gr.Markdown(
                    "**Tap the dresses that are similar to yours. Tap again to deselect. "
                    "When you are done, press Submit.**",
                    visible=False
                )

                # Custom HTML grid for toggle selection
                results_grid_html = gr.HTML(value="", elem_id="results-grid-container")

                # Hidden textbox to communicate selected indices from JS to Python
                selected_indices_input = gr.Textbox(
                    value="[]",
                    visible=False,
                    elem_id="selected-indices-input"
                )

                # Submit button - always interactive (0 selected is valid = none similar)
                submit_btn = gr.Button(
                    "Submit (0 selected)",
                    variant="primary",
                    size="lg",
                    visible=False,
                    interactive=True,
                    elem_id="submit-btn"
                )

                # Status message
                submit_status = gr.Markdown("")

        # ==================== Event Handlers ====================

        def generate_results_grid_html(gallery_images: list, selected_indices: list) -> str:
            """Generate HTML for the results grid with toggle selection."""
            if not gallery_images:
                return ""

            # CSS styles
            css = """
            <style>
            .results-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 12px;
                padding: 12px;
                padding-bottom: 20px;
            }

            @media (min-width: 768px) {
                .results-grid {
                    grid-template-columns: repeat(4, 1fr);
                }
            }

            @media (min-width: 1024px) {
                .results-grid {
                    grid-template-columns: repeat(5, 1fr);
                }
            }

            .result-item {
                position: relative;
                aspect-ratio: 3/4;
                cursor: pointer;
                border-radius: 8px;
                overflow: hidden;
                border: 3px solid transparent;
                transition: border-color 0.2s, transform 0.1s;
                -webkit-tap-highlight-color: transparent;
                user-select: none;
                background: #f0f0f0;
            }

            .result-item:hover {
                transform: scale(1.02);
            }

            .result-item:active {
                transform: scale(0.98);
            }

            .result-item img {
                width: 100%;
                height: 100%;
                object-fit: cover;
                pointer-events: none;
            }

            .result-item.selected {
                border-color: #22c55e;
                box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.3);
            }

            .result-item.selected::after {
                content: '';
                position: absolute;
                top: 8px;
                right: 8px;
                width: 32px;
                height: 32px;
                background: #22c55e url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='white'%3E%3Cpath d='M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z'/%3E%3C/svg%3E") center/60% no-repeat;
                border-radius: 50%;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }

            .result-item .index-badge {
                position: absolute;
                bottom: 8px;
                left: 8px;
                background: rgba(0,0,0,0.7);
                color: white;
                padding: 4px 10px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: 500;
            }

            /* Touch-friendly minimum size */
            @media (max-width: 767px) {
                .result-item {
                    min-height: 140px;
                }
            }
            </style>
            """

            # JavaScript - no separate script tag needed, use inline handlers

            # Generate image grid with base64 encoded images
            grid_items = []
            for i, img_path in enumerate(gallery_images):
                selected_class = "selected" if i in selected_indices else ""
                # Convert image to base64 data URI
                try:
                    with Image.open(img_path) as img:
                        # Resize for web display (max 400px width)
                        img.thumbnail((400, 600), Image.Resampling.LANCZOS)
                        buffer = io.BytesIO()
                        img.save(buffer, format='JPEG', quality=85)
                        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        img_src = f"data:image/jpeg;base64,{img_base64}"
                except Exception as e:
                    logger.error(f"Failed to load image {img_path}: {e}")
                    continue
                grid_items.append(f'''
                <div class="result-item {selected_class}" data-index="{i}" onclick="toggleSelection({i})">
                    <img src="{img_src}" alt="Dress {i+1}">
                    <span class="index-badge">{i+1}</span>
                </div>
                ''')

            html = f"""
            {css}
            <div class="results-grid">
                {''.join(grid_items)}
            </div>
            """

            return html

        def on_search(image, user_id, upload_id, progress=gr.Progress()):
            """Handle search button click."""
            progress(0.0, desc="Validating input")
            if image is None:
                return (
                    user_id, upload_id, [], [], [],
                    "Please upload an image first.",
                    "Upload an image to start searching...",
                    gr.update(visible=False),  # selection_instructions
                    "",  # results_grid_html
                    "[]",  # selected_indices_input
                    gr.update(visible=False, interactive=False),  # submit_btn
                    ""  # submit_status
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
            results = search_similar_dresses(image, user_id, upload_id, progress=progress)

            # Filter results to corpus-only images and build gallery
            filtered_results, gallery_images = filter_results_for_gallery(results)

            if not gallery_images:
                return (
                    user_id, upload_id, filtered_results, [], [],
                    "Search complete, but no images found. Check embeddings files.",
                    "No results found.",
                    gr.update(visible=False),
                    "",
                    "[]",
                    gr.update(visible=False, interactive=False),
                    ""
                )

            progress_msg = f"Found **{len(gallery_images)}** similar dresses."

            # Generate HTML grid
            grid_html = generate_results_grid_html(gallery_images, [])

            return (
                user_id, upload_id, filtered_results, [], gallery_images,
                "Search complete!",
                progress_msg,
                gr.update(visible=True),  # selection_instructions
                grid_html,  # results_grid_html
                "[]",  # selected_indices_input
                gr.update(visible=True, interactive=True, value="Submit (0 selected)"),  # submit_btn
                ""  # submit_status
            )

        def on_selection_change(selected_indices_json, gallery_images):
            """Handle selection change from JavaScript."""
            import json
            try:
                selected_indices = json.loads(selected_indices_json)
            except (json.JSONDecodeError, TypeError):
                selected_indices = []

            count = len(selected_indices)
            btn_text = f"Submit ({count} selected)"

            return (
                selected_indices,
                gr.update(value=btn_text, interactive=True)  # Always interactive - 0 selected is valid
            )

        def on_submit(user_id, upload_id, results, selected_indices_json, gallery_images):
            """Handle submit button click - save all ratings."""
            import json

            logger.info("=" * 50)
            logger.info("SUBMIT BUTTON CLICKED")
            logger.info(f"User ID: {user_id}")
            logger.info(f"Upload ID: {upload_id}")
            logger.info(f"Selected indices JSON: {selected_indices_json}")

            try:
                selected_indices = json.loads(selected_indices_json)
            except (json.JSONDecodeError, TypeError):
                selected_indices = []

            logger.info(f"Parsed selected indices: {selected_indices}")
            logger.info(f"Total results: {len(results) if results else 0}")

            if not results:
                logger.warning("No results to rate")
                return (
                    "No results to rate.",
                    gr.update(visible=False),
                    "[]",
                    gr.update(visible=False),
                    ""  # Clear the grid
                )

            # 0 selected is valid - means none are similar
            selected_set = set(selected_indices)

            # Save ratings for all images
            similar_count = 0
            not_similar_count = 0

            logger.info("Saving ratings to database...")
            for i, result in enumerate(results):
                if i in selected_set:
                    rating = "similar"
                    similar_count += 1
                else:
                    rating = "not_similar"
                    not_similar_count += 1

                logger.info(f"  [{i}] {result['image_path']}: {rating} (provenance: {result['provenance']})")

                db.save_evaluation_rating(
                    user_id=user_id,
                    upload_id=upload_id,
                    result_image_id=result['image_path'],
                    rating=rating,
                    provenance=result['provenance'],
                    display_position=result['display_position']
                )

            logger.info(f"SAVED: {similar_count} similar, {not_similar_count} not similar")
            logger.info("=" * 50)

            # Check if we should add to corpus
            upload = db.get_upload(upload_id)
            if upload and upload['num_ratings'] >= CORPUS_THRESHOLD:
                if not upload['added_to_corpus']:
                    try:
                        add_to_corpus(upload_id, upload['filepath'])
                        logger.info(f"Added upload {upload_id} to corpus!")
                    except Exception as e:
                        logger.error(f"Failed to add to corpus: {e}")

            status = f"**Thank you!** Saved {similar_count} similar and {not_similar_count} not similar ratings. Upload another dress to continue."

            return (
                status,
                gr.update(visible=False),  # Hide instructions
                "[]",  # Reset selected indices
                gr.update(visible=False),  # Hide submit button
                ""  # Clear the grid HTML
            )

        # Wire up events
        search_btn.click(
            fn=on_search,
            inputs=[upload_image, user_id_state, upload_id_state],
            outputs=[
                user_id_state, upload_id_state, current_results_state,
                selected_indices_state, gallery_images_state,
                status_text, progress_text,
                selection_instructions, results_grid_html,
                selected_indices_input, submit_btn, submit_status
            ]
        )

        # Handle selection changes from JavaScript
        selected_indices_input.change(
            fn=on_selection_change,
            inputs=[selected_indices_input, gallery_images_state],
            outputs=[selected_indices_state, submit_btn]
        )

        # Handle submit button
        submit_btn.click(
            fn=on_submit,
            inputs=[
                user_id_state, upload_id_state, current_results_state,
                selected_indices_input, gallery_images_state
            ],
            outputs=[submit_status, selection_instructions, selected_indices_input, submit_btn, results_grid_html]
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
        show_error=True,
        allowed_paths=[str(IMAGES_DIR), str(UPLOADS_DIR)],
        js=TOGGLE_JS
    )


if __name__ == "__main__":
    main()
