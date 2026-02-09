"""
app.py - Gradio Web Interface for Dressa User Study

Main application that:
1. Shows consent screen first
2. Allows users to upload dress photos
3. Searches 4 CLIP models for similar dresses
4. Displays results in a randomized gallery
5. Collects binary ratings (Similar / Not Similar)
6. Shows debrief screen after completion
"""

import gradio as gr
import numpy as np
import hashlib
import base64
import uuid
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
MIN_UPLOADS_FOR_DEBRIEF = 3  # Minimum uploads before showing debrief
# Optional: preload models at startup to avoid first-search delay
PRELOAD_MODELS = os.getenv("DRESSA_PRELOAD_MODELS", "0") == "1"
# Optional: allow user uploads to be added to the corpus
ENABLE_CORPUS_GROWTH = os.getenv("DRESSA_ENABLE_CORPUS_GROWTH", "0") == "1"


def init_app():
    """Initialize models and database at startup."""
    global model_manager, db

    logger.info("Initializing Dressa App...")

    # Initialize database
    db = Database()

    # Initialize model manager (models loaded lazily on first use)
    model_manager = ModelManager()

    if PRELOAD_MODELS:
        logger.info("Preloading all models (first run may take several minutes)...")
        model_manager.load_all_models()

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

    with gr.Blocks(title="Dressa - Dress Similarity Study") as app:

        # State variables
        session_id_state = gr.State(value=None)
        user_id_state = gr.State(value=None)
        upload_id_state = gr.State(value=None)
        current_results_state = gr.State(value=[])
        selected_indices_state = gr.State(value=[])
        gallery_images_state = gr.State(value=[])
        upload_count_state = gr.State(value=0)

        # ==================== CONSENT SCREEN ====================
        with gr.Column(visible=True) as consent_screen:

            gr.Markdown("""
# Fashion Similarity Study

**Student:** Ryan Magaya (2786968m@student.gla.ac.uk) | **Supervisor:** Prof. Craig Macdonald
University of Glasgow - School of Computing Science

---

## Purpose of This Study

**Research Question:** Which AI vision-language models perform best at finding similar fashion items when users upload real wardrobe photos (not professional product images)?

**Why this matters:** Existing fashion recommendation systems are trained on professional product photos. We're testing if they work equally well on the kinds of photos real users take (dress on hanger, on floor, worn by you). Your ratings will help determine which AI model is best for real-world fashion search applications.

**Academic Goal:** This research contributes to my dissertation on building AI-powered circular fashion marketplaces.

---
            """)

            # What You'll Need - highlighted box
            gr.HTML("""
                <div style="background-color: #fef3c7; border: 2px solid #f59e0b; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
                    <strong style="font-size: 16px;">What You'll Need</strong><br><br>
                    Before starting, please have ready:<br>
                    - <strong>3-5 photos of dresses</strong> from your wardrobe<br>
                    - Photos should show: dress on hanger, laid flat on floor, or worn by you<br>
                    - <strong>IMPORTANT:</strong> Crop out your face before uploading<br>
                    - Phone photos are fine - doesn't need to be professional quality<br><br>
                    <strong>Don't have photos ready?</strong> Please take them before clicking "Start" below.
                </div>
            """)

            # Warning banner
            gr.HTML("""
                <div style="background-color: #dc2626; color: white; padding: 12px; border-radius: 8px; text-align: center; font-weight: bold; margin-bottom: 16px;">
                    DO NOT UPLOAD PHOTOS WITH YOUR FACE - Crop out faces before uploading
                </div>
            """)

            # Two-column layout for remaining info
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("""
**What You'll Do (5 min)**
1. Upload a dress photo
2. View AI-recommended similar dresses
3. Click which ones are "Similar" or "Not Similar"
4. Repeat for 3-5 of your dress photos

**Data Collected**
- Your dress photos and ratings
- Anonymous session ID only
- No names, emails, or personal info
- Used for comparing 4 AI models
                    """)

                with gr.Column(scale=1):
                    gr.Markdown("""
**Your Rights**
- Voluntary and anonymous
- Close browser anytime to withdraw
- Request deletion: email session ID

**By clicking "I Agree and Start":**
- You are 16+ years old
- You have dress photos ready
- You will NOT upload photos with faces
- You consent to anonymous data collection

**Questions?** 2786968m@student.gla.ac.uk
                    """)

            with gr.Row():
                agree_btn = gr.Button("I Agree and Start", variant="primary", size="lg")
                disagree_btn = gr.Button("I Do Not Agree", variant="secondary", size="lg")

            disagree_message = gr.Markdown("", visible=False)

        # ==================== MAIN APP SCREEN ====================
        with gr.Column(visible=False) as main_app_screen:

            gr.Markdown("""
# Dressa - Find Similar Dresses

**How it works:**
1. Upload a photo of a dress from your wardrobe
2. Browse similar dresses found by our AI
3. Tap dresses that look similar to yours

Your ratings help us improve fashion search!
            """)

            # Progress tracker
            upload_progress = gr.Markdown("**Uploads: 0 / 3-5**")

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

                    # Finish button (appears after minimum uploads)
                    finish_btn = gr.Button("Finish Study", variant="secondary", visible=False)

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

                    # Hidden textbox for selected indices
                    with gr.Row(visible=False) as hidden_row:
                        selected_indices_input = gr.Textbox(
                            value="[]",
                            elem_id="selected-indices-input"
                        )

                    # Submit button
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

        # ==================== DEBRIEF SCREEN ====================
        with gr.Column(visible=False) as debrief_screen:

            gr.Markdown("""
# Thank You for Participating

You helped test 4 AI models: OpenAI CLIP, FashionCLIP, Marqo-FashionCLIP, Marqo-FashionSigLIP
            """)

            gr.Markdown("**Your Session ID (select and copy):**")
            session_id_display = gr.Textbox(
                label="",
                interactive=True,
                elem_id="session-id-display"
            )

            gr.Markdown("""
**Your data:**
- Stored anonymously for dissertation research
- Used to compare AI model performance

**Questions?** 2786968m@student.gla.ac.uk
**Delete data?** Email with session ID above

**Supervisor:** craig.macdonald@glasgow.ac.uk
            """)

            close_btn = gr.Button("Close", variant="primary", size="lg")
            close_message = gr.Markdown("")

        # ==================== Event Handlers ====================

        def on_agree():
            """Handle consent agreement - generate session ID and show main app."""
            session_id = str(uuid.uuid4())
            user_id = db.create_user()
            logger.info(f"New session: {session_id}, user: {user_id}")
            return (
                session_id,
                user_id,
                gr.update(visible=False),  # Hide consent screen
                gr.update(visible=True),   # Show main app
                gr.update(visible=False),  # Keep debrief hidden
            )

        def on_disagree():
            """Handle consent disagreement."""
            return gr.update(
                value="Thank you. You may close this window.",
                visible=True
            )

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

            @media (max-width: 767px) {
                .result-item {
                    min-height: 140px;
                }
            }
            </style>
            """

            # Generate image grid with base64 encoded images
            grid_items = []
            for i, img_path in enumerate(gallery_images):
                selected_class = "selected" if i in selected_indices else ""
                try:
                    with Image.open(img_path) as img:
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

        def on_search(image, user_id, upload_id, upload_count, progress=gr.Progress()):
            """Handle search button click."""
            progress(0.0, desc="Validating input")
            if image is None:
                return (
                    user_id, upload_id, [], [], [], upload_count,
                    "Please upload an image first.",
                    "Upload an image to start searching...",
                    gr.update(visible=False),
                    "",
                    "[]",
                    gr.update(visible=False, interactive=False),
                    "",
                    f"**Uploads: {upload_count} / 3-5**",
                    gr.update(visible=upload_count >= MIN_UPLOADS_FOR_DEBRIEF)
                )

            # Save uploaded image
            progress(0.05, desc="Saving upload")
            filepath = save_uploaded_image(image, user_id)
            upload_id = db.create_upload(user_id, filepath)
            new_upload_count = upload_count + 1
            logger.info(f"New upload: {upload_id} (count: {new_upload_count})")

            # Search for similar dresses
            results = search_similar_dresses(image, user_id, upload_id, progress=progress)

            # Filter results to corpus-only images and build gallery
            progress(0.92, desc="Filtering images")
            filtered_results, gallery_images = filter_results_for_gallery(results)

            if not gallery_images:
                return (
                    user_id, upload_id, filtered_results, [], [], new_upload_count,
                    "Search complete, but no corpus images found. Check embeddings files.",
                    "No results found.",
                    gr.update(visible=False),
                    "",
                    "[]",
                    gr.update(visible=False, interactive=False),
                    "",
                    f"**Uploads: {new_upload_count} / 3-5**",
                    gr.update(visible=new_upload_count >= MIN_UPLOADS_FOR_DEBRIEF)
                )

            progress(1.0, desc="Done")
            progress_msg = f"Found **{len(gallery_images)}** similar dresses."
            grid_html = generate_results_grid_html(gallery_images, [])

            return (
                user_id, upload_id, filtered_results, [], gallery_images, new_upload_count,
                "Search complete!",
                progress_msg,
                gr.update(visible=True),
                grid_html,
                "[]",
                gr.update(visible=True, interactive=True, value="Submit (0 selected)"),
                "",
                f"**Uploads: {new_upload_count} / 3-5**",
                gr.update(visible=new_upload_count >= MIN_UPLOADS_FOR_DEBRIEF)
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
                gr.update(value=btn_text, interactive=True)
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
                    ""
                )

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

            # Check if we should add to corpus (disabled by default)
            if ENABLE_CORPUS_GROWTH:
                upload = db.get_upload(upload_id)
                if upload and upload['num_ratings'] >= CORPUS_THRESHOLD:
                    if not upload['added_to_corpus']:
                        try:
                            add_to_corpus(upload_id, upload['filepath'])
                            logger.info(f"Added upload {upload_id} to corpus!")
                        except Exception as e:
                            logger.error(f"Failed to add to corpus: {e}")
            else:
                logger.info("Corpus growth disabled; skipping add_to_corpus")

            status = f"**Thank you!** Saved {similar_count} similar and {not_similar_count} not similar ratings. Upload another dress to continue."

            return (
                status,
                gr.update(visible=False),
                "[]",
                gr.update(visible=False),
                ""
            )

        def on_finish(session_id):
            """Handle finish study button - show debrief screen."""
            return (
                gr.update(visible=False),  # Hide main app
                gr.update(visible=True),   # Show debrief
                session_id  # Display session ID
            )

        def on_close():
            """Handle close button on debrief screen."""
            return gr.update(value="Thank you for participating. You may close this window.")

        # ==================== Wire up events ====================

        # Consent screen events
        agree_btn.click(
            fn=on_agree,
            inputs=[],
            outputs=[
                session_id_state,
                user_id_state,
                consent_screen,
                main_app_screen,
                debrief_screen
            ]
        )

        disagree_btn.click(
            fn=on_disagree,
            inputs=[],
            outputs=[disagree_message]
        )

        # Search events
        search_btn.click(
            fn=on_search,
            inputs=[upload_image, user_id_state, upload_id_state, upload_count_state],
            outputs=[
                user_id_state, upload_id_state, current_results_state,
                selected_indices_state, gallery_images_state, upload_count_state,
                status_text, progress_text,
                selection_instructions, results_grid_html,
                selected_indices_input, submit_btn, submit_status,
                upload_progress, finish_btn
            ]
        )

        # Selection change events
        selected_indices_input.change(
            fn=on_selection_change,
            inputs=[selected_indices_input, gallery_images_state],
            outputs=[selected_indices_state, submit_btn]
        )

        # Submit events
        submit_btn.click(
            fn=on_submit,
            inputs=[
                user_id_state, upload_id_state, current_results_state,
                selected_indices_input, gallery_images_state
            ],
            outputs=[submit_status, selection_instructions, selected_indices_input, submit_btn, results_grid_html]
        )

        # Finish study events
        finish_btn.click(
            fn=on_finish,
            inputs=[session_id_state],
            outputs=[main_app_screen, debrief_screen, session_id_display]
        )

        # Close events
        close_btn.click(
            fn=on_close,
            inputs=[],
            outputs=[close_message]
        )

    return app


def main():
    """Launch the Gradio app."""
    init_app()
    app = create_app()

    logger.info("Starting Gradio server...")
    app.queue()
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    app.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        share=False,
        show_error=True,
        allowed_paths=[str(IMAGES_DIR), str(UPLOADS_DIR)],
        js=TOGGLE_JS
    )


if __name__ == "__main__":
    main()
