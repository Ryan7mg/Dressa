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
    encode_start = time.perf_counter()
    query_embeddings = model_manager.encode_image_all_models(pil_image)
    encode_elapsed = time.perf_counter() - encode_start
    logger.info(f"Encode time (all models): {encode_elapsed:.2f}s")

    # Search each model
    logger.info("Searching corpus...")
    search_start = time.perf_counter()
    results_dict = search_all_models(query_embeddings, top_k=TOP_K)
    search_elapsed = time.perf_counter() - search_start
    logger.info(f"Search time (all models): {search_elapsed:.2f}s")

    # Union and randomize with provenance (deterministic based on image content)
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

# Global styles for a more polished, responsive UI
FONT_LINKS_HTML = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,600;700&display=swap" rel="stylesheet">
"""

APP_CSS = """
:root {
    --bg: #fefdfb;
    --card: #ffffff;
    --ink: #191816;
    --muted: #6f6b65;
    --accent: #d46b3b;
    --accent-2: #236b5b;
    --border: #e7ddd1;
    --shadow: 0 16px 44px rgba(24, 16, 8, 0.10);
    --display: 'Fraunces', 'Space Grotesk', 'Helvetica Neue', sans-serif;
    --body: 'Space Grotesk', 'Helvetica Neue', Helvetica, sans-serif;
}

body, .gradio-container {
    background: var(--bg);
    color: var(--ink);
    font-family: var(--body);
    overflow-x: hidden;
}

.gradio-container {
    width: 100% !important;
    max-width: none !important;
    margin: 0 !important;
    min-height: 100vh;
    padding: 28px clamp(16px, 3vw, 40px) 80px;
    box-sizing: border-box;
}

.gradio-container .container,
.gradio-container .wrap,
.gradio-container .contain,
.gradio-container .block,
.gradio-container .gr-row,
.gradio-container .gr-column,
.gradio-container .gr-form,
.gradio-container .gr-panel {
    max-width: none !important;
    width: 100% !important;
    overflow: visible !important;
}

.gradio-container .wrap::-webkit-scrollbar,
.gradio-container .block::-webkit-scrollbar,
.gradio-container .gr-row::-webkit-scrollbar,
.gradio-container .gr-column::-webkit-scrollbar,
.gradio-container .gr-panel::-webkit-scrollbar {
    width: 0;
    height: 0;
}

#main-row {
    gap: 26px;
    align-items: flex-start;
}

#upload-col, #results-col {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 16px;
    box-shadow: 0 10px 28px rgba(24, 16, 8, 0.06);
    overflow: visible !important;
}

#results-grid-container {
    overflow: visible !important;
}

#hero {
    background: linear-gradient(135deg, #fff8f1, #fff2e8);
    border: 1px solid var(--border);
    border-radius: 22px;
    box-shadow: var(--shadow);
    padding: 22px 24px;
    margin-bottom: 18px;
}

#hero .hero-title {
    font-family: var(--display);
    font-size: clamp(28px, 3vw, 36px);
    font-weight: 700;
    margin-bottom: 6px;
}

h1, h2, h3 {
    font-family: var(--display);
    letter-spacing: -0.01em;
}

#hero .hero-subtitle {
    color: var(--muted);
    font-size: 16px;
    margin-bottom: 16px;
}

#hero .hero-steps {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
}

#hero .step {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 12px 14px;
    font-size: 14px;
    display: flex;
    gap: 10px;
    align-items: center;
}

#hero .step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: var(--accent);
    color: #fff;
    font-weight: 700;
    font-size: 12px;
}

#upload-progress {
    font-weight: 600;
    color: var(--muted);
    margin-bottom: 6px;
}

#status-text, #progress-text, #submit-status {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 10px 12px;
    box-shadow: 0 8px 18px rgba(24, 16, 8, 0.05);
}

#progress-text {
    margin-bottom: 8px;
}

#selection-instructions {
    font-weight: 600;
    color: var(--ink);
}

#selection-count {
    color: var(--muted);
    margin-bottom: 8px;
}

#upload-image .image-preview,
#upload-image .image-container {
    border-radius: 16px;
    overflow: hidden;
}

#upload-image .image-preview .toolbar,
#upload-image .image-preview .image-preview-controls,
#upload-image .image-preview .icon,
#upload-image .image-preview .buttons,
#upload-image .image-preview .toolbar button[aria-label="Clear"],
#upload-image .image-preview .toolbar button[aria-label="Remove"],
#upload-image .image-preview .toolbar button[aria-label="Fullscreen"],
#upload-image .image-preview .toolbar button[aria-label="Zoom"],
#upload-image .image-preview .toolbar button[aria-label="View"],
#upload-image .image-preview button[aria-label="Clear"],
#upload-image .image-preview button[aria-label="Remove"],
#upload-image .image-preview button[aria-label="Fullscreen"] {
    display: none !important;
}

.gr-markdown, .gr-markdown > div {
    overflow: visible !important;
}

#status-text, #progress-text, #submit-status {
    white-space: normal !important;
}

.results-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px;
    padding: 6px;
}

.result-item {
    position: relative;
    aspect-ratio: 3/4;
    cursor: pointer;
    border-radius: 16px;
    overflow: hidden;
    border: 2px solid transparent;
    transition: border-color 0.2s ease, transform 0.12s ease, box-shadow 0.2s ease;
    background: #f4f1ec;
    padding: 0;
}

.result-item img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}

.result-item:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(24, 16, 8, 0.12);
}

.result-item.selected {
    border-color: var(--accent-2);
    box-shadow: 0 0 0 3px rgba(31, 111, 91, 0.2);
}

.result-item.selected::after {
    content: '';
    position: absolute;
    top: 8px;
    right: 8px;
    width: 32px;
    height: 32px;
    background: var(--accent-2) url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='white'%3E%3Cpath d='M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z'/%3E%3C/svg%3E") center/60% no-repeat;
    border-radius: 50%;
    box-shadow: 0 2px 6px rgba(0,0,0,0.2);
}

.result-item .index-badge {
    position: absolute;
    bottom: 8px;
    left: 8px;
    background: rgba(0, 0, 0, 0.7);
    color: white;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
}

#submit-btn button,
#submit-btn {
    width: 100% !important;
}

.gr-button.primary {
    background: var(--accent) !important;
    border: none !important;
    color: #fff !important;
}

.gr-button.secondary {
    background: #fff !important;
    border: 1px solid var(--border) !important;
    color: var(--ink) !important;
}

.footer, footer {
    display: none !important;
}

div[data-testid="progress"] {
    display: none !important;
}

.progress, .progress-bar, .progress-text, .wrap .progress {
    display: none !important;
}
"""

# JavaScript for toggle selection functionality (passed to launch() for Gradio 6.0+)
TOGGLE_JS = """
function toggleSelection(index) {
    const item = document.querySelector(`[data-index="${index}"]`);
    if (!item) return;
    item.classList.toggle('selected');
    item.setAttribute('aria-pressed', item.classList.contains('selected'));

    const selected = [...document.querySelectorAll('.result-item.selected')]
        .map(el => parseInt(el.dataset.index));

    const input = document.querySelector('#selected-indices-input textarea, #selected-indices-input input');
    if (input) {
        input.value = JSON.stringify(selected);
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
        input.focus();
        input.blur();
    }

    const submitBtn = document.querySelector('#submit-btn button') || document.querySelector('#submit-btn');
    if (submitBtn) {
        submitBtn.textContent = `Submit ratings (${selected.length} selected)`;
    }

    const countLabel = document.getElementById('selection-count');
    if (countLabel) {
        const total = document.querySelectorAll('.result-item').length;
        countLabel.textContent = total ? `Selected: ${selected.length} of ${total}` : `Selected: ${selected.length}`;
    }
}
window.toggleSelection = toggleSelection;
"""

def create_app():
    """Create the Gradio app interface."""

    with gr.Blocks(title="Dressa - Dress Similarity Study", css=APP_CSS) as app:
        gr.HTML(FONT_LINKS_HTML)

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

            gr.HTML("""
            <div id="hero">
                <div class="hero-title">Dressa</div>
                <div class="hero-subtitle">Upload a dress photo and tap all results that look similar.</div>
                <div class="hero-steps">
                    <div class="step"><span class="step-num">1</span>Upload a clear dress photo</div>
                    <div class="step"><span class="step-num">2</span>Tap every similar dress</div>
                    <div class="step"><span class="step-num">3</span>Submit ratings and continue</div>
                </div>
            </div>
            """)

            # Progress tracker
            upload_progress = gr.Markdown("Uploads completed: 0 of 3-5 recommended", elem_id="upload-progress")

            # Main layout
            with gr.Row(elem_id="main-row"):
                # Left column: Upload
                with gr.Column(scale=1, elem_id="upload-col"):
                    gr.Markdown("### 1. Upload Your Dress")
                    upload_image = gr.Image(
                        label="Upload a dress photo",
                        type="numpy",
                        sources=["upload", "webcam"],
                        elem_id="upload-image"
                    )
                    search_btn = gr.Button("Find Similar Dresses", variant="primary")
                    status_text = gr.Markdown("", elem_id="status-text")

                    # Finish button (appears after minimum uploads)
                    finish_btn = gr.Button("Finish Study", variant="secondary", visible=False)

                # Right column: Results
                with gr.Column(scale=2, elem_id="results-col"):
                    gr.Markdown("### 2. Choose Similar Dresses")
                    progress_text = gr.Markdown("Upload a photo to begin.", elem_id="progress-text")

                    # Instructions for selection
                    selection_instructions = gr.Markdown(
                        "Tap all dresses that look similar to yours. Tap again to deselect. You can submit with zero selected.",
                        visible=False,
                        elem_id="selection-instructions"
                    )

                    selection_count = gr.Markdown("", visible=False, elem_id="selection-count")

                    # Custom HTML grid for toggle selection
                    results_grid_html = gr.HTML(value="", elem_id="results-grid-container")

                    # Hidden textbox for selected indices
                    selected_indices_input = gr.Textbox(
                        value="[]",
                        visible=False,
                        elem_id="selected-indices-input"
                    )

                    # Submit button
                    submit_btn = gr.Button(
                        "Submit ratings (0 selected)",
                        variant="primary",
                        size="lg",
                        visible=False,
                        interactive=True,
                        elem_id="submit-btn"
                    )

                    # Status message
                    submit_status = gr.Markdown("", elem_id="submit-status")

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
                <button class="result-item {selected_class}" data-index="{i}" onclick="toggleSelection({i})" aria-pressed="{str(i in selected_indices).lower()}" type="button">
                    <img src="{img_src}" alt="Dress {i+1}">
                    <span class="index-badge">{i+1}</span>
                </button>
                ''')

            html = f"""
            <div class="results-grid">
                {''.join(grid_items)}
            </div>
            """

            return html

        def on_search(image, user_id, upload_id, upload_count):
            """Handle search button click."""
            if image is None:
                return (
                    user_id, upload_id, [], [], [], upload_count,
                    "Please upload a dress photo first.",
                    "Upload a photo to begin.",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    "",
                    "[]",
                    gr.update(visible=False, interactive=False),
                    "",
                    f"Uploads completed: {upload_count} of 3-5 recommended",
                    gr.update(visible=upload_count >= MIN_UPLOADS_FOR_DEBRIEF)
                )

            # Save uploaded image
            filepath = save_uploaded_image(image, user_id)
            upload_id = db.create_upload(user_id, filepath)
            new_upload_count = upload_count + 1
            logger.info(f"New upload: {upload_id} (count: {new_upload_count})")

            # Search for similar dresses
            results = search_similar_dresses(image, user_id, upload_id)

            # Filter results to corpus-only images and build gallery
            filtered_results, gallery_images = filter_results_for_gallery(results)

            if not gallery_images:
                return (
                    user_id, upload_id, filtered_results, [], [], new_upload_count,
                    "Search complete, but no corpus images were found.",
                    "No results found. Try another photo.",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    "",
                    "[]",
                    gr.update(visible=False, interactive=False),
                    "",
                    f"Uploads completed: {new_upload_count} of 3-5 recommended",
                    gr.update(visible=new_upload_count >= MIN_UPLOADS_FOR_DEBRIEF)
                )

            progress_msg = f"Found **{len(gallery_images)}** similar dresses. Tap every item that matches."
            selection_text = f"Selected: 0 of {len(gallery_images)}"
            grid_html = generate_results_grid_html(gallery_images, [])

            return (
                user_id, upload_id, filtered_results, [], gallery_images, new_upload_count,
                "Search complete. Review the results and submit your ratings.",
                progress_msg,
                gr.update(visible=True),
                gr.update(visible=True, value=selection_text),
                grid_html,
                "[]",
                gr.update(visible=True, interactive=True, value="Submit ratings (0 selected)"),
                "",
                f"Uploads completed: {new_upload_count} of 3-5 recommended",
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
            btn_text = f"Submit ratings ({count} selected)"
            total = len(gallery_images) if gallery_images else 0
            count_text = f"Selected: {count} of {total}" if total else f"Selected: {count}"

            return (
                selected_indices,
                gr.update(value=btn_text, interactive=True),
                gr.update(value=count_text, visible=True)
            )

        def on_submit(user_id, upload_id, results, selected_indices_json, selected_indices_state, gallery_images):
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

            if selected_indices_state and len(selected_indices_state) >= len(selected_indices):
                selected_indices = selected_indices_state

            logger.info(f"Parsed selected indices: {selected_indices}")
            logger.info(f"Total results: {len(results) if results else 0}")

            if not results:
                logger.warning("No results to rate")
                return (
                    "No results to rate.",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    [],
                    "[]",
                    gr.update(visible=False),
                    "",
                    "Upload a photo to begin.",
                    "Ready for a new upload.",
                    gr.update(value=None)
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

            total_results = len(results) if results else 0
            status = (
                f"**Thank you!** You marked **{similar_count} of {total_results}** as similar "
                f"and **{not_similar_count}** as not similar. You can upload another dress now."
            )

            return (
                status,
                gr.update(visible=False),
                gr.update(visible=False, value=""),
                [],
                "[]",
                gr.update(visible=False),
                "",
                "Upload another photo to continue.",
                "Ready for another upload.",
                gr.update(value=None)
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
                selection_instructions, selection_count, results_grid_html,
                selected_indices_input, submit_btn, submit_status,
                upload_progress, finish_btn
            ]
        )

        # Selection change events
        selected_indices_input.change(
            fn=on_selection_change,
            inputs=[selected_indices_input, gallery_images_state],
            outputs=[selected_indices_state, submit_btn, selection_count]
        )

        # Submit events
        submit_btn.click(
            fn=on_submit,
            inputs=[
                user_id_state, upload_id_state, current_results_state,
                selected_indices_input, selected_indices_state, gallery_images_state
            ],
            outputs=[
                submit_status, selection_instructions, selection_count, selected_indices_state,
                selected_indices_input, submit_btn, results_grid_html,
                progress_text, status_text, upload_image
            ]
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
