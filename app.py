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
import json
import csv
import html
import uuid
from PIL import Image
from pathlib import Path
from datetime import datetime
import shutil
import logging
import os
import io
import time
import tempfile

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
UPLOADS_DIR = Path(
    os.getenv("DRESSA_UPLOADS_DIR", str(APP_DIR / "uploads"))
).resolve()
IMAGES_DIR = Path(
    os.getenv("DRESSA_IMAGES_DIR", str(APP_DIR / "dress_images"))
).resolve()

# Ensure uploads directory exists
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

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
ADMIN_PASSWORD = os.getenv("DRESSA_ADMIN_PASSWORD", "")
MIN_SUPPORT_FOR_BEST_MODEL = 30


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


def _resolve_upload_image_path(filepath: str) -> Path | None:
    """Resolve a query upload path from DB to an existing local file."""
    if not filepath:
        return None

    path = Path(filepath)
    if path.is_absolute() and path.exists():
        return path.resolve()

    candidates = [
        APP_DIR / filepath,
        UPLOADS_DIR / path.name,
        UPLOADS_DIR / filepath,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return None


def _image_to_data_url(image_path: Path, max_size: tuple[int, int] = (360, 520)) -> str | None:
    """Return a base64 data URL for an image path."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{img_base64}"
    except Exception as exc:
        logger.warning(f"Failed to encode image for admin panel: {image_path} ({exc})")
        return None


def _resolve_result_image_path(result_image_id: str) -> Path | None:
    """Resolve a corpus result image path stored in evaluation_ratings."""
    if not result_image_id:
        return None

    path = Path(result_image_id)
    if path.is_absolute() and path.exists():
        return path.resolve()

    full_path = get_image_full_path(result_image_id)
    if full_path.exists():
        return full_path.resolve()

    fallback = IMAGES_DIR / path.name
    if fallback.exists():
        return fallback.resolve()

    return None


def _render_admin_thumb(image_path: Path, alt_text: str) -> str:
    """Render one admin thumbnail image as HTML."""
    src = _image_to_data_url(image_path, max_size=(220, 300))
    if not src:
        return ""
    safe_alt = html.escape(alt_text)
    return f'<img class="admin-thumb" src="{src}" alt="{safe_alt}">'


def _render_admin_thumb_grid(image_paths: list[str], empty_text: str, is_query: bool = False) -> str:
    """Render a bounded thumbnail grid from image path strings."""
    thumbs = []
    for raw_path in image_paths:
        if is_query:
            resolved = _resolve_upload_image_path(raw_path)
        else:
            resolved = _resolve_result_image_path(raw_path)
        if not resolved:
            continue
        thumb = _render_admin_thumb(resolved, f"dress image {resolved.name}")
        if thumb:
            thumbs.append(thumb)

    if not thumbs:
        return f'<div class="admin-empty">{html.escape(empty_text)}</div>'

    grid_class = "admin-query-grid" if is_query else "admin-thumb-grid"
    return f'<div class="{grid_class}">{"".join(thumbs)}</div>'


def _render_visual_table_html(records: list[dict]) -> str:
    """Render visual comparison table (query, selected, not-selected)."""
    if not records:
        return '<div class="admin-empty">No uploads available for visual comparison.</div>'

    rows_html = []
    for rec in records:
        uploaded_at = html.escape(rec.get("uploaded_at", ""))
        user_id = html.escape(rec.get("user_id", ""))
        upload_id = html.escape(rec.get("upload_id", ""))
        total_results = rec.get("total_results", 0)
        similar_count = rec.get("similar_count", 0)
        not_similar_count = rec.get("not_similar_count", 0)

        query_html = _render_admin_thumb_grid(
            [rec.get("query_image_path", "")],
            empty_text="No query image",
            is_query=True,
        )
        selected_html = _render_admin_thumb_grid(
            rec.get("selected_image_paths", []),
            empty_text="No selected images",
            is_query=False,
        )
        not_selected_html = _render_admin_thumb_grid(
            rec.get("not_selected_image_paths", []),
            empty_text="No not-selected images",
            is_query=False,
        )

        rows_html.append(
            f"""
            <tr>
                <td><div class="admin-meta">{uploaded_at}</div></td>
                <td><div class="admin-meta">{user_id}</div></td>
                <td><div class="admin-meta">{upload_id}</div></td>
                <td><div class="admin-meta">{total_results}</div></td>
                <td><div class="admin-meta">{similar_count}</div></td>
                <td><div class="admin-meta">{not_similar_count}</div></td>
                <td>{query_html}</td>
                <td>{selected_html}</td>
                <td>{not_selected_html}</td>
            </tr>
            """
        )

    return f"""
    <div class="admin-visual-wrap">
      <table class="admin-visual-table">
        <thead>
          <tr>
            <th>uploaded_at</th>
            <th>user_id</th>
            <th>upload_id</th>
            <th>total_results</th>
            <th>selected_count</th>
            <th>not_selected_count</th>
            <th>query_image</th>
            <th>selected_images</th>
            <th>not_selected_images</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
    </div>
    """


def verify_admin_password(input_password: str) -> bool:
    """Validate admin password from Space secret."""
    if not ADMIN_PASSWORD:
        return False
    return input_password == ADMIN_PASSWORD


def load_model_leaderboard(min_support: int = MIN_SUPPORT_FOR_BEST_MODEL) -> tuple[str, list[list], str]:
    """
    Build model leaderboard from evaluation_ratings provenance.

    Returns:
        summary_markdown, leaderboard_rows, status_message
    """
    global db

    if db is None:
        return "### Best Model Right Now\nApp database is not initialized.", [], "Database is not initialized."

    with db._get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT rating, provenance FROM evaluation_ratings")
        rows = [dict(row) for row in cur.fetchall()]

    if not rows:
        summary = "### Best Model Right Now\nNo winner yet. No evaluation data has been collected."
        return summary, [], "No evaluation ratings found."

    stats: dict[str, dict[str, float]] = {}
    for row in rows:
        rating = row.get("rating")
        try:
            provenance = json.loads(row.get("provenance", "{}"))
        except Exception:
            continue
        if not isinstance(provenance, dict):
            continue

        for model_name in provenance.keys():
            if model_name not in stats:
                stats[model_name] = {
                    "total_recommendations": 0,
                    "similar_count": 0,
                }
            stats[model_name]["total_recommendations"] += 1
            if rating == "similar":
                stats[model_name]["similar_count"] += 1

    leaderboard_records = []
    for model_name, model_stats in stats.items():
        total = int(model_stats["total_recommendations"])
        similar = int(model_stats["similar_count"])
        hit_rate = (similar / total) if total > 0 else 0.0
        eligible = total >= min_support
        leaderboard_records.append({
            "model": model_name,
            "total_recommendations": total,
            "similar_count": similar,
            "hit_rate": hit_rate,
            "eligible": eligible,
        })

    leaderboard_records.sort(
        key=lambda rec: (
            rec["hit_rate"],
            rec["similar_count"],
            rec["total_recommendations"],
            rec["model"],
        ),
        reverse=True,
    )

    eligible_records = [rec for rec in leaderboard_records if rec["eligible"]]
    refreshed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if eligible_records:
        winner = eligible_records[0]
        summary = (
            "### Best Model Right Now\n"
            f"**{winner['model']}** | "
            f"Hit rate: **{winner['hit_rate']:.3f}** | "
            f"Similar: **{winner['similar_count']}** / **{winner['total_recommendations']}**\n\n"
            f"Refreshed: `{refreshed_at}`"
        )
    else:
        summary = (
            "### Best Model Right Now\n"
            f"No winner yet (need at least **{min_support}** recommendations per model).\n\n"
            f"Refreshed: `{refreshed_at}`"
        )

    leaderboard_rows = [
        [
            rec["model"],
            rec["total_recommendations"],
            rec["similar_count"],
            round(rec["hit_rate"], 3),
            "yes" if rec["eligible"] else "no",
        ]
        for rec in leaderboard_records
    ]

    status = f"Loaded leaderboard for {len(leaderboard_records)} models from {len(rows)} ratings."
    return summary, leaderboard_rows, status


def load_live_entries(max_uploads: int = 100) -> tuple[str, list[list], str]:
    """
    Build live per-upload analytics rows.

    Returns:
        status_message, summary_rows, visual_table_html
    """
    global db

    if db is None:
        return "Database is not initialized.", [], '<div class="admin-empty">Database unavailable.</div>'

    with db._get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT upload_id, user_id, filepath, uploaded_at
            FROM uploads
            ORDER BY uploaded_at DESC
            LIMIT ?
            """,
            (int(max_uploads),),
        )
        upload_rows = [dict(row) for row in cur.fetchall()]

        upload_ids = [row["upload_id"] for row in upload_rows]
        if upload_ids:
            placeholders = ",".join(["?"] * len(upload_ids))
            cur.execute(
                f"""
                SELECT upload_id, result_image_id, rating, display_position, timestamp
                FROM evaluation_ratings
                WHERE upload_id IN ({placeholders})
                ORDER BY upload_id, display_position ASC, timestamp ASC
                """,
                tuple(upload_ids),
            )
            rating_rows = [dict(row) for row in cur.fetchall()]
        else:
            rating_rows = []

    if not upload_rows:
        return "No uploads found yet.", [], '<div class="admin-empty">No uploads found yet.</div>'

    ratings_by_upload: dict[str, list[dict]] = {}
    for row in rating_rows:
        upload_id = row["upload_id"]
        if upload_id not in ratings_by_upload:
            ratings_by_upload[upload_id] = []
        ratings_by_upload[upload_id].append(row)

    summary_rows: list[list] = []
    visual_records: list[dict] = []

    for upload in upload_rows:
        upload_id = upload["upload_id"]
        rows = ratings_by_upload.get(upload_id, [])
        similar_rows = [r for r in rows if r.get("rating") == "similar"]
        not_similar_rows = [r for r in rows if r.get("rating") == "not_similar"]

        similar_image_paths = [r.get("result_image_id", "") for r in similar_rows if r.get("result_image_id")]
        not_selected_image_paths = [
            r.get("result_image_id", "")
            for r in not_similar_rows
            if r.get("result_image_id")
        ]

        total_results = len(rows)
        summary_rows.append(
            [
                upload.get("uploaded_at", ""),
                upload.get("user_id", ""),
                upload_id,
                total_results,
                len(similar_rows),
                len(not_similar_rows),
            ]
        )

        visual_records.append(
            {
                "uploaded_at": upload.get("uploaded_at", ""),
                "user_id": upload.get("user_id", ""),
                "upload_id": upload_id,
                "total_results": total_results,
                "similar_count": len(similar_rows),
                "not_similar_count": len(not_similar_rows),
                "query_image_path": upload.get("filepath", ""),
                "selected_image_paths": similar_image_paths,
                "not_selected_image_paths": not_selected_image_paths,
            }
        )

    visual_table_html = _render_visual_table_html(visual_records)
    status = f"Loaded {len(summary_rows)} uploads (default limit {int(max_uploads)})."
    return status, summary_rows, visual_table_html


def export_raw_ratings_csv() -> tuple[str | None, str]:
    """Export raw joined evaluation data to CSV and return file path."""
    global db

    if db is None:
        return None, "Database is not initialized."

    with db._get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                e.rating_id,
                e.user_id,
                e.upload_id,
                u.uploaded_at,
                u.filepath AS query_image_path,
                e.result_image_id,
                e.rating,
                e.display_position,
                e.provenance,
                e.timestamp
            FROM evaluation_ratings e
            LEFT JOIN uploads u ON e.upload_id = u.upload_id
            ORDER BY e.timestamp DESC
            """
        )
        rows = [dict(row) for row in cur.fetchall()]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(tempfile.gettempdir()) / f"dressa_raw_ratings_{timestamp}.csv"
    fieldnames = [
        "rating_id",
        "user_id",
        "upload_id",
        "uploaded_at",
        "query_image_path",
        "result_image_id",
        "rating",
        "display_position",
        "provenance",
        "timestamp",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return str(output_path), f"Exported {len(rows)} raw rows to CSV."


def _normalize_selected_indices(raw_indices, total_results: int) -> list[int]:
    """Parse, dedupe, and bounds-check selected index values."""
    if raw_indices is None:
        parsed = []
    elif isinstance(raw_indices, str):
        try:
            parsed = json.loads(raw_indices) if raw_indices else []
        except (json.JSONDecodeError, TypeError):
            parsed = []
    elif isinstance(raw_indices, list):
        parsed = raw_indices
    else:
        parsed = []

    normalized = []
    for idx in parsed:
        try:
            idx_int = int(idx)
        except (TypeError, ValueError):
            continue
        if 0 <= idx_int < total_results:
            normalized.append(idx_int)

    return sorted(set(normalized))


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
APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,600;700&display=swap');

:root {
    --bg: #fefdfb;
    --card: #ffffff;
    --ink: #191816;
    --muted: #6f6b65;
    --accent: #d46b3b;
    --accent-2: #236b5b;
    --accent-soft: #f6aa64;
    --glass: rgba(255, 255, 255, 0.78);
    --border: #e7ddd1;
    --shadow: 0 16px 44px rgba(24, 16, 8, 0.10);
    --display: 'Fraunces', 'Space Grotesk', 'Helvetica Neue', sans-serif;
    --body: 'Space Grotesk', 'Helvetica Neue', Helvetica, sans-serif;
}

body, .gradio-container {
    background:
        radial-gradient(circle at 10% -5%, rgba(246, 170, 100, 0.20), transparent 45%),
        radial-gradient(circle at 95% 10%, rgba(49, 121, 108, 0.13), transparent 40%),
        var(--bg);
    color: var(--ink);
    font-family: var(--body);
    overflow-x: hidden;
}

.gradio-container {
    width: 100% !important;
    max-width: none !important;
    margin: 0 !important;
    min-height: 100vh;
    padding: 12px clamp(16px, 3vw, 40px) 80px;
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
    display: grid !important;
    grid-template-columns: minmax(280px, 360px) minmax(0, 1fr);
    grid-template-areas:
        "upload results"
        "search results";
    gap: 26px;
    align-items: flex-start;
}

#upload-col {
    grid-area: upload;
}

#search-row {
    grid-area: search;
    margin: 0;
    justify-content: flex-start;
    align-self: start;
    width: 100%;
}

#search-row #search-btn {
    width: min(360px, 100%);
}

#results-col {
    grid-area: results;
}

#upload-col, #results-col {
    background: linear-gradient(155deg, rgba(255, 255, 255, 0.92), rgba(254, 249, 244, 0.88));
    border: 1px solid rgba(235, 219, 205, 0.95);
    border-radius: 24px;
    padding: 18px;
    box-shadow: 0 18px 42px rgba(24, 16, 8, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.7);
    overflow: visible !important;
    backdrop-filter: blur(3px);
}

#results-grid-container {
    overflow: visible !important;
    min-height: 120px;
}

#upload-helper {
    margin-top: 8px;
    color: var(--muted);
    font-size: 14px;
}

#main-app-screen {
    position: relative;
}

#hero {
    background: linear-gradient(135deg, #fff9f2, #ffeede);
    border: 1px solid var(--border);
    border-radius: 24px;
    box-shadow: 0 16px 44px rgba(24, 16, 8, 0.10), inset 0 1px 0 rgba(255, 255, 255, 0.8);
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
    background: var(--glass);
    border: 1px solid rgba(228, 214, 200, 0.9);
    border-radius: 12px;
    padding: 10px 12px;
    box-shadow: 0 8px 18px rgba(24, 16, 8, 0.05), inset 0 1px 0 rgba(255, 255, 255, 0.6);
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
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid rgba(232, 220, 207, 0.95);
    box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.35);
}

#upload-image .image-preview,
#upload-image .image-container {
    padding: 0 !important;
}

#upload-image .image-preview .toolbar,
#upload-image .image-preview .image-preview-controls,
#upload-image .image-preview .buttons {
    display: flex !important;
    gap: 8px !important;
}

#upload-image .image-preview button[aria-label="Fullscreen"],
#upload-image .image-preview button[aria-label="Zoom"],
#upload-image .image-preview button[aria-label="View"] {
    display: none !important;
}

#upload-image .image-preview button[aria-label="Clear"],
#upload-image .image-preview button[aria-label="Remove"] {
    display: inline-flex !important;
    width: 34px !important;
    height: 34px !important;
    min-width: 34px !important;
    border-radius: 999px !important;
    border: 1px solid rgba(255, 255, 255, 0.35) !important;
    background: rgba(20, 22, 24, 0.62) !important;
    color: #fff !important;
    box-shadow: 0 6px 14px rgba(0, 0, 0, 0.22) !important;
}

#upload-image .image-preview [role="tab"],
#upload-image .image-preview [role="button"] {
    border-radius: 999px !important;
}

.gr-markdown, .gr-markdown > div {
    overflow: visible !important;
}

#status-text, #progress-text, #submit-status {
    white-space: normal !important;
}

.results-grid {
    column-count: 2;
    column-gap: 12px;
    padding: 2px;
}

.result-item {
    position: relative;
    display: block;
    width: 100%;
    margin: 0 0 12px;
    break-inside: avoid;
    cursor: pointer;
    border-radius: 18px;
    overflow: hidden;
    border: 2px solid rgba(233, 225, 216, 0.95);
    transition: border-color 0.2s ease, transform 0.12s ease, box-shadow 0.2s ease;
    background: #f4f1ec;
    padding: 0;
    box-shadow: 0 11px 28px rgba(24, 16, 8, 0.10);
}

.result-item img {
    width: 100%;
    height: auto;
    min-height: 170px;
    max-height: 360px;
    object-fit: cover;
    display: block;
}

.result-item:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(24, 16, 8, 0.12);
}

.result-item.selected {
    border-color: rgba(246, 150, 47, 0.92);
    box-shadow: 0 0 0 3px rgba(246, 150, 47, 0.34), 0 14px 32px rgba(96, 51, 29, 0.24);
}

.result-item .select-chip,
.result-item .index-badge {
    display: none !important;
}

#submit-btn button,
#submit-btn {
    width: auto !important;
}

#submit-btn {
    position: sticky;
    bottom: 18px;
    z-index: 60;
    margin-top: 8px;
}

button#agree-btn,
#agree-btn button,
button#search-btn,
#search-btn button,
button#submit-btn,
#submit-btn button,
button#finish-btn,
#finish-btn button,
button#close-btn,
#close-btn button,
button#disagree-btn,
#disagree-btn button {
    min-height: 52px !important;
    border-radius: 999px !important;
    border: 1px solid rgba(255, 191, 126, 0.9) !important;
    background: linear-gradient(160deg, #ffc07a 0%, #ef8f26 52%, #e0801b 100%) !important;
    color: #fff !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 18px 26px rgba(224, 128, 27, 0.33), inset 0 1px 0 rgba(255, 255, 255, 0.35) !important;
}

button#disagree-btn,
#disagree-btn button {
    border-color: rgba(255, 255, 255, 0.72) !important;
    background: linear-gradient(155deg, rgba(255, 255, 255, 0.60), rgba(247, 247, 247, 0.36)) !important;
    color: rgba(27, 27, 30, 0.90) !important;
    box-shadow: 0 12px 24px rgba(72, 72, 78, 0.14), inset 0 1px 0 rgba(255, 255, 255, 0.75) !important;
    backdrop-filter: blur(16px) saturate(145%) !important;
    -webkit-backdrop-filter: blur(16px) saturate(145%) !important;
}

#disagree-btn,
#disagree-btn button,
#disagree-btn * {
    color: rgba(27, 27, 30, 0.90) !important;
}

#search-btn button[disabled],
#search-btn button:disabled,
#search-btn button[aria-disabled="true"] {
    border-color: rgba(188, 188, 188, 0.95) !important;
    background: linear-gradient(160deg, #dddddd 0%, #cccccc 52%, #bdbdbd 100%) !important;
    color: rgba(67, 67, 67, 0.92) !important;
    box-shadow: none !important;
    cursor: not-allowed !important;
    opacity: 1 !important;
}

.gr-button.primary,
.gradio-container button.primary {
    min-height: 52px !important;
    border-radius: 999px !important;
    border: 1px solid rgba(255, 191, 126, 0.9) !important;
    background: linear-gradient(160deg, #ffc07a 0%, #ef8f26 52%, #e0801b 100%) !important;
    color: #fff !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 18px 26px rgba(224, 128, 27, 0.33), inset 0 1px 0 rgba(255, 255, 255, 0.35) !important;
}

.gr-button.secondary,
.gradio-container button.secondary {
    min-height: 52px !important;
    border-radius: 999px !important;
    border: 1px solid rgba(205, 205, 205, 0.95) !important;
    background: linear-gradient(160deg, #efefef 0%, #dfdfdf 52%, #cecece 100%) !important;
    color: #232323 !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 12px 22px rgba(75, 75, 75, 0.18), inset 0 1px 0 rgba(255, 255, 255, 0.6) !important;
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

/* Hide participant-side Gradio status trackers/icons; admin remains unchanged */
#main-app-screen [data-testid="status-tracker"] .wrap {
    display: none !important;
}

/* Participant-only top loading bar */
#dressa-top-loader {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 5px;
    display: none;
    z-index: 10050;
    pointer-events: none;
    background: rgba(212, 107, 59, 0.08);
}

#dressa-top-loader.active {
    display: block;
}

#dressa-top-loader .top-loader-track {
    position: relative;
    width: 100%;
    height: 100%;
    overflow: hidden;
    background: linear-gradient(90deg, rgba(212, 107, 59, 0.1), rgba(35, 107, 91, 0.12));
}

#dressa-top-loader .top-loader-bar {
    position: absolute;
    top: 0;
    left: -42%;
    width: 42%;
    height: 100%;
    background: linear-gradient(90deg, transparent, var(--accent), var(--accent-2), transparent);
    animation: dressa-top-loader-slide 1.05s ease-in-out infinite;
    box-shadow: 0 0 12px rgba(212, 107, 59, 0.35);
}

@keyframes dressa-top-loader-slide {
    0% {
        left: -42%;
    }
    100% {
        left: 100%;
    }
}

#search-overlay {
    position: fixed;
    inset: 0;
    z-index: 10040;
    display: none;
    align-items: center;
    justify-content: center;
    pointer-events: none;
    background: rgba(16, 16, 17, 0.34);
    backdrop-filter: blur(3px);
}

body.dressa-searching #search-overlay {
    display: flex;
}

body.dressa-searching #search-row {
    display: none !important;
}

body.dressa-has-results #search-row {
    display: none !important;
}

#search-overlay .search-overlay-card {
    background: linear-gradient(170deg, rgba(255, 255, 255, 0.95), rgba(255, 243, 231, 0.9));
    border: 1px solid rgba(255, 230, 206, 0.95);
    border-radius: 20px;
    padding: 20px 22px;
    box-shadow: 0 26px 48px rgba(18, 10, 4, 0.2);
    text-align: center;
    min-width: 250px;
}

#search-overlay .spinner {
    width: 38px;
    height: 38px;
    margin: 0 auto 12px;
    border-radius: 50%;
    border: 3px solid rgba(212, 107, 59, 0.24);
    border-top-color: #e68929;
    animation: spin-loader 0.8s linear infinite;
}

#search-overlay .search-title {
    font-family: var(--display);
    font-size: 21px;
    margin-bottom: 4px;
}

#search-overlay .search-copy {
    color: var(--muted);
    font-size: 14px;
}

@keyframes spin-loader {
    to {
        transform: rotate(360deg);
    }
}

@media (min-width: 860px) {
    .results-grid {
        column-count: 3;
    }
}

@media (min-width: 1200px) {
    .results-grid {
        column-count: 4;
    }
}

@media (min-width: 1025px) {
    #consent-actions {
        width: 100% !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 12px !important;
    }

    #agree-btn,
    #disagree-btn,
    #close-btn,
    #finish-btn,
    #search-btn,
    #submit-btn {
        width: auto !important;
        max-width: max-content !important;
        align-self: center !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    #agree-btn button,
    #disagree-btn button,
    #close-btn button,
    #finish-btn button,
    #search-btn button,
    #submit-btn button,
    #submit-btn > button {
        width: auto !important;
        max-width: max-content !important;
        min-width: 260px !important;
        padding: 0 34px !important;
        white-space: nowrap !important;
        display: inline-flex !important;
        justify-content: center !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    #submit-btn {
        margin-left: auto !important;
        margin-right: auto !important;
    }

    #search-row {
        justify-content: center !important;
    }

    #agree-btn,
    #disagree-btn {
        width: min(420px, 100%) !important;
        max-width: 420px !important;
    }

    #agree-btn button,
    #disagree-btn button {
        width: 100% !important;
        max-width: 100% !important;
    }
}

@media (max-width: 1024px) {
    .gradio-container.fill_width {
        padding: 0 !important;
    }

    .main.fillable.app.fill_width,
    .main.fillable.svelte-99kmwu.app.fill_width {
        --size-8: 0px !important;
        padding: 0 !important;
    }

    .main.svelte-99kmwu {
        --size-8: 0px !important;
        padding: 0 !important;
    }

    .gradio-container .contain,
    .gradio-container .container,
    .gradio-container .wrap {
        margin: 0 !important;
        padding: 0 !important;
        width: 100% !important;
    }

    #consent-info-row {
        flex-direction: column !important;
        gap: 12px !important;
    }

    #consent-screen {
        padding-bottom: 120px !important;
    }

    #consent-screen h1 {
        margin-bottom: 6px !important;
    }

    #consent-screen h2 {
        margin-top: 10px !important;
        margin-bottom: 8px !important;
    }

    #consent-screen .gr-markdown p,
    #consent-screen .gr-markdown ul {
        margin-bottom: 8px !important;
    }

    #consent-actions {
        position: sticky;
        bottom: max(12px, env(safe-area-inset-bottom));
        padding: 6px 8px 0;
        gap: 10px;
        z-index: 20;
    }
}

#upload-progress-mobile {
    display: none;
}

#mobile-instructions {
    display: none;
}

#mobile-bottom-nav {
    display: none;
}

.selection-badge {
    position: absolute;
    top: 8px;
    right: 8px;
    width: 24px;
    height: 24px;
    border-radius: 999px;
    background: linear-gradient(155deg, #ffb14f, #f07f21);
    color: #fff !important;
    display: none;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 12px;
    line-height: 1;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.35);
    box-shadow: 0 5px 12px rgba(150, 82, 25, 0.35);
    z-index: 8;
}

.selection-badge.visible {
    display: flex;
}

@keyframes dressaGridFade {
    from {
        opacity: 0;
        transform: translateY(8px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@media (max-width: 1024px) {
    :root {
        --mobile-max-width: 390px;
    }

    body,
    .gradio-container {
        background:
            radial-gradient(circle at 84% 18%, rgba(255, 255, 255, 0.42), rgba(255, 255, 255, 0) 46%),
            linear-gradient(180deg, #ffd8bb 0%, #f7b18d 48%, #f3a37b 100%) !important;
        color: #2b2b2b !important;
    }

    .gradio-container {
        padding: 0 0 calc(96px + env(safe-area-inset-bottom)) !important;
    }

    .gradio-container .contain,
    .gradio-container .container,
    .gradio-container .wrap,
    .gradio-container .gr-row,
    .gradio-container .gr-column {
        overflow: visible !important;
    }

    #main-app-screen,
    #main-row {
        overflow: visible !important;
    }

    #main-app-screen {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    #hero,
    #upload-progress,
    #upload-col h3,
    #upload-helper,
    #status-text,
    #results-col h3,
    #progress-text,
    #selection-instructions,
    #selection-count,
    #submit-status,
    #admin-gate-screen,
    #admin-panel-screen {
        display: none !important;
    }

    #main-row {
        display: grid !important;
        grid-template-columns: minmax(0, 1fr) !important;
        grid-template-areas:
            "upload"
            "search"
            "results";
        gap: 4px !important;
        align-items: stretch !important;
    }

    #upload-col {
        position: sticky !important;
        top: max(0px, env(safe-area-inset-top)) !important;
        z-index: 7200 !important;
        align-self: start !important;
        height: fit-content !important;
        margin: 0 auto 4px !important;
        width: min(352px, calc(100vw - 24px)) !important;
        max-width: 352px !important;
        left: auto !important;
        right: auto !important;
        transform: none !important;
        box-sizing: border-box;
        background:
            radial-gradient(circle at 20% 6%, rgba(255, 255, 255, 0.42), rgba(255, 255, 255, 0) 52%),
            linear-gradient(150deg, rgba(255, 243, 231, 0.78), rgba(248, 193, 149, 0.66));
        border: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 22px;
        padding: 12px;
        box-shadow: 0 12px 28px rgba(88, 47, 25, 0.16), inset 0 1px 0 rgba(255, 255, 255, 0.58);
        backdrop-filter: blur(14px) saturate(135%);
        -webkit-backdrop-filter: blur(14px) saturate(135%);
        display: grid;
        grid-template-columns: 120px minmax(0, 1fr);
        grid-template-rows: auto auto;
        column-gap: 10px;
        row-gap: 8px;
        align-items: start;
    }

    #upload-image {
        grid-column: 1;
        grid-row: 1 / span 2;
        width: 120px !important;
        min-width: 120px !important;
        max-width: 120px !important;
        margin: 0 !important;
        padding: 0 !important;
        border: 0 !important;
        border-style: none !important;
        border-width: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
        overflow: visible !important;
        --block-border-width: 0px !important;
        --input-border-width: 0px !important;
        --block-background-fill: transparent !important;
        --block-background-fill-dark: transparent !important;
    }

    #upload-image::before,
    #upload-image::after {
        display: none !important;
        border: 0 !important;
        box-shadow: none !important;
        background: transparent !important;
    }

    #upload-image .image-container {
        width: 120px !important;
        min-height: 0 !important;
        height: auto !important;
        margin: 0 !important;
        border-radius: 22px !important;
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        overflow: visible !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
        position: relative !important;
        color: transparent !important;
        display: block !important;
        font-size: 0 !important;
        line-height: 0 !important;
    }

    #upload-image .image-container::before {
        display: none !important;
        content: none !important;
    }

    #upload-image .image-container::after {
        display: none !important;
        content: none !important;
    }

    #upload-image .upload-container {
        width: 120px !important;
        min-height: 96px !important;
        margin: 0 !important;
        padding: 0 !important;
        border: 0 !important;
        background: transparent !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    #upload-image .upload-container > button,
    #upload-image .upload-container > button:hover,
    #upload-image .upload-container > button:focus,
    #upload-image .upload-container > button:focus-visible {
        width: 120px !important;
        min-width: 120px !important;
        min-height: 96px !important;
        margin: 0 !important;
        padding: 0 !important;
        border: 0 !important;
        outline: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
        border-radius: 22px !important;
    }

    #upload-image .upload-container > button::before,
    #upload-image .upload-container > button::after {
        display: none !important;
        border: 0 !important;
    }

    #upload-image .upload-container > button .wrap {
        width: 96px !important;
        min-height: 96px !important;
        margin: 0 !important;
        padding: 0 !important;
        border: 1px solid rgba(246, 246, 246, 0.9) !important;
        outline: 0 !important;
        background: linear-gradient(160deg, rgba(234, 234, 234, 0.86), rgba(209, 209, 209, 0.76)) !important;
        box-shadow: 0 10px 18px rgba(88, 47, 25, 0.16), inset 0 1px 0 rgba(255, 255, 255, 0.72) !important;
        border-radius: 22px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        position: relative !important;
        z-index: 1;
    }

    #upload-image .upload-container > button .wrap::before {
        content: "";
        position: absolute;
        width: 34px;
        height: 34px;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        background-repeat: no-repeat;
        background-position: center;
        background-size: 100% 100%;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23506a86' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 15V5'/%3E%3Cpath d='m7 10 5-5 5 5'/%3E%3Cpath d='M5 16v2a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2'/%3E%3C/svg%3E");
        opacity: 0.92;
        filter: drop-shadow(0 1px 0 rgba(255, 255, 255, 0.48)) drop-shadow(0 6px 10px rgba(80, 106, 134, 0.16));
        pointer-events: none;
        z-index: 1;
    }

    #upload-image .image-preview {
        width: 120px !important;
        max-width: 120px !important;
        height: auto !important;
        max-height: none !important;
        margin: 0 !important;
        border-radius: 22px !important;
        background: linear-gradient(160deg, rgba(238, 238, 238, 0.86), rgba(217, 217, 217, 0.78)) !important;
        border: 1px solid rgba(246, 246, 246, 0.9) !important;
        box-shadow: 0 12px 24px rgba(88, 47, 25, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.72) !important;
        overflow: hidden !important;
        position: relative !important;
        display: block !important;
        aspect-ratio: auto !important;
    }

    #upload-image .image-container:has(img),
    #upload-image .image-container:has(canvas),
    #upload-image .image-preview {
        border-radius: 22px !important;
        border: 1px solid rgba(246, 246, 246, 0.9) !important;
        background: linear-gradient(160deg, rgba(238, 238, 238, 0.86), rgba(217, 217, 217, 0.78)) !important;
        box-shadow: 0 12px 24px rgba(88, 47, 25, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.72) !important;
        overflow: hidden !important;
    }

    #upload-image label,
    #upload-image .label {
        display: none !important;
    }

    #upload-image .image-container [class*="text"],
    #upload-image .image-container [class*="label"],
    #upload-image .image-container p,
    #upload-image .image-container span {
        display: none !important;
    }

    #upload-image .image-container p,
    #upload-image .image-container span {
        color: transparent !important;
    }

    #upload-image .image-container svg {
        width: 38px !important;
        height: 38px !important;
        color: rgba(96, 79, 66, 0.9) !important;
    }

    #upload-image .image-preview .toolbar,
    #upload-image .image-preview .image-preview-controls,
    #upload-image .image-preview .buttons {
        position: absolute !important;
        top: 6px !important;
        right: 6px !important;
        z-index: 4 !important;
        margin: 0 !important;
        padding: 0 !important;
        gap: 0 !important;
    }

    #upload-image .image-preview button[aria-label="Fullscreen"],
    #upload-image .image-preview button[aria-label="Zoom"],
    #upload-image .image-preview button[aria-label="View"] {
        display: none !important;
    }

    #upload-image .image-preview img {
        width: 100% !important;
        max-width: 100% !important;
        height: auto !important;
        max-height: none !important;
        object-fit: contain !important;
        display: block !important;
        margin: 0 !important;
        border-radius: 22px !important;
        box-shadow: 0 8px 16px rgba(88, 47, 25, 0.18) !important;
    }

    #upload-image .image-container img,
    #upload-image .image-container canvas {
        width: 100% !important;
        max-width: 100% !important;
        height: auto !important;
        display: block !important;
        border-radius: 22px !important;
        box-shadow: 0 8px 16px rgba(88, 47, 25, 0.18) !important;
    }

    #upload-image .image-preview button[aria-label="Clear"],
    #upload-image .image-preview button[aria-label="Remove"] {
        width: 26px !important;
        height: 26px !important;
        min-width: 26px !important;
        border-radius: 999px !important;
        background: rgba(18, 18, 19, 0.74) !important;
        border: 1px solid rgba(255, 255, 255, 0.55) !important;
        color: #fff !important;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.26) !important;
        position: relative !important;
    }

    #upload-image [role="tablist"] {
        display: none !important;
    }

    #mobile-instructions {
        grid-column: 2;
        grid-row: 1;
        display: flex;
        flex-direction: column;
        gap: 6px;
        margin-top: 2px !important;
    }

    #mobile-instructions .mobile-step {
        background: rgba(255, 255, 255, 0.74);
        border: 1px solid rgba(255, 255, 255, 0.72);
        border-radius: 20px;
        min-height: 36px;
        padding: 6px 10px;
        box-shadow: 0 8px 18px rgba(92, 48, 25, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.65);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        display: flex;
        align-items: flex-start;
        gap: 6px;
        font-size: 12px;
        line-height: 1.3;
        color: #2b2b2b;
    }

    #mobile-instructions .mobile-step-num {
        width: 19px;
        height: 19px;
        min-width: 19px;
        flex: 0 0 19px;
        aspect-ratio: 1 / 1;
        border-radius: 50%;
        background: linear-gradient(155deg, #ffac4e, #f17f21);
        color: #fff;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 10px;
        line-height: 1;
        box-shadow: 0 4px 10px rgba(179, 93, 25, 0.24);
    }

    #mobile-instructions .mobile-step > span:last-child {
        white-space: normal;
        overflow: visible;
        text-overflow: initial;
    }

    #search-btn {
        width: 100%;
        margin: 0;
        display: flex;
        justify-content: center;
    }

    #search-row {
        width: min(352px, calc(100vw - 24px));
        margin: 2px auto 4px !important;
        justify-content: center !important;
    }

    #search-btn button {
        width: 100% !important;
        min-width: 100% !important;
        max-width: none !important;
        min-height: 52px !important;
        border-radius: 999px !important;
        border: 1px solid rgba(255, 212, 166, 0.88) !important;
        padding: 0 30px !important;
        background: linear-gradient(
            150deg,
            rgba(255, 190, 120, 0.80),
            rgba(238, 137, 34, 0.86) 55%,
            rgba(224, 124, 24, 0.82)
        ) !important;
        color: #fff !important;
        font-weight: 700 !important;
        letter-spacing: 0.01em !important;
        box-shadow: 0 20px 30px rgba(224, 128, 27, 0.30), inset 0 1px 0 rgba(255, 255, 255, 0.38) !important;
        backdrop-filter: blur(10px) saturate(145%) !important;
        -webkit-backdrop-filter: blur(10px) saturate(145%) !important;
        position: relative;
        overflow: hidden;
    }

    #search-btn button::before {
        content: "";
        position: absolute;
        inset: 1px;
        border-radius: inherit;
        background: linear-gradient(
            120deg,
            rgba(255, 255, 255, 0.34),
            rgba(255, 255, 255, 0.10) 38%,
            rgba(255, 255, 255, 0.24)
        );
        pointer-events: none;
    }

    #search-btn button::after {
        content: "";
        position: absolute;
        left: 12%;
        right: 12%;
        bottom: 8px;
        height: 16px;
        border-radius: 999px;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.26), rgba(255, 255, 255, 0));
        filter: blur(6px);
        pointer-events: none;
    }

    #upload-progress-mobile {
        grid-column: 2;
        grid-row: 2;
        margin: 2px 0 0 0 !important;
        padding: 6px 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.72) !important;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.74) !important;
        box-shadow: 0 8px 18px rgba(92, 48, 25, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.65) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
        display: flex !important;
        align-items: center !important;
        font-size: 12px;
        color: #2b2b2b !important;
        font-weight: 600;
        line-height: 1.3;
        white-space: normal !important;
    }

    #results-col {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 auto !important;
        width: min(var(--mobile-max-width), calc(100vw - 8px)) !important;
        max-width: var(--mobile-max-width);
        position: relative !important;
        overflow: visible !important;
    }

    #results-grid-stage {
        width: 100%;
    }

    #results-grid-container {
        position: static !important;
        left: auto !important;
        transform: none !important;
        width: 100% !important;
        margin: 0 !important;
        box-sizing: border-box;
        padding: 0 6px calc(146px + env(safe-area-inset-bottom));
    }

    .results-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
        padding: 0;
        column-count: unset;
        animation: dressaGridFade 0.35s ease;
    }

    .result-item {
        border: none !important;
        border-radius: 22px !important;
        box-shadow: 0 8px 20px rgba(97, 51, 29, 0.16) !important;
        background: #fff !important;
        margin: 0 !important;
    }

    .result-item.selected {
        box-shadow: 0 0 0 2px rgba(246, 150, 47, 0.68), 0 10px 22px rgba(96, 51, 29, 0.2) !important;
    }

    .result-item .select-chip,
    .result-item .index-badge {
        display: none !important;
    }

    .selection-badge {
        width: 24px;
        height: 24px;
        border-radius: 999px;
        background: linear-gradient(155deg, #ffb14f, #f07f21);
        color: #fff !important;
        font-weight: 700 !important;
        line-height: 1 !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.35);
        box-shadow: 0 5px 12px rgba(150, 82, 25, 0.35);
        font-size: 12px;
    }

    .selection-badge.visible {
        display: flex;
    }

    #submit-btn {
        position: fixed !important;
        left: 50% !important;
        right: auto !important;
        top: auto !important;
        transform: translateX(-50%) !important;
        bottom: calc(62px + env(safe-area-inset-bottom)) !important;
        width: auto !important;
        margin: 0 !important;
        text-align: center !important;
        z-index: 6505 !important;
        pointer-events: none !important;
    }

    button#submit-btn,
    #submit-btn button,
    #submit-btn > button {
        position: relative !important;
        overflow: hidden;
        display: flex !important;
        align-items: center;
        justify-content: center;
        margin: 0 auto !important;
        white-space: nowrap;
        min-height: 52px !important;
        border-radius: 999px !important;
        width: auto !important;
        min-width: 164px !important;
        max-width: calc(100vw - 24px) !important;
        padding: 0 30px !important;
        border: 1px solid rgba(255, 212, 166, 0.88) !important;
        background: linear-gradient(
            150deg,
            rgba(255, 190, 120, 0.80),
            rgba(238, 137, 34, 0.86) 55%,
            rgba(224, 124, 24, 0.82)
        ) !important;
        backdrop-filter: blur(10px) saturate(145%) !important;
        -webkit-backdrop-filter: blur(10px) saturate(145%) !important;
        box-shadow: 0 20px 30px rgba(224, 128, 27, 0.30), inset 0 1px 0 rgba(255, 255, 255, 0.38) !important;
        pointer-events: auto !important;
        z-index: 6506 !important;
    }

    button#submit-btn::before,
    #submit-btn button::before {
        content: "";
        position: absolute;
        inset: 1px;
        border-radius: inherit;
        background: linear-gradient(
            120deg,
            rgba(255, 255, 255, 0.34),
            rgba(255, 255, 255, 0.10) 38%,
            rgba(255, 255, 255, 0.24)
        );
        pointer-events: none;
    }

    button#submit-btn::after,
    #submit-btn button::after {
        content: "";
        position: absolute;
        left: 12%;
        right: 12%;
        bottom: 8px;
        height: 16px;
        border-radius: 999px;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.26), rgba(255, 255, 255, 0));
        filter: blur(6px);
        pointer-events: none;
    }

    #mobile-bottom-nav,
    button#mobile-bottom-nav {
        display: block !important;
        position: fixed;
        left: 50%;
        transform: translateX(-50%);
        bottom: 0;
        width: 100vw;
        padding: 0 0 env(safe-area-inset-bottom);
        min-height: 52px;
        border-radius: 22px 22px 0 0 !important;
        background: rgba(255, 255, 255, 0.95) !important;
        border: none !important;
        box-shadow: 0 -8px 18px rgba(95, 54, 28, 0.16) !important;
        color: #e46339 !important;
        font-weight: 700 !important;
        font-size: 18px !important;
        text-align: center !important;
        z-index: 6400;
    }

    #mobile-bottom-nav button {
        display: block !important;
        width: 100% !important;
        min-height: 52px !important;
        border-radius: 22px 22px 0 0 !important;
        background: transparent !important;
        color: inherit !important;
        font-weight: inherit !important;
        font-size: inherit !important;
        border: none !important;
        box-shadow: none !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
    }

    #mobile-bottom-nav button::before,
    #mobile-bottom-nav button::after,
    button#mobile-bottom-nav::before,
    button#mobile-bottom-nav::after {
        display: none !important;
    }
}

.admin-visual-wrap {
    overflow-x: auto;
    border: 1px solid var(--border);
    border-radius: 12px;
    background: #fff;
}

.admin-visual-table {
    width: 100%;
    border-collapse: collapse;
    min-width: 1200px;
}

.admin-visual-table th,
.admin-visual-table td {
    border-bottom: 1px solid var(--border);
    border-right: 1px solid var(--border);
    padding: 10px;
    vertical-align: top;
}

.admin-visual-table th:last-child,
.admin-visual-table td:last-child {
    border-right: none;
}

.admin-visual-table th {
    background: #faf5ef;
    font-weight: 700;
    text-align: left;
}

.admin-meta {
    font-size: 12px;
    line-height: 1.4;
    white-space: nowrap;
}

.admin-thumb-grid,
.admin-query-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));
    gap: 8px;
    max-height: 280px;
    overflow-y: auto;
    min-width: 250px;
}

.admin-query-grid {
    min-width: 180px;
    max-width: 220px;
}

.admin-thumb {
    width: 100%;
    max-width: 180px;
    aspect-ratio: 3/4;
    object-fit: cover;
    border-radius: 10px;
    border: 1px solid var(--border);
    background: #f5f2ed;
}

.admin-empty {
    color: var(--muted);
    font-size: 12px;
}

/* Keep selected-indices textbox mounted for JS->Gradio sync, but visually hidden */
#selected-indices-input {
    display: block !important;
    height: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    overflow: hidden !important;
}

#selected-indices-input textarea,
#selected-indices-input input {
    opacity: 0 !important;
    height: 1px !important;
    min-height: 1px !important;
    pointer-events: none !important;
    padding: 0 !important;
    margin: 0 !important;
    border: 0 !important;
}
"""

# JavaScript for toggle selection functionality (passed to launch() for Gradio 6.0+)
TOGGLE_JS = """
window.__dressaSelectionOrder = window.__dressaSelectionOrder || [];

function isCompactLayout() {
    return window.matchMedia('(max-width: 1024px), (hover: none) and (pointer: coarse)').matches;
}

function getResultItemIndex(item) {
    const value = Number.parseInt(item?.dataset?.index ?? '', 10);
    return Number.isInteger(value) ? value : null;
}

function reconcileSelectionOrder() {
    const selectedItems = [...document.querySelectorAll('.result-item.selected')];
    const selectedIndices = selectedItems
        .map(getResultItemIndex)
        .filter((value) => value !== null);
    const selectedSet = new Set(selectedIndices);
    const current = (window.__dressaSelectionOrder || [])
        .map((value) => Number.parseInt(value, 10))
        .filter((value) => Number.isInteger(value) && selectedSet.has(value));

    for (const idx of selectedIndices) {
        if (!current.includes(idx)) {
            current.push(idx);
        }
    }

    window.__dressaSelectionOrder = current;
    return current;
}

function updateSelectionBadges() {
    const order = reconcileSelectionOrder();
    document.querySelectorAll('.result-item').forEach(item => {
        const badge = item.querySelector('.selection-badge');
        if (!badge) return;
        const idx = getResultItemIndex(item);
        if (idx === null) {
            badge.textContent = '';
            badge.classList.remove('visible');
            return;
        }
        const position = order.indexOf(idx);
        if (position === -1) {
            badge.textContent = '';
            badge.classList.remove('visible');
        } else {
            badge.textContent = String(position + 1);
            badge.classList.add('visible');
        }
    });
}

function resetSelectionOrder() {
    window.__dressaSelectionOrder = [];
    updateSelectionBadges();
}

window.__dressa_reset_selection_order = resetSelectionOrder;

function dockMobileSubmitButton() {
    const submitHost = document.getElementById('submit-btn');
    if (!submitHost) return;

    const submitBtn = submitHost.matches('button')
        ? submitHost
        : (submitHost.querySelector('button') || submitHost);
    if (!submitBtn) return;

    if (isCompactLayout()) {
        if (submitBtn.dataset.mobileDocked !== '1') {
            const dockBottom = 'calc(62px + env(safe-area-inset-bottom))';
            submitBtn.style.setProperty('position', 'fixed', 'important');
            submitBtn.style.setProperty('top', 'auto', 'important');
            submitBtn.style.setProperty('bottom', dockBottom, 'important');
            submitBtn.style.setProperty('left', '50%', 'important');
            submitBtn.style.setProperty('right', 'auto', 'important');
            submitBtn.style.setProperty('transform', 'translateX(-50%)', 'important');
            submitBtn.style.setProperty('z-index', '6501', 'important');
            submitBtn.style.setProperty('pointer-events', 'auto', 'important');
            submitBtn.dataset.mobileDocked = '1';
        }
    } else {
        if (submitBtn.dataset.mobileDocked === '1') {
            submitBtn.style.removeProperty('position');
            submitBtn.style.removeProperty('top');
            submitBtn.style.removeProperty('bottom');
            submitBtn.style.removeProperty('left');
            submitBtn.style.removeProperty('right');
            submitBtn.style.removeProperty('transform');
            submitBtn.style.removeProperty('z-index');
            submitBtn.style.removeProperty('pointer-events');
            delete submitBtn.dataset.mobileDocked;
        }
    }
}

window.__dressa_dock_mobile_submit = dockMobileSubmitButton;

function syncMobileLabels() {
    const resultCount = document.querySelectorAll('.result-item').length;
    document.body.classList.toggle('dressa-has-results', resultCount > 0);
    if (resultCount === 0) {
        window.__dressaSelectionOrder = [];
    }
    const submitHost = document.getElementById('submit-btn');
    if (submitHost) {
        if (resultCount === 0) {
            submitHost.style.setProperty('display', 'none', 'important');
        } else {
            submitHost.style.removeProperty('display');
        }
    }
    updateSelectionBadges();

    if (!isCompactLayout()) {
        dockMobileSubmitButton();
        return;
    }
    const searchBtn = document.querySelector('#search-btn button') || document.querySelector('#search-btn');
    if (searchBtn) {
        searchBtn.textContent = 'Search';
    }
    const submitBtn = document.querySelector('#submit-btn button') || document.querySelector('#submit-btn');
    if (submitBtn) {
        const selectedCount = getSelectedIndices().length;
        submitBtn.textContent = `Submit (${selectedCount})`;
        submitBtn.disabled = selectedCount === 0 || resultCount === 0;
        submitBtn.setAttribute('aria-disabled', String(submitBtn.disabled));
    }
    dockMobileSubmitButton();
}

window.__dressa_sync_mobile_labels = syncMobileLabels;

function getSelectedIndices() {
    return [...document.querySelectorAll('.result-item.selected')]
        .map(el => Number.parseInt(el.dataset.index, 10))
        .filter(Number.isInteger)
        .sort((a, b) => a - b);
}

function syncSelectedIndicesToInput() {
    const selected = getSelectedIndices();
    let input = document.querySelector('#selected-indices-input textarea, #selected-indices-input input');
    if (!input) {
        const maybeInput = document.getElementById('selected-indices-input');
        if (maybeInput && (maybeInput.tagName === 'INPUT' || maybeInput.tagName === 'TEXTAREA')) {
            input = maybeInput;
        }
    }
    if (input) {
        const jsonValue = JSON.stringify(selected);
        input.value = jsonValue;
        input.setAttribute('value', jsonValue);
        input.dispatchEvent(new Event('input', { bubbles: true, cancelable: true }));
        input.dispatchEvent(new Event('change', { bubbles: true, cancelable: true }));
    } else {
        console.warn('selected-indices-input not found in DOM; submit may lose selections');
    }
    return selected;
}

function updateSelectionUi(selected) {
    const resultCount = document.querySelectorAll('.result-item').length;
    document.body.classList.toggle('dressa-has-results', resultCount > 0);
    if (resultCount === 0) {
        window.__dressaSelectionOrder = [];
    }
    const submitHost = document.getElementById('submit-btn');
    if (submitHost) {
        if (resultCount === 0) {
            submitHost.style.setProperty('display', 'none', 'important');
        } else {
            submitHost.style.removeProperty('display');
        }
    }
    updateSelectionBadges();

    const submitBtn = document.querySelector('#submit-btn button') || document.querySelector('#submit-btn');
    if (submitBtn) {
        if (isCompactLayout()) {
            submitBtn.textContent = `Submit (${selected.length})`;
        } else {
            submitBtn.textContent = `Submit Similar Selections (${selected.length})`;
        }
        submitBtn.disabled = selected.length === 0 || resultCount === 0;
        submitBtn.setAttribute('aria-disabled', String(submitBtn.disabled));
    }
    dockMobileSubmitButton();

    const countLabel = document.getElementById('selection-count');
    if (countLabel) {
        const total = document.querySelectorAll('.result-item').length;
        countLabel.textContent = total ? `Selected: ${selected.length} of ${total}` : `Selected: ${selected.length}`;
    }
}

function toggleSelection(index) {
    const numericIndex = Number.parseInt(index, 10);
    if (!Number.isInteger(numericIndex)) return;
    const item = document.querySelector(`[data-index="${numericIndex}"]`);
    if (!item) return;
    const isSelected = item.classList.toggle('selected');
    item.setAttribute('aria-pressed', isSelected);

    const order = reconcileSelectionOrder();
    const existingIndex = order.indexOf(numericIndex);
    if (isSelected && existingIndex === -1) {
        order.push(numericIndex);
    } else if (!isSelected && existingIndex !== -1) {
        order.splice(existingIndex, 1);
    }
    window.__dressaSelectionOrder = order;

    const selected = syncSelectedIndicesToInput();
    updateSelectionBadges();
    updateSelectionUi(selected);
    console.log('Selection updated:', selected);
}

window.toggleSelection = toggleSelection;
window.syncSelectedIndicesToInput = syncSelectedIndicesToInput;

function attachResultsObserver() {
    if (window.__dressaResultsObserverAttached) return;
    const container = document.getElementById('results-grid-container');
    if (!container) {
        setTimeout(attachResultsObserver, 400);
        return;
    }
    let lastResultItemCount = -1;
    const observer = new MutationObserver(() => {
        const currentCount = container.querySelectorAll('.result-item').length;
        if (currentCount === lastResultItemCount) {
            return;
        }
        lastResultItemCount = currentCount;
        window.__dressa_reset_selection_order?.();
        syncMobileLabels();
    });
    observer.observe(container, { childList: true, subtree: false });
    window.__dressaResultsObserverAttached = true;
}

function enforceMobileMainPadding() {
    const main = document.querySelector('div.main.fillable.app.fill_width');
    if (!main) return;

    if (isCompactLayout()) {
        main.style.setProperty('--size-8', '0px', 'important');
        main.style.setProperty('padding', '0px', 'important');
    } else {
        main.style.removeProperty('--size-8');
        main.style.removeProperty('padding');
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', enforceMobileMainPadding, { once: true });
} else {
    enforceMobileMainPadding();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', attachResultsObserver, { once: true });
    document.addEventListener('DOMContentLoaded', syncMobileLabels, { once: true });
    document.addEventListener('DOMContentLoaded', dockMobileSubmitButton, { once: true });
} else {
    attachResultsObserver();
    syncMobileLabels();
    dockMobileSubmitButton();
}

if (!window.__dressaMainPaddingWatcherAttached) {
    window.addEventListener('resize', enforceMobileMainPadding, { passive: true });
    window.addEventListener('orientationchange', enforceMobileMainPadding, { passive: true });
    window.addEventListener('resize', syncMobileLabels, { passive: true });
    window.addEventListener('orientationchange', syncMobileLabels, { passive: true });
    window.addEventListener('resize', dockMobileSubmitButton, { passive: true });
    window.addEventListener('orientationchange', dockMobileSubmitButton, { passive: true });
    const mainPaddingObserver = new MutationObserver(() => {
        enforceMobileMainPadding();
    });
    mainPaddingObserver.observe(document.documentElement, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['class', 'style']
    });
    window.__dressaMainPaddingWatcherAttached = true;
}

function ensureGlobalOverlayElements() {
    if (!document.body) return;

    if (!document.getElementById('dressa-top-loader')) {
        document.body.insertAdjacentHTML(
            'beforeend',
            '<div id="dressa-top-loader" aria-hidden="true"><div class="top-loader-track"><div class="top-loader-bar"></div></div></div>'
        );
    }

    if (!document.getElementById('search-overlay')) {
        document.body.insertAdjacentHTML(
            'beforeend',
            '<div id="search-overlay" aria-hidden="true"><div class="search-overlay-card"><div class="spinner"></div><div class="search-title">Finding similar dresses</div><div class="search-copy">We are matching your upload now.</div></div></div>'
        );
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', ensureGlobalOverlayElements, { once: true });
} else {
    ensureGlobalOverlayElements();
}

if (!window.__dressaSubmitSyncAttached) {
    document.addEventListener('click', function(event) {
        const submit = event.target.closest('#submit-btn button, #submit-btn');
        if (!submit) return;
        const selected = syncSelectedIndicesToInput();
        updateSelectionUi(selected);
        window.__dressa_start_participant_loader?.();
    }, true);
    window.__dressaSubmitSyncAttached = true;
}

if (!window.__dressaSearchLoaderClickAttached) {
    document.addEventListener('click', function(event) {
        const search = event.target.closest('#search-btn button, #search-btn');
        if (!search) return;
        document.body.classList.add('dressa-searching');
        window.__dressa_start_participant_loader?.();
    }, true);
    window.__dressaSearchLoaderClickAttached = true;
}

function getParticipantLoader() {
    ensureGlobalOverlayElements();
    return document.getElementById('dressa-top-loader');
}

window.__dressaParticipantLoaderPending = 0;
window.__dressaParticipantLoaderTimer = null;

window.__dressa_start_participant_loader = function() {
    const loader = getParticipantLoader();
    if (!loader) return;
    window.__dressaParticipantLoaderPending = 1;
    loader.classList.add('active');
    document.body.classList.add('dressa-searching');
    if (window.__dressaParticipantLoaderTimer) {
        clearTimeout(window.__dressaParticipantLoaderTimer);
    }
    window.__dressaParticipantLoaderTimer = window.setTimeout(() => {
        window.__dressa_clear_participant_loader?.();
    }, 15000);
};

window.__dressa_stop_participant_loader = function() {
    const loader = getParticipantLoader();
    if (!loader) return;
    window.__dressaParticipantLoaderPending = Math.max(0, window.__dressaParticipantLoaderPending - 1);
    if (window.__dressaParticipantLoaderPending === 0) {
        loader.classList.remove('active');
        document.body.classList.remove('dressa-searching');
        if (window.__dressaParticipantLoaderTimer) {
            clearTimeout(window.__dressaParticipantLoaderTimer);
            window.__dressaParticipantLoaderTimer = null;
        }
    }
};

window.__dressa_clear_participant_loader = function() {
    const loader = getParticipantLoader();
    if (!loader) return;
    window.__dressaParticipantLoaderPending = 0;
    loader.classList.remove('active');
    document.body.classList.remove('dressa-searching');
    if (window.__dressaParticipantLoaderTimer) {
        clearTimeout(window.__dressaParticipantLoaderTimer);
        window.__dressaParticipantLoaderTimer = null;
    }
};
"""

def create_app():
    """Create the Gradio app interface."""

    with gr.Blocks(title="Dressa - Dress Similarity Study", css=APP_CSS, fill_width=True) as app:

        # State variables
        session_id_state = gr.State(value=None)
        user_id_state = gr.State(value=None)
        upload_id_state = gr.State(value=None)
        current_results_state = gr.State(value=[])
        selected_indices_state = gr.State(value=[])
        gallery_images_state = gr.State(value=[])
        upload_count_state = gr.State(value=0)

        # ==================== CONSENT SCREEN ====================
        with gr.Column(visible=True, min_width=0, elem_id="consent-screen") as consent_screen:

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
            with gr.Row(elem_id="consent-info-row"):
                with gr.Column(scale=1, min_width=0):
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

                with gr.Column(scale=1, min_width=0):
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

            with gr.Column(elem_id="consent-actions"):
                agree_btn = gr.Button("I Agree and Start", variant="primary", size="lg", elem_id="agree-btn")
                disagree_btn = gr.Button("I Do Not Agree", variant="secondary", size="lg", elem_id="disagree-btn")

            disagree_message = gr.Markdown("", visible=False)

        # ==================== MAIN APP SCREEN ====================
        with gr.Column(visible=False, elem_id="main-app-screen", min_width=0) as main_app_screen:

            gr.HTML("""
            <div id="hero">
                <div class="hero-title">Dressa</div>
                <div class="hero-subtitle">Upload one dress photo, select all images that look similar, then submit.</div>
                <div class="hero-steps">
                    <div class="step"><span class="step-num">1</span>Upload a clear dress photo</div>
                    <div class="step"><span class="step-num">2</span>Select every image that looks similar</div>
                    <div class="step"><span class="step-num">3</span>Submit ratings and continue</div>
                </div>
            </div>
            """)

            # Progress tracker
            upload_progress = gr.Markdown("Uploads completed: 0 of 3-5 recommended", elem_id="upload-progress")

            # Main layout
            with gr.Row(elem_id="main-row"):
                # Left column: Upload
                with gr.Column(scale=1, min_width=0, elem_id="upload-col"):
                    gr.Markdown("### 1. Upload Your Dress")
                    upload_image = gr.Image(
                        label="Upload a dress photo",
                        type="numpy",
                        sources=["upload"],
                        elem_id="upload-image"
                    )
                    gr.HTML("""
                        <div class="mobile-step"><span class="mobile-step-num">1</span><span>Upload a photo of a dress</span></div>
                        <div class="mobile-step"><span class="mobile-step-num">2</span><span>Select similar dresses</span></div>
                        <div class="mobile-step"><span class="mobile-step-num">3</span><span>Submit and Continue</span></div>
                    """, elem_id="mobile-instructions")
                    gr.Markdown(
                        "Use gallery or camera. If you pick the wrong photo, use the X icon to remove and re-upload.",
                        elem_id="upload-helper",
                    )
                    upload_progress_mobile = gr.Markdown(
                        "Uploads completed: 0 of 3-5 recommended",
                        elem_id="upload-progress-mobile"
                    )
                    status_text = gr.Markdown("Upload a dress photo to begin.", elem_id="status-text")

                    # Finish button (appears after minimum uploads)
                    finish_btn = gr.Button("Finish Study", variant="secondary", visible=False, elem_id="finish-btn")

                with gr.Row(elem_id="search-row"):
                    search_btn = gr.Button(
                        "Find Similar Dresses",
                        variant="primary",
                        elem_id="search-btn",
                        interactive=False,
                    )

                # Right column: Results
                with gr.Column(scale=2, min_width=0, elem_id="results-col"):
                    gr.Markdown("### 2. Select Similar Dresses")
                    progress_text = gr.Markdown("Upload a photo, then tap Find Similar Dresses.", elem_id="progress-text")

                    # Instructions for selection
                    selection_instructions = gr.Markdown(
                        "Select all images you think are similar. Tap again to unselect.",
                        visible=False,
                        elem_id="selection-instructions"
                    )

                    selection_count = gr.Markdown("", visible=False, elem_id="selection-count")

                    # Hidden textbox for selected indices
                    selected_indices_input = gr.Textbox(
                        value="[]",
                        visible=True,
                        container=False,
                        elem_id="selected-indices-input",
                        label="",
                    )

                    # Status message
                    submit_status = gr.Markdown("", elem_id="submit-status")

                    with gr.Column(elem_id="results-grid-stage", min_width=0):
                        results_grid_html = gr.HTML(value="", elem_id="results-grid-container")

            # Submit button (outside results column so it can overlay results area)
            submit_btn = gr.Button(
                "Submit Similar Selections (0)",
                variant="primary",
                size="lg",
                visible=False,
                interactive=True,
                elem_id="submit-btn"
            )

            mobile_bottom_nav = gr.Button("Dressa", elem_id="mobile-bottom-nav")

        # ==================== DEBRIEF SCREEN ====================
        with gr.Column(visible=False, min_width=0) as debrief_screen:

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

            close_btn = gr.Button("Close", variant="primary", size="lg", elem_id="close-btn")
            close_message = gr.Markdown("")

        # ==================== ADMIN ANALYTICS (PASSWORD GATED) ====================
        with gr.Column(visible=True, elem_id="admin-gate-screen") as admin_gate_screen:
            gr.Markdown("## Admin Analytics")
            gr.Markdown(
                "Private research panel for live model performance and per-upload results."
            )

            admin_setup_warning = gr.Markdown(
                "" if ADMIN_PASSWORD else
                "⚠️ `DRESSA_ADMIN_PASSWORD` is not set. "
                "Set it as a Space Secret before using admin analytics."
            )
            admin_password_input = gr.Textbox(
                label="Admin password",
                type="password",
                placeholder="Enter admin password",
            )
            admin_unlock_btn = gr.Button("Unlock Admin", variant="secondary")
            admin_unlock_status = gr.Markdown("")

        with gr.Column(visible=False, elem_id="admin-panel-screen") as admin_panel_screen:
            with gr.Tabs():
                with gr.Tab("Model leaderboard"):
                    leaderboard_summary = gr.Markdown(
                        "Unlock admin to load leaderboard."
                    )
                    leaderboard_refresh_btn = gr.Button(
                        "Refresh leaderboard", variant="primary"
                    )
                    leaderboard_status = gr.Markdown("")
                    leaderboard_df = gr.Dataframe(
                        headers=[
                            "model",
                            "total_recommendations",
                            "similar_count",
                            "hit_rate",
                            "eligible",
                        ],
                        datatype=["str", "number", "number", "number", "str"],
                        value=[],
                        interactive=False,
                        wrap=True,
                        label="Per-model performance",
                    )

                with gr.Tab("Live entries (tables + visuals)"):
                    with gr.Row():
                        max_uploads_dropdown = gr.Dropdown(
                            choices=[20, 50, 100, 200],
                            value=100,
                            label="Recent uploads to load",
                        )
                        entries_refresh_btn = gr.Button(
                            "Refresh live entries", variant="primary"
                        )

                    entries_status = gr.Markdown("")
                    summary_df = gr.Dataframe(
                        headers=[
                            "uploaded_at",
                            "user_id",
                            "upload_id",
                            "total_results",
                            "selected_count",
                            "not_similar_count",
                        ],
                        datatype=["str", "str", "str", "number", "number", "number"],
                        value=[],
                        interactive=False,
                        wrap=True,
                        label="Table A: Upload/session summary",
                    )
                    gr.Markdown("### Table B: Visual comparison")
                    visual_table_html = gr.HTML(value="")

                    with gr.Row():
                        raw_csv_btn = gr.Button("Download raw data CSV", variant="secondary")
                        raw_csv_file = gr.File(label="CSV file", interactive=False)
                    raw_csv_status = gr.Markdown("")

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

        def on_back_to_terms():
            """Return to consent screen and reset participant state."""
            reset_progress = "Uploads completed: 0 of 3-5 recommended"
            return (
                gr.update(visible=True),   # Show consent screen
                gr.update(visible=False),  # Hide main app
                gr.update(visible=False),  # Hide debrief
                None,  # session_id_state
                None,  # user_id_state
                None,  # upload_id_state
                [],    # current_results_state
                [],    # selected_indices_state
                [],    # gallery_images_state
                0,     # upload_count_state
                gr.update(value=None),  # upload_image
                gr.update(value="Upload a dress photo to begin."),  # status_text
                gr.update(value="Upload a photo, then tap Find Similar Dresses."),  # progress_text
                gr.update(visible=False),  # selection_instructions
                gr.update(visible=False, value=""),  # selection_count
                gr.update(value="[]"),  # selected_indices_input
                gr.update(visible=False, interactive=False, value="Submit Similar Selections (0)"),  # submit_btn
                gr.update(value=""),  # submit_status
                gr.update(value=""),  # results_grid_html
                gr.update(value=reset_progress),  # upload_progress
                gr.update(value=reset_progress),  # upload_progress_mobile
                gr.update(interactive=False),  # search_btn
                gr.update(visible=False),  # finish_btn
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
                    <span class="select-chip" aria-hidden="true"></span>
                    <span class="selection-badge" aria-hidden="true"></span>
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
                    "Please upload or take a dress photo first.",
                    "Upload a photo, then tap Find Similar Dresses.",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    "",
                    "[]",
                    gr.update(visible=False, interactive=False, value="Submit Similar Selections (0)"),
                    "",
                    f"Uploads completed: {upload_count} of 3-5 recommended",
                    f"Uploads completed: {upload_count} of 3-5 recommended",
                    gr.update(interactive=False),
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
                    gr.update(visible=False, interactive=False, value="Submit Similar Selections (0)"),
                    "",
                    f"Uploads completed: {new_upload_count} of 3-5 recommended",
                    f"Uploads completed: {new_upload_count} of 3-5 recommended",
                    gr.update(interactive=True),
                    gr.update(visible=new_upload_count >= MIN_UPLOADS_FOR_DEBRIEF)
                )

            progress_msg = f"Found **{len(gallery_images)}** similar dresses. Select all matching images, then submit."
            selection_text = f"Selected: 0 of {len(gallery_images)}"
            grid_html = generate_results_grid_html(gallery_images, [])

            return (
                user_id, upload_id, filtered_results, [], gallery_images, new_upload_count,
                "Search complete. Select similar dresses and submit.",
                progress_msg,
                gr.update(visible=True),
                gr.update(visible=True, value=selection_text),
                grid_html,
                "[]",
                gr.update(visible=True, interactive=False, value="Submit Similar Selections (0)"),
                "",
                f"Uploads completed: {new_upload_count} of 3-5 recommended",
                f"Uploads completed: {new_upload_count} of 3-5 recommended",
                gr.update(interactive=True),
                gr.update(visible=new_upload_count >= MIN_UPLOADS_FOR_DEBRIEF)
            )

        def on_upload_image_change(image):
            """Enable search only when an upload is present."""
            return gr.update(interactive=(image is not None))

        def on_selection_change(selected_indices_json, gallery_images):
            """Handle selection change from JavaScript."""
            total = len(gallery_images) if gallery_images else 0
            selected_indices = _normalize_selected_indices(selected_indices_json, total)

            count = len(selected_indices)
            btn_text = f"Submit Similar Selections ({count})"
            count_text = f"Selected: {count} of {total}" if total else f"Selected: {count}"

            return (
                selected_indices,
                gr.update(value=btn_text, interactive=(count > 0)),
                gr.update(value=count_text, visible=True)
            )

        def on_submit(user_id, upload_id, results, selected_indices_json, selected_indices_state, gallery_images):
            """Handle submit button click - save all ratings."""
            logger.info("=" * 50)
            logger.info("SUBMIT BUTTON CLICKED")
            logger.info(f"User ID: {user_id}")
            logger.info(f"Upload ID: {upload_id}")
            logger.info(f"Selected indices JSON: {selected_indices_json}")
            logger.info(f"Selected indices STATE: {selected_indices_state}")
            total_results = len(results) if results else 0
            selected_from_json = _normalize_selected_indices(selected_indices_json, total_results)
            selected_from_state = _normalize_selected_indices(selected_indices_state, total_results)
            selected_indices = selected_from_json if selected_from_json else selected_from_state

            logger.info(f"Normalized selected indices (JSON): {selected_from_json}")
            logger.info(f"Normalized selected indices (state): {selected_from_state}")
            logger.info(f"Final selected indices: {selected_indices}")
            logger.info(f"Total results: {total_results}")

            if not results:
                logger.warning("No results to rate")
                return (
                    "No results to rate.",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    [],
                    "[]",
                    gr.update(visible=False, interactive=False, value="Submit Similar Selections (0)"),
                    "",
                    "Upload a photo, then tap Find Similar Dresses.",
                    "Ready for a new upload.",
                    gr.update(value=None),
                    gr.update(interactive=False),
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

            status = (
                f"**Submitted.** You selected **{similar_count}** similar photos out of **{total_results}**. "
                f"({not_similar_count} marked not similar.)"
            )

            return (
                status,
                gr.update(visible=False),
                gr.update(visible=False, value=""),
                [],
                "[]",
                gr.update(visible=False, interactive=False, value="Submit Similar Selections (0)"),
                "",
                "Upload another photo, then tap Find Similar Dresses.",
                "Ready for another upload.",
                gr.update(value=None),
                gr.update(interactive=False),
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

        def load_admin_tables(max_uploads: int):
            """Load summary + visual admin tables."""
            return load_live_entries(int(max_uploads))

        def on_unlock_admin(input_password: str):
            """Unlock admin panel and auto-load leaderboard + live entries."""
            if not ADMIN_PASSWORD:
                return (
                    "⚠️ `DRESSA_ADMIN_PASSWORD` is not configured in environment secrets.",
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(value="### Best Model Right Now\nAdmin is locked."),
                    [],
                    "Admin secret is not configured.",
                    "Admin is locked.",
                    [],
                    '<div class="admin-empty">Admin is locked.</div>',
                )

            if not verify_admin_password(input_password):
                return (
                    "❌ Incorrect password.",
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(value="### Best Model Right Now\nAdmin is locked."),
                    [],
                    "Leaderboard not loaded.",
                    "Entries not loaded.",
                    [],
                    '<div class="admin-empty">Admin is locked.</div>',
                )

            leaderboard_summary_text, leaderboard_rows, leaderboard_status_text = load_model_leaderboard(
                min_support=MIN_SUPPORT_FOR_BEST_MODEL
            )
            (
                entries_status_text,
                summary_rows,
                visual_html,
            ) = load_admin_tables(100)

            return (
                "✅ Admin unlocked.",
                gr.update(visible=False),
                gr.update(visible=True),
                leaderboard_summary_text,
                leaderboard_rows,
                leaderboard_status_text,
                entries_status_text,
                summary_rows,
                visual_html,
            )

        def on_refresh_leaderboard():
            """Refresh leaderboard tab."""
            return load_model_leaderboard(min_support=MIN_SUPPORT_FOR_BEST_MODEL)

        def on_refresh_live_entries(max_uploads: int):
            """Refresh live entries tab."""
            return load_admin_tables(max_uploads)

        def on_download_raw_csv():
            """Generate and return raw CSV export."""
            filepath, status = export_raw_ratings_csv()
            if filepath:
                return gr.update(value=filepath, visible=True), status
            return gr.update(value=None, visible=False), status

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

        mobile_bottom_nav.click(
            fn=on_back_to_terms,
            inputs=[],
            outputs=[
                consent_screen,
                main_app_screen,
                debrief_screen,
                session_id_state,
                user_id_state,
                upload_id_state,
                current_results_state,
                selected_indices_state,
                gallery_images_state,
                upload_count_state,
                upload_image,
                status_text,
                progress_text,
                selection_instructions,
                selection_count,
                selected_indices_input,
                submit_btn,
                submit_status,
                results_grid_html,
                upload_progress,
                upload_progress_mobile,
                search_btn,
                finish_btn,
            ],
            show_progress="hidden",
        )

        upload_image.change(
            fn=on_upload_image_change,
            inputs=[upload_image],
            outputs=[search_btn],
            show_progress="hidden",
        )

        # Search events
        search_dep = search_btn.click(
            fn=on_search,
            inputs=[upload_image, user_id_state, upload_id_state, upload_count_state],
            outputs=[
                user_id_state, upload_id_state, current_results_state,
                selected_indices_state, gallery_images_state, upload_count_state,
                status_text, progress_text,
                selection_instructions, selection_count, results_grid_html,
                selected_indices_input, submit_btn, submit_status,
                upload_progress, upload_progress_mobile, search_btn, finish_btn
            ],
            show_progress="hidden",
        )
        search_dep.success(
            fn=None,
            inputs=[],
            outputs=[],
            show_progress="hidden",
            queue=False,
            js="() => { window.__dressa_clear_participant_loader?.(); window.__dressa_reset_selection_order?.(); window.__dressa_sync_mobile_labels?.(); }",
        )
        search_dep.failure(
            fn=None,
            inputs=[],
            outputs=[],
            show_progress="hidden",
            queue=False,
            js="() => { window.__dressa_clear_participant_loader?.(); }",
        )

        # Submit events
        submit_dep = submit_btn.click(
            fn=on_submit,
            inputs=[
                user_id_state, upload_id_state, current_results_state,
                selected_indices_input, selected_indices_state, gallery_images_state
            ],
            outputs=[
                submit_status, selection_instructions, selection_count, selected_indices_state,
                selected_indices_input, submit_btn, results_grid_html,
                progress_text, status_text, upload_image, search_btn
            ],
            show_progress="hidden",
        )
        submit_dep.success(
            fn=None,
            inputs=[],
            outputs=[],
            show_progress="hidden",
            queue=False,
            js="() => { window.__dressa_clear_participant_loader?.(); window.__dressa_reset_selection_order?.(); window.__dressa_sync_mobile_labels?.(); }",
        )
        submit_dep.failure(
            fn=None,
            inputs=[],
            outputs=[],
            show_progress="hidden",
            queue=False,
            js="() => { window.__dressa_clear_participant_loader?.(); }",
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

        # Admin unlock events
        admin_unlock_btn.click(
            fn=on_unlock_admin,
            inputs=[admin_password_input],
            outputs=[
                admin_unlock_status,
                admin_gate_screen,
                admin_panel_screen,
                leaderboard_summary,
                leaderboard_df,
                leaderboard_status,
                entries_status,
                summary_df,
                visual_table_html,
            ],
        )

        leaderboard_refresh_btn.click(
            fn=on_refresh_leaderboard,
            inputs=[],
            outputs=[leaderboard_summary, leaderboard_df, leaderboard_status],
        )

        entries_refresh_btn.click(
            fn=on_refresh_live_entries,
            inputs=[max_uploads_dropdown],
            outputs=[
                entries_status,
                summary_df,
                visual_table_html,
            ],
        )

        raw_csv_btn.click(
            fn=on_download_raw_csv,
            inputs=[],
            outputs=[raw_csv_file, raw_csv_status],
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
