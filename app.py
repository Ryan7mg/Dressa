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
import zipfile
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from datetime import datetime
import shutil
import logging
import os
import io
import time
import tempfile
from typing import Any

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


def _normalize_image_for_jpeg(image: np.ndarray) -> np.ndarray:
    """Normalize uploaded image arrays for robust JPEG save/search handling."""
    if image is None:
        raise ValueError("Image is missing.")

    arr = np.asarray(image)
    if arr.size == 0:
        raise ValueError("Image is empty.")

    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    elif arr.dtype == np.bool_:
        arr = arr.astype(np.uint8) * 255
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3:
        channels = arr.shape[2]
        if channels == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif channels == 3:
            pass
        elif channels == 4:
            arr = arr[:, :, :3]
        else:
            raise ValueError(f"Unsupported channel count: {channels}. Expected 1, 3, or 4.")
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}.")

    return np.ascontiguousarray(arr, dtype=np.uint8)


def save_uploaded_image(image: np.ndarray, user_id: str) -> str:
    """Save uploaded image and return path."""
    normalized_image = _normalize_image_for_jpeg(image)

    # Create user directory
    user_dir = UPLOADS_DIR / user_id
    user_dir.mkdir(exist_ok=True)

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"upload_{timestamp}.jpg"
    filepath = user_dir / filename

    # Save image
    Image.fromarray(normalized_image).save(filepath, "JPEG", quality=95)

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
    normalized_image = _normalize_image_for_jpeg(image)

    # Ensure models are loaded
    if not model_manager.is_loaded('openai_clip'):
        logger.info("Loading models for first search...")
        load_start = time.perf_counter()
        model_manager.load_all_models()
        load_elapsed = time.perf_counter() - load_start
        logger.info(f"Model load total time: {load_elapsed:.2f}s")

    # Compute image hash for deterministic shuffle
    image_hash = compute_image_hash(normalized_image)
    logger.info(f"Image hash: {image_hash}")

    # Convert to PIL
    pil_image = Image.fromarray(normalized_image)

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


def _parse_provenance(raw_value: Any) -> dict[str, Any]:
    """Parse provenance JSON into a dictionary."""
    try:
        parsed = json.loads(raw_value or "{}")
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _get_model_leaderboard_records(min_support: int) -> tuple[list[dict[str, Any]], int]:
    """Return sorted leaderboard records and total evaluation row count."""
    global db

    if db is None:
        return [], 0

    with db._get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT rating, provenance FROM evaluation_ratings")
        rows = [dict(row) for row in cur.fetchall()]

    stats: dict[str, dict[str, int]] = {}
    for row in rows:
        rating = row.get("rating")
        provenance = _parse_provenance(row.get("provenance", "{}"))
        for model_name in provenance.keys():
            if model_name not in stats:
                stats[model_name] = {
                    "total_recommendations": 0,
                    "similar_count": 0,
                }
            stats[model_name]["total_recommendations"] += 1
            if rating == "similar":
                stats[model_name]["similar_count"] += 1

    leaderboard_records: list[dict[str, Any]] = []
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

    return leaderboard_records, len(rows)


def _build_leaderboard_summary(
    leaderboard_records: list[dict[str, Any]],
    min_support: int
) -> str:
    """Create the admin markdown summary for leaderboard."""
    refreshed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    eligible_records = [rec for rec in leaderboard_records if rec["eligible"]]
    if eligible_records:
        winner = eligible_records[0]
        return (
            "### Best Model Right Now\n"
            f"**{winner['model']}** | "
            f"Hit rate: **{winner['hit_rate']:.3f}** | "
            f"Similar: **{winner['similar_count']}** / **{winner['total_recommendations']}**\n\n"
            f"Refreshed: `{refreshed_at}`"
        )

    return (
        "### Best Model Right Now\n"
        f"No winner yet (need at least **{min_support}** recommendations per model).\n\n"
        f"Refreshed: `{refreshed_at}`"
    )


def load_model_leaderboard(min_support: int = MIN_SUPPORT_FOR_BEST_MODEL) -> tuple[str, list[list], str]:
    """
    Build model leaderboard from evaluation_ratings provenance.

    Returns:
        summary_markdown, leaderboard_rows, status_message
    """
    global db

    if db is None:
        return "### Best Model Right Now\nApp database is not initialized.", [], "Database is not initialized."

    leaderboard_records, total_rows = _get_model_leaderboard_records(min_support=min_support)
    if total_rows == 0:
        summary = "### Best Model Right Now\nNo winner yet. No evaluation data has been collected."
        return summary, [], "No evaluation ratings found."

    summary = _build_leaderboard_summary(leaderboard_records, min_support=min_support)
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
    status = f"Loaded leaderboard for {len(leaderboard_records)} models from {total_rows} ratings."
    return summary, leaderboard_rows, status


def _build_live_entries_payload(max_uploads: int = 100) -> tuple[str, list[list], list[dict[str, Any]]]:
    """Build normalized summary + visual rows for the admin live entries table."""
    global db

    if db is None:
        return "Database is not initialized.", [], []

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
        return "No uploads found yet.", [], []

    ratings_by_upload: dict[str, list[dict[str, Any]]] = {}
    for row in rating_rows:
        upload_id = row["upload_id"]
        if upload_id not in ratings_by_upload:
            ratings_by_upload[upload_id] = []
        ratings_by_upload[upload_id].append(row)

    summary_rows: list[list[Any]] = []
    visual_records: list[dict[str, Any]] = []
    for upload in upload_rows:
        upload_id = upload["upload_id"]
        rows = ratings_by_upload.get(upload_id, [])
        similar_rows = [r for r in rows if r.get("rating") == "similar"]
        not_similar_rows = [r for r in rows if r.get("rating") == "not_similar"]
        similar_image_paths = [r.get("result_image_id", "") for r in similar_rows if r.get("result_image_id")]
        not_selected_image_paths = [r.get("result_image_id", "") for r in not_similar_rows if r.get("result_image_id")]
        total_results = len(rows)

        summary_rows.append([
            upload.get("uploaded_at", ""),
            upload.get("user_id", ""),
            upload_id,
            total_results,
            len(similar_rows),
            len(not_similar_rows),
        ])

        visual_records.append({
            "uploaded_at": upload.get("uploaded_at", ""),
            "user_id": upload.get("user_id", ""),
            "upload_id": upload_id,
            "total_results": total_results,
            "similar_count": len(similar_rows),
            "not_similar_count": len(not_similar_rows),
            "query_image_path": upload.get("filepath", ""),
            "selected_image_paths": similar_image_paths,
            "not_selected_image_paths": not_selected_image_paths,
        })

    status = f"Loaded {len(summary_rows)} uploads (default limit {int(max_uploads)})."
    return status, summary_rows, visual_records


def load_live_entries(max_uploads: int = 100) -> tuple[str, list[list], str]:
    """
    Build live per-upload analytics rows.

    Returns:
        status_message, summary_rows, visual_table_html
    """
    status, summary_rows, visual_records = _build_live_entries_payload(max_uploads=max_uploads)
    if not visual_records:
        if "Database is not initialized" in status:
            return status, [], '<div class="admin-empty">Database unavailable.</div>'
        return status, [], '<div class="admin-empty">No uploads found yet.</div>'

    visual_table_html = _render_visual_table_html(visual_records)
    return status, summary_rows, visual_table_html


def _export_path(prefix: str, extension: str) -> Path:
    """Build a timestamped temp export path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(tempfile.gettempdir()) / f"{prefix}_{timestamp}.{extension}"


def _load_pdf_font(size: int) -> ImageFont.ImageFont:
    """Load a readable TrueType font for PDF rendering, with fallback."""
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in font_candidates:
        font_path = Path(candidate)
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _wrap_text_for_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    """Wrap a text line to pixel width, including long path-like tokens."""
    clean = str(text or "").strip()
    if not clean:
        return [""]

    wrapped: list[str] = []
    remaining = clean
    while remaining:
        if draw.textlength(remaining, font=font) <= max_width:
            wrapped.append(remaining)
            break

        cutoff = 1
        for i in range(1, len(remaining) + 1):
            if draw.textlength(remaining[:i], font=font) > max_width:
                cutoff = max(1, i - 1)
                break
        split_at = remaining.rfind(" ", 0, cutoff + 1)
        if split_at <= 0:
            split_at = cutoff
        wrapped.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()

    return wrapped or [clean]


def _write_text_report_pdf(
    title: str,
    lines: list[str],
    output_path: Path,
    subtitle: str
) -> None:
    """Render a plain-text report PDF with pagination using Pillow."""
    page_width, page_height = 1654, 2339  # A4-ish at 150dpi
    margin = 90
    line_height = 32
    section_spacing = 10
    title_font = _load_pdf_font(40)
    subtitle_font = _load_pdf_font(24)
    body_font = _load_pdf_font(22)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    queue = lines[:] if lines else ["No rows found."]
    pages: list[Image.Image] = []
    cursor = 0
    page_num = 0

    while cursor < len(queue):
        page_num += 1
        canvas = Image.new("RGB", (page_width, page_height), color="white")
        draw = ImageDraw.Draw(canvas)
        y = margin

        draw.text((margin, y), title, fill="#191816", font=title_font)
        y += 58
        draw.text((margin, y), subtitle, fill="#3f3a33", font=subtitle_font)
        y += 40
        draw.text((margin, y), f"Generated: {generated_at} | Page {page_num}", fill="#6f6b65", font=subtitle_font)
        y += 52

        page_start_cursor = cursor
        max_text_width = page_width - (2 * margin)
        while cursor < len(queue):
            raw_line = queue[cursor]
            wrapped_lines = _wrap_text_for_width(draw, raw_line, body_font, max_text_width)
            needed_height = (line_height * len(wrapped_lines)) + section_spacing
            if y + needed_height > (page_height - margin):
                break
            for wrapped_line in wrapped_lines:
                draw.text((margin, y), wrapped_line, fill="#191816", font=body_font)
                y += line_height
            y += section_spacing
            cursor += 1

        if cursor == page_start_cursor:
            # Hard stop safeguard for unexpectedly tall lines.
            cursor += 1
        pages.append(canvas)

    pages[0].save(
        output_path,
        format="PDF",
        save_all=True,
        append_images=pages[1:],
        resolution=150.0,
    )


def export_leaderboard_csv(min_support: int = MIN_SUPPORT_FOR_BEST_MODEL) -> tuple[str | None, str]:
    """Export model leaderboard data as CSV."""
    global db
    if db is None:
        return None, "Database is not initialized."

    leaderboard_records, total_rows = _get_model_leaderboard_records(min_support=min_support)
    if total_rows == 0:
        return None, "No evaluation ratings found."

    output_path = _export_path("dressa_model_leaderboard", "csv")
    fieldnames = ["model", "total_recommendations", "similar_count", "hit_rate", "eligible"]
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for rec in leaderboard_records:
            writer.writerow({
                "model": rec["model"],
                "total_recommendations": rec["total_recommendations"],
                "similar_count": rec["similar_count"],
                "hit_rate": round(float(rec["hit_rate"]), 6),
                "eligible": "yes" if rec["eligible"] else "no",
            })

    return str(output_path), f"Exported leaderboard CSV for {len(leaderboard_records)} models."


def export_leaderboard_pdf(min_support: int = MIN_SUPPORT_FOR_BEST_MODEL) -> tuple[str | None, str]:
    """Export model leaderboard data as a PDF report."""
    global db
    if db is None:
        return None, "Database is not initialized."

    leaderboard_records, total_rows = _get_model_leaderboard_records(min_support=min_support)
    if total_rows == 0:
        return None, "No evaluation ratings found."

    lines: list[str] = []
    for rank, rec in enumerate(leaderboard_records, start=1):
        lines.append(
            f"{rank}. {rec['model']} | total={rec['total_recommendations']} | "
            f"similar={rec['similar_count']} | hit_rate={float(rec['hit_rate']):.4f} | "
            f"eligible={'yes' if rec['eligible'] else 'no'}"
        )

    output_path = _export_path("dressa_model_leaderboard", "pdf")
    _write_text_report_pdf(
        title="Dressa Model Leaderboard",
        lines=lines,
        output_path=output_path,
        subtitle=f"Source rows: {total_rows} | Eligibility threshold: {int(min_support)}",
    )
    return str(output_path), f"Exported leaderboard PDF for {len(leaderboard_records)} models."


def export_live_entries_csv(max_uploads: int = 100) -> tuple[str | None, str]:
    """Export live entry summary + visual linkage data as CSV."""
    status, _, visual_records = _build_live_entries_payload(max_uploads=max_uploads)
    if not visual_records:
        return None, status

    output_path = _export_path("dressa_live_entries", "csv")
    fieldnames = [
        "uploaded_at",
        "user_id",
        "upload_id",
        "total_results",
        "selected_count",
        "not_selected_count",
        "query_image_path",
        "selected_image_paths_json",
        "not_selected_image_paths_json",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for rec in visual_records:
            writer.writerow({
                "uploaded_at": rec.get("uploaded_at", ""),
                "user_id": rec.get("user_id", ""),
                "upload_id": rec.get("upload_id", ""),
                "total_results": rec.get("total_results", 0),
                "selected_count": rec.get("similar_count", 0),
                "not_selected_count": rec.get("not_similar_count", 0),
                "query_image_path": rec.get("query_image_path", ""),
                "selected_image_paths_json": json.dumps(rec.get("selected_image_paths", []), ensure_ascii=False),
                "not_selected_image_paths_json": json.dumps(rec.get("not_selected_image_paths", []), ensure_ascii=False),
            })

    return str(output_path), f"Exported live entries CSV with {len(visual_records)} rows."


def export_live_entries_pdf(max_uploads: int = 100) -> tuple[str | None, str]:
    """Export live entry details as a paginated text PDF."""
    status, _, visual_records = _build_live_entries_payload(max_uploads=max_uploads)
    if not visual_records:
        return None, status

    lines: list[str] = []
    for idx, rec in enumerate(visual_records, start=1):
        lines.append("=" * 140)
        lines.append(
            f"Entry {idx} | uploaded_at={rec.get('uploaded_at', '')} | "
            f"user_id={rec.get('user_id', '')} | upload_id={rec.get('upload_id', '')}"
        )
        lines.append(
            f"Counts | total_results={rec.get('total_results', 0)} | "
            f"selected={rec.get('similar_count', 0)} | not_selected={rec.get('not_similar_count', 0)}"
        )
        lines.append(f"Query image path: {rec.get('query_image_path', '')}")

        lines.append("Selected images:")
        selected_paths = rec.get("selected_image_paths", []) or []
        if selected_paths:
            for image_path in selected_paths:
                lines.append(f"  - {image_path}")
        else:
            lines.append("  - None")

        lines.append("Not selected images:")
        not_selected_paths = rec.get("not_selected_image_paths", []) or []
        if not_selected_paths:
            for image_path in not_selected_paths:
                lines.append(f"  - {image_path}")
        else:
            lines.append("  - None")
        lines.append("")

    output_path = _export_path("dressa_live_entries", "pdf")
    _write_text_report_pdf(
        title="Dressa Live Entries Report",
        lines=lines,
        output_path=output_path,
        subtitle=f"Rows included: {len(visual_records)} | Requested limit: {int(max_uploads)}",
    )
    return str(output_path), f"Exported live entries PDF with {len(visual_records)} rows."


def export_upload_images_zip() -> tuple[str | None, str]:
    """Export full-resolution uploaded query images as ZIP with a manifest CSV."""
    global db
    if db is None:
        return None, "Database is not initialized."

    with db._get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT upload_id, user_id, filepath, uploaded_at
            FROM uploads
            ORDER BY uploaded_at DESC
            """
        )
        upload_rows = [dict(row) for row in cur.fetchall()]

    if not upload_rows:
        return None, "No uploads found yet."

    output_path = _export_path("dressa_uploaded_images_fullres", "zip")
    manifest_rows: list[dict[str, Any]] = []
    added = 0
    missing = 0
    used_arcnames: set[str] = set()

    with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for row in upload_rows:
            upload_id = str(row.get("upload_id", ""))
            user_id = str(row.get("user_id", ""))
            uploaded_at = str(row.get("uploaded_at", ""))
            filepath = str(row.get("filepath", "")).strip()
            path_obj = Path(filepath)
            exists = path_obj.exists() and path_obj.is_file()

            arcname = ""
            if exists:
                resolved = path_obj.resolve()
                if _is_under_dir(resolved, UPLOADS_DIR):
                    relative = resolved.relative_to(UPLOADS_DIR).as_posix()
                    arcname = f"full_res_uploads/{relative}"
                else:
                    arcname = f"full_res_uploads/external/{resolved.name}"
                if arcname in used_arcnames:
                    arcname = f"full_res_uploads/{upload_id}_{resolved.name}"
                zipf.write(resolved, arcname=arcname)
                used_arcnames.add(arcname)
                added += 1
            else:
                missing += 1

            manifest_rows.append({
                "upload_id": upload_id,
                "user_id": user_id,
                "uploaded_at": uploaded_at,
                "filepath": filepath,
                "exists_on_disk": "yes" if exists else "no",
                "zip_member_path": arcname,
            })

        manifest_buffer = io.StringIO()
        fieldnames = [
            "upload_id",
            "user_id",
            "uploaded_at",
            "filepath",
            "exists_on_disk",
            "zip_member_path",
        ]
        writer = csv.DictWriter(manifest_buffer, fieldnames=fieldnames)
        writer.writeheader()
        for record in manifest_rows:
            writer.writerow(record)
        zipf.writestr("metadata/upload_manifest.csv", manifest_buffer.getvalue())

    status = (
        f"Exported {added} full-resolution uploaded images to ZIP. "
        f"Missing files: {missing}. Manifest included."
    )
    return str(output_path), status


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

html,
body {
    color-scheme: light !important;
    background-color: #fefdfb !important;
}

.gradio-container {
    color-scheme: light !important;
    --body-background-fill: #fefdfb !important;
    --body-background-fill-dark: #fefdfb !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-primary-dark: #ffffff !important;
    --background-fill-secondary: #faf5ef !important;
    --background-fill-secondary-dark: #faf5ef !important;
    --block-background-fill: #ffffff !important;
    --block-background-fill-dark: #ffffff !important;
    --input-background-fill: #fffaf3 !important;
    --input-background-fill-dark: #fffaf3 !important;
    --table-even-background-fill: #ffffff !important;
    --table-odd-background-fill: #ffffff !important;
    --table-border-color: #e7ddd1 !important;
    --table-header-background-fill: #faf5ef !important;
    --body-text-color: #191816 !important;
    --body-text-color-subdued: #3f3a33 !important;
    --block-info-text-color: #3f3a33 !important;
    --block-title-text-color: #191816 !important;
    --block-label-text-color: #191816 !important;
    --input-text-color: #191816 !important;
    --input-placeholder-color: #6f6b65 !important;
    --color-text: #191816 !important;
    --link-text-color: #0b5bd3 !important;
    color: #191816 !important;
}

html.dark,
body.dark,
.dark,
[data-theme="dark"],
[data-theme="dark"] .gradio-container,
.dark .gradio-container {
    color-scheme: light !important;
    background-color: #fefdfb !important;
    color: #191816 !important;
}

html.dark .gradio-container *,
body.dark .gradio-container *,
.dark .gradio-container *,
[data-theme="dark"] .gradio-container * {
    color-scheme: light !important;
}

/* Force light on admin screens regardless of system theme */
#admin-panel-screen,
#admin-gate-screen {
    background-color: #fefdfb !important;
    color: #191816 !important;
    color-scheme: light !important;
}

#admin-panel-screen .tabs,
#admin-panel-screen .tabitem,
#admin-panel-screen .block,
#admin-panel-screen .form,
#admin-panel-screen .wrap {
    background-color: #ffffff !important;
    color: #191816 !important;
    border-color: #e7ddd1 !important;
}

#admin-panel-screen table,
#admin-panel-screen thead,
#admin-panel-screen tbody,
#admin-panel-screen tr,
#admin-panel-screen th,
#admin-panel-screen td {
    background-color: #ffffff !important;
    color: #191816 !important;
    border-color: #e7ddd1 !important;
}

#admin-panel-screen th {
    background-color: #faf5ef !important;
}

#admin-panel-screen tr:nth-child(even) td {
    background-color: #fefdfb !important;
}

#admin-panel-screen input,
#admin-panel-screen textarea,
#admin-panel-screen select,
#admin-panel-screen [class*="cell"],
#admin-panel-screen [class*="table"] {
    background-color: #fffaf3 !important;
    color: #191816 !important;
    -webkit-text-fill-color: #191816 !important;
}

#admin-gate-screen .block,
#admin-gate-screen input,
#admin-gate-screen textarea {
    background-color: #ffffff !important;
    color: #191816 !important;
}

.gradio-container table,
.gradio-container table thead,
.gradio-container table tbody,
.gradio-container table tr,
.gradio-container table th,
.gradio-container table td,
.gradio-container .gr-dataframe table,
.gradio-container .gr-dataframe th,
.gradio-container .gr-dataframe td,
.gradio-container .gr-dataframe .table-wrap,
.gradio-container .gr-dataframe .table-container {
    background: #ffffff !important;
    color: #191816 !important;
    border-color: #e7ddd1 !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    background-color: #fffaf3 !important;
    color: #191816 !important;
    -webkit-text-fill-color: #191816 !important;
    border-color: #d8cabc !important;
}

.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
    color: #7b7065 !important;
    opacity: 1 !important;
}

#admin-gate-screen input,
#admin-gate-screen textarea,
#admin-panel-screen input,
#admin-panel-screen textarea,
#admin-gate-screen [data-testid="textbox"] input,
#admin-panel-screen [data-testid="textbox"] input {
    background: #fffaf3 !important;
    color: #191816 !important;
    -webkit-text-fill-color: #191816 !important;
    border-color: #d8cabc !important;
}

#upload-image .upload-container,
#upload-image .image-container,
#upload-image .image-preview,
#upload-image .image-preview * {
    background-color: #fffaf3 !important;
    color: #191816 !important;
}

#upload-image,
#upload-image .block,
#upload-image .upload-container,
#upload-image label.upload,
#upload-image .upload,
#upload-image .wrap,
#upload-image .drop-target,
#upload-image [data-testid="image"] {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}

#consent-screen,
#consent-screen .gr-markdown,
#consent-screen .gr-markdown h1,
#consent-screen .gr-markdown h2,
#consent-screen .gr-markdown h3,
#consent-screen .gr-markdown p,
#consent-screen .gr-markdown li,
#consent-screen .gr-markdown strong {
    color: #191816 !important;
}

#consent-screen .gr-markdown a {
    color: #0b5bd3 !important;
}

.gradio-container .prose,
.gradio-container .prose * {
    color: #191816 !important;
}

.gradio-container .prose a {
    color: #0b5bd3 !important;
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
        "finish results";
    gap: 26px;
    align-items: flex-start;
}

#upload-col {
    grid-area: upload;
}

#upload-col #search-btn {
    width: 100%;
    margin-top: 10px;
}

#finish-row {
    margin: 10px 0 0 0;
    justify-content: flex-start;
    align-self: start;
    width: 100%;
}

#finish-row #finish-btn {
    width: 100%;
}

#results-col {
    grid-area: results;
}

#upload-col, #results-col {
    background: linear-gradient(155deg, rgba(255, 255, 255, 0.92), rgba(254, 249, 244, 0.88));
    border-radius: 24px;
    padding: 18px;
    box-shadow: 0 18px 42px rgba(24, 16, 8, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.7);
    overflow: visible !important;
    backdrop-filter: none;
    -webkit-backdrop-filter: none;
}

#results-col {
    border: 1px solid rgba(235, 219, 205, 0.95);
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

#hero .step-num,
html #hero .step-num { color: #ffffff !important; }

html .mobile-step-num,
#mobile-instructions .mobile-step-num { color: #ffffff !important; }

html .selection-badge,
.result-item .selection-badge { color: #ffffff !important; }

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
    border: none !important;
    box-shadow: none !important;
}

#upload-image .image-preview,
#upload-image .image-container {
    padding: 0 !important;
}

#upload-image .image-preview img,
#upload-image .image-preview canvas,
#upload-image .image-container img,
#upload-image .image-container canvas {
    display: block !important;
    width: 100% !important;
    max-width: 100% !important;
    height: auto !important;
    border: 0 !important;
    border-radius: 18px !important;
    box-shadow: none !important;
    background: transparent !important;
}

#upload-image [role="tablist"],
#upload-image [role="tab"] {
    display: none !important;
}

#upload-image .image-preview .toolbar,
#upload-image .image-preview .image-preview-controls,
#upload-image .image-preview .buttons {
    display: flex !important;
    gap: 8px !important;
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}

#upload-image .image-preview .toolbar > button:not(:last-child),
#upload-image .image-preview .image-preview-controls > button:not(:last-child),
#upload-image .image-preview .buttons > button:not(:last-child),
#upload-image .image-preview button[aria-label="Fullscreen"],
#upload-image .image-preview button[aria-label="Zoom"],
#upload-image .image-preview button[aria-label="View"],
#upload-image .image-preview button[aria-label="Edit"] {
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

#status-text .prose:empty,
#submit-status .prose:empty { display: none; }
#status-text:has(.prose:empty),
#submit-status:has(.prose:empty) { display: none; }

.results-grid {
    display: block;
    column-count: 2;
    column-gap: 12px;
    padding: 2px;
}

.results-grid .result-item,
.results-grid button.result-item {
    position: relative;
    display: block;
    width: 100%;
    height: auto !important;
    min-height: 0 !important;
    margin: 0 0 12px;
    break-inside: avoid;
    cursor: pointer;
    border-radius: 22px !important;
    overflow: hidden !important;
    border: 0 !important;
    outline: none !important;
    transition: box-shadow 0.15s ease;
    background: #ffffff !important;
    padding: 0 !important;
    box-shadow: 0 8px 20px rgba(97, 51, 29, 0.16) !important;
}

.results-grid .result-item img,
.results-grid button.result-item img {
    width: 100% !important;
    height: auto !important;
    min-height: 0 !important;
    max-height: none !important;
    display: block !important;
    vertical-align: top !important;
}

.results-grid .result-item:hover,
.results-grid button.result-item:hover {
    box-shadow: 0 10px 24px rgba(97, 51, 29, 0.20) !important;
}

.results-grid .result-item.selected,
.results-grid button.result-item.selected {
    border: 0 !important;
    box-shadow: 0 0 0 2px rgba(246, 150, 47, 0.68), 0 10px 22px rgba(96, 51, 29, 0.20) !important;
}

.result-item .select-chip,
.result-item .index-badge,
.result-item .selection-badge:not(.visible) {
    display: none !important;
}

#submit-btn {
    position: static;
    margin-top: 16px;
    width: 100%;
}
#submit-btn button { width: 100% !important; }

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
    position: relative !important;
    overflow: hidden !important;
    isolation: isolate !important;
    text-shadow: 0 1px 0 rgba(255, 255, 255, 0.34), 0 -1px 0 rgba(138, 72, 20, 0.38), 0 6px 12px rgba(98, 51, 14, 0.30) !important;
    -webkit-font-smoothing: antialiased !important;
}

button#disagree-btn,
#disagree-btn button {
    border-color: rgba(255, 255, 255, 0.72) !important;
    background: linear-gradient(155deg, rgba(255, 255, 255, 0.60), rgba(247, 247, 247, 0.36)) !important;
    color: rgba(27, 27, 30, 0.90) !important;
    box-shadow: 0 12px 24px rgba(72, 72, 78, 0.14), inset 0 1px 0 rgba(255, 255, 255, 0.75) !important;
    backdrop-filter: blur(16px) saturate(145%) !important;
    -webkit-backdrop-filter: blur(16px) saturate(145%) !important;
    text-shadow: 0 1px 0 rgba(255, 255, 255, 0.82), 0 6px 12px rgba(98, 98, 108, 0.16) !important;
}

button#agree-btn::before,
#agree-btn button::before,
button#search-btn::before,
#search-btn button::before,
button#submit-btn::before,
#submit-btn button::before,
button#finish-btn::before,
#finish-btn button::before,
button#close-btn::before,
#close-btn button::before,
button#disagree-btn::before,
#disagree-btn button::before {
    content: "";
    position: absolute;
    inset: 1px;
    border-radius: inherit;
    background: linear-gradient(140deg, rgba(255, 255, 255, 0.40), rgba(255, 255, 255, 0.14) 42%, rgba(255, 255, 255, 0.28));
    pointer-events: none;
    z-index: 0;
}

button#agree-btn::after,
#agree-btn button::after,
button#search-btn::after,
#search-btn button::after,
button#submit-btn::after,
#submit-btn button::after,
button#finish-btn::after,
#finish-btn button::after,
button#close-btn::after,
#close-btn button::after,
button#disagree-btn::after,
#disagree-btn button::after {
    content: "";
    position: absolute;
    left: 14%;
    right: 14%;
    bottom: 8px;
    height: 14px;
    border-radius: 999px;
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.34), rgba(255, 255, 255, 0));
    filter: blur(6px);
    pointer-events: none;
    z-index: 0;
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

#search-btn button[disabled]::before,
#search-btn button:disabled::before,
#search-btn button[aria-disabled="true"]::before,
#search-btn button[disabled]::after,
#search-btn button:disabled::after,
#search-btn button[aria-disabled="true"]::after {
    display: none !important;
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

#admin-back-gate-btn,
#admin-back-gate-btn button,
#admin-back-panel-btn,
#admin-back-panel-btn button {
    width: auto !important;
    min-width: 180px !important;
    max-width: max-content !important;
    margin: 0 0 10px 0 !important;
}

#admin-secret-link {
    position: fixed !important;
    right: 10px !important;
    bottom: 8px !important;
    z-index: 6445 !important;
    width: auto !important;
    margin: 0 !important;
}

#admin-secret-link button,
button#admin-secret-link {
    min-height: 28px !important;
    height: 28px !important;
    border-radius: 999px !important;
    padding: 0 10px !important;
    font-size: 10px !important;
    letter-spacing: 0.04em !important;
    text-transform: lowercase !important;
    font-weight: 600 !important;
    border: 1px solid rgba(170, 152, 138, 0.45) !important;
    background: rgba(255, 255, 255, 0.42) !important;
    color: rgba(72, 58, 46, 0.55) !important;
    box-shadow: 0 4px 12px rgba(32, 21, 12, 0.08) !important;
    backdrop-filter: blur(6px) saturate(120%) !important;
    -webkit-backdrop-filter: blur(6px) saturate(120%) !important;
    opacity: 0.22 !important;
}

#admin-secret-link button:hover,
#admin-secret-link button:focus-visible,
button#admin-secret-link:hover,
button#admin-secret-link:focus-visible {
    opacity: 0.85 !important;
    color: rgba(58, 45, 34, 0.92) !important;
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

#main-app-screen [data-testid="status-tracker"] {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: 0 !important;
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

body.dressa-searching #search-btn {
    display: none !important;
}

body:not(.dressa-main-active) #submit-btn {
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

#consent-screen {
    position: relative;
    padding-bottom: 190px;
}

#consent-actions {
    position: fixed;
    left: 0;
    right: 0;
    bottom: max(8px, env(safe-area-inset-bottom));
    z-index: 10020;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    width: 100%;
    margin: 0;
    padding: 0 8px;
    background: transparent !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
    box-shadow: none !important;
    pointer-events: none;
}

#consent-actions #agree-btn,
#consent-actions #disagree-btn {
    width: min(420px, calc(100% - 16px));
    max-width: 420px;
    margin: 0 auto;
    pointer-events: auto;
}

#consent-actions #agree-btn button,
#consent-actions #disagree-btn button {
    width: 100%;
    max-width: 100%;
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
    #consent-screen {
        padding-bottom: 190px !important;
    }

    #consent-actions {
        position: fixed !important;
        bottom: 12px !important;
        left: 0 !important;
        right: 0 !important;
        width: 100% !important;
        padding: 0 8px !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 12px !important;
        background: transparent !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
        box-shadow: none !important;
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

    #upload-click-hint-mobile {
        display: none !important;
    }

    #upload-col {
        position: static !important;
        top: auto !important;
        align-self: start !important;
    }

    #upload-col #search-btn,
    #upload-col #finish-btn,
    #upload-col #submit-btn {
        position: static !important;
        bottom: auto !important;
        width: 100% !important;
        max-width: none !important;
        margin: 10px 0 0 0 !important;
        display: flex !important;
        justify-content: center !important;
    }

    #upload-col #submit-btn button,
    #upload-col #search-btn button,
    #upload-col #finish-btn button,
    #upload-col #submit-btn > button {
        width: min(330px, 100%) !important;
        min-width: 280px !important;
        max-width: 100% !important;
        margin: 0 !important;
        justify-content: center !important;
    }

    #upload-col #search-btn button {
        font-weight: 400 !important;
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
        padding: 10px 10px 190px !important;
        box-sizing: border-box !important;
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
        position: fixed !important;
        left: 0 !important;
        right: 0 !important;
        bottom: max(8px, env(safe-area-inset-bottom)) !important;
        width: 100% !important;
        padding: 0 8px !important;
        gap: 10px !important;
        z-index: 10020 !important;
        background: transparent !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
        box-shadow: none !important;
    }

    #consent-actions #agree-btn,
    #consent-actions #disagree-btn {
        width: min(300px, calc(100vw - 20px)) !important;
        min-width: min(300px, calc(100vw - 20px)) !important;
        max-width: min(300px, calc(100vw - 20px)) !important;
        margin: 0 auto !important;
        display: flex !important;
        justify-content: center !important;
    }

    #consent-actions button#agree-btn,
    #consent-actions button#disagree-btn,
    #consent-actions #agree-btn button,
    #consent-actions #disagree-btn button {
        width: min(300px, calc(100vw - 20px)) !important;
        min-width: min(300px, calc(100vw - 20px)) !important;
        max-width: min(300px, calc(100vw - 20px)) !important;
        padding: 0 50px !important;
        white-space: nowrap !important;
    }
}

@media (hover: none) and (pointer: coarse) {
    #upload-image .upload-container > button,
    #upload-image .upload-container > button .wrap,
    #upload-image .upload-container > button .wrap * {
        font-size: 0 !important;
        line-height: 0 !important;
        color: transparent !important;
        -webkit-text-fill-color: transparent !important;
        text-shadow: none !important;
    }

    #upload-image .upload-container > button svg {
        opacity: 0 !important;
    }

    #upload-image [role="tablist"] {
        display: none !important;
    }
}

#upload-progress-mobile {
    display: none;
}

#upload-click-hint-mobile {
    margin: 0;
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

    html,
    body {
        margin: 0 !important;
        padding: 0 !important;
        overflow-y: auto !important;
    }

    body,
    .gradio-container {
        background:
            radial-gradient(circle at 84% 18%, rgba(255, 255, 255, 0.42), rgba(255, 255, 255, 0) 46%),
            linear-gradient(180deg, #ffd8bb 0%, #f7b18d 48%, #f3a37b 100%) !important;
        color: #2b2b2b !important;
    }

    .gradio-container.fill_width {
        margin: 0 !important;
    }

    .gradio-container {
        margin: 0 !important;
        padding: 0 0 calc(96px + env(safe-area-inset-bottom)) !important;
        overflow-y: auto !important;
    }

    .main.fillable.app.fill_width,
    .main.fillable.svelte-99kmwu.app.fill_width,
    .main.svelte-99kmwu {
        margin: 0 !important;
        padding: 0 !important;
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
        margin: 0 !important;
        margin-top: 0 !important;
        padding: 0 1px 0 !important;
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }

    #hero-wrap {
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        border: 0 !important;
        overflow: hidden !important;
    }

    #hero,
    #upload-progress,
    #upload-col h3,
    #upload-helper,
    #results-col h3,
    #progress-text,
    #selection-instructions,
    #selection-count,
    #submit-status {
        display: none !important;
    }

    #main-row {
        display: grid !important;
        grid-template-columns: minmax(0, 1fr) !important;
        grid-template-areas:
            "upload"
            "finish"
            "results";
        gap: 1px !important;
        align-items: stretch !important;
        margin: 0 !important;
        padding: 0 !important;
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
        overflow-x: visible !important;
    }

    #upload-col {
        position: static !important;
        top: auto !important;
        left: auto !important;
        right: auto !important;
        z-index: auto !important;
        align-self: start !important;
        height: fit-content !important;
        justify-self: stretch !important;
        margin: 0 !important;
        width: 100% !important;
        max-width: none !important;
        transform: none !important;
        box-sizing: border-box;
        background:
            radial-gradient(circle at 14% 0%, rgba(255, 255, 255, 0.46), rgba(255, 255, 255, 0) 56%),
            radial-gradient(circle at 98% 0%, rgba(255, 255, 255, 0.24), rgba(255, 255, 255, 0) 52%),
            linear-gradient(150deg, rgba(255, 245, 236, 0.20), rgba(247, 189, 144, 0.14));
        border: 1px solid rgba(255, 255, 255, 0.34);
        border-radius: 22px;
        padding: 12px;
        box-shadow: 0 16px 28px rgba(88, 47, 25, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.42), inset 0 -12px 18px rgba(196, 132, 84, 0.06);
        backdrop-filter: blur(28px) saturate(162%);
        -webkit-backdrop-filter: blur(28px) saturate(162%);
        display: grid;
        grid-template-columns: 120px minmax(0, 1fr);
        grid-template-rows: auto auto auto;
        column-gap: 10px;
        row-gap: 7px;
        align-items: start;
        align-content: start !important;
        min-height: 0 !important;
        max-height: none !important;
        overflow: visible !important;
        isolation: isolate !important;
    }

    #upload-col::before {
        content: "";
        position: absolute;
        left: 10px;
        right: 10px;
        top: 4px;
        height: 36%;
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.42), rgba(255, 255, 255, 0.02));
        pointer-events: none;
        z-index: 0;
    }

    #upload-image {
        grid-column: 1;
        grid-row: 1;
        width: 120px !important;
        height: 120px !important;
        min-height: 120px !important;
        max-height: 120px !important;
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
        z-index: 1;
    }

    #upload-image > div,
    #upload-image .block,
    #upload-image .contain,
    #upload-image .container,
    #upload-image .wrap {
        min-height: 0 !important;
        height: auto !important;
        max-height: 120px !important;
        overflow: visible !important;
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
        max-width: 120px !important;
        min-height: 0 !important;
        height: 120px !important;
        max-height: 120px !important;
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
        min-height: 120px !important;
        max-height: 120px !important;
        height: 120px !important;
        margin: 0 !important;
        padding: 0 !important;
        border: 0 !important;
        background: transparent !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    #upload-image .upload-container [class*="text"],
    #upload-image .upload-container [class*="label"],
    #upload-image .upload-container p,
    #upload-image .upload-container span {
        display: none !important;
        color: transparent !important;
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
        border: 0 !important;
        outline: 0 !important;
        background: linear-gradient(168deg, rgba(246, 246, 246, 0.24), rgba(182, 182, 182, 0.12)) !important;
        box-shadow: 0 14px 24px rgba(88, 47, 25, 0.16), inset 0 1px 0 rgba(255, 255, 255, 0.44), inset 0 -12px 18px rgba(112, 112, 112, 0.12) !important;
        backdrop-filter: blur(22px) saturate(168%) !important;
        -webkit-backdrop-filter: blur(22px) saturate(168%) !important;
        border-radius: 22px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        position: relative !important;
        z-index: 1;
    }

    #upload-image .upload-container > button .wrap::after {
        content: "";
        position: absolute;
        left: 6px;
        right: 6px;
        top: 6px;
        height: 40px;
        border-radius: 14px;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.76), rgba(255, 255, 255, 0.04));
        pointer-events: none;
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
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 15V5'/%3E%3Cpath d='m7 10 5-5 5 5'/%3E%3Cpath d='M5 16v2a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2'/%3E%3C/svg%3E");
        opacity: 1;
        filter: drop-shadow(0 1px 0 rgba(255, 255, 255, 0.92)) drop-shadow(0 16px 24px rgba(0, 0, 0, 0.48));
        pointer-events: none;
        z-index: 1;
    }

    #upload-image .image-preview {
        width: 120px !important;
        max-width: 120px !important;
        height: 120px !important;
        max-height: 120px !important;
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
        height: 120px !important;
        max-height: 120px !important;
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
        gap: 0;
        margin-top: 0 !important;
        z-index: 1;
    }

    #mobile-instructions .mobile-step {
        background: rgba(255, 255, 255, 0.22);
        border: 0 !important;
        border-radius: 20px;
        min-height: 34px;
        padding: 5px 10px;
        box-shadow: 0 6px 14px rgba(92, 48, 25, 0.10);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 11px;
        line-height: 1.2;
        color: #2b2b2b;
    }

    #mobile-instructions .mobile-step + .mobile-step {
        margin-top: 6px;
    }

    #mobile-instructions .mobile-step-num {
        width: 18px;
        height: 18px;
        min-width: 18px;
        flex: 0 0 18px;
        aspect-ratio: 1 / 1;
        border-radius: 50%;
        background: linear-gradient(155deg, #ffac4e, #f17f21);
        color: #fff;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 9px;
        line-height: 1;
        box-shadow: 0 4px 10px rgba(179, 93, 25, 0.24);
    }

    #mobile-instructions .mobile-step > span:last-child {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        display: block;
    }

    #upload-click-hint-mobile {
        grid-column: 1;
        grid-row: 1;
        position: absolute !important;
        left: 12px !important;
        top: 104px !important;
        width: 96px !important;
        max-width: 96px !important;
        justify-self: start !important;
        margin: 0 !important;
        padding: 0 !important;
        border: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
        color: rgba(255, 255, 255, 0.94) !important;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.28), 0 1px 0 rgba(255, 255, 255, 0.24) !important;
        font-size: 10px !important;
        line-height: 1.15 !important;
        font-weight: 700 !important;
        white-space: nowrap !important;
        text-align: center !important;
        z-index: 2;
        pointer-events: none !important;
    }

    #upload-click-hint-mobile p {
        margin: 0 !important;
        color: inherit !important;
    }

    #search-btn {
        width: auto;
        margin: 0 auto;
        display: flex;
        justify-content: center;
    }

    #upload-col #search-btn {
        grid-column: 1 / -1 !important;
        width: 100% !important;
        max-width: 100% !important;
        margin: 8px 0 0 !important;
        display: flex !important;
        justify-content: center !important;
        box-sizing: border-box !important;
    }

    #finish-row {
        width: 100% !important;
        max-width: 100% !important;
        margin: 0 !important;
        justify-content: center !important;
        box-sizing: border-box !important;
    }

    #search-btn button {
        width: auto !important;
        min-width: 0 !important;
        max-width: calc(100vw - 20px) !important;
        min-height: 52px !important;
        border-radius: 999px !important;
        border: 1px solid rgba(255, 212, 166, 0.88) !important;
        padding: 0 50px !important;
        background: linear-gradient(
            150deg,
            rgba(255, 190, 120, 0.80),
            rgba(238, 137, 34, 0.86) 55%,
            rgba(224, 124, 24, 0.82)
        ) !important;
        color: #fff !important;
        font-weight: 700 !important;
        letter-spacing: 0.01em !important;
        white-space: nowrap !important;
        text-align: center !important;
        justify-content: center !important;
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
        grid-column: 1 / -1;
        grid-row: 3;
        width: fit-content !important;
        max-width: calc(100% - 12px) !important;
        margin: 6px auto 0 !important;
        padding: 6px 14px !important;
        border: 0 !important;
        border-style: none !important;
        border-width: 0 !important;
        --block-border-width: 0px !important;
        border-radius: 20px;
        background: linear-gradient(160deg, rgba(255, 255, 255, 0.30), rgba(255, 255, 255, 0.15)) !important;
        box-shadow: 0 10px 18px rgba(92, 48, 25, 0.12) !important;
        backdrop-filter: blur(12px) saturate(145%) !important;
        -webkit-backdrop-filter: blur(12px) saturate(145%) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 11px;
        color: #2b1f16 !important;
        font-weight: 700;
        line-height: 1.2;
        text-shadow: 0 1px 0 rgba(255, 255, 255, 0.42), 0 2px 8px rgba(90, 45, 20, 0.18);
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        text-align: center !important;
    }

    #status-text {
        display: none !important;
    }

    #close-btn,
    button#close-btn,
    #close-btn button,
    #finish-btn,
    button#finish-btn,
    #finish-btn button {
        width: auto !important;
        min-width: min(280px, calc(100vw - 20px)) !important;
        max-width: calc(100vw - 20px) !important;
        padding: 0 50px !important;
        margin: 0 auto !important;
        display: inline-flex !important;
        justify-content: center !important;
    }

    #results-col {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
        width: 100% !important;
        max-width: 100% !important;
        position: relative !important;
        overflow: visible !important;
        min-height: 0 !important;
        height: auto !important;
    }

    #results-grid-stage {
        width: 100%;
        overflow: visible !important;
    }

    #main-row {
        overflow: visible !important;
    }

    #results-grid-container {
        position: static !important;
        left: auto !important;
        transform: none !important;
        width: 100% !important;
        margin: 0 !important;
        box-sizing: border-box;
        padding: 0 6px calc(136px + env(safe-area-inset-bottom));
    }

    .results-grid {
        column-count: 2 !important;
        column-gap: 8px !important;
        display: block !important;
        padding: 0;
        animation: none !important;
    }

    .result-item {
        display: block !important;
        width: 100% !important;
        height: auto !important;
        min-height: 0 !important;
        break-inside: avoid !important;
        -webkit-column-break-inside: avoid !important;
        page-break-inside: avoid !important;
        border: none !important;
        border-radius: 22px !important;
        box-shadow: 0 8px 20px rgba(97, 51, 29, 0.16) !important;
        background: #fff !important;
        margin: 0 0 8px !important;
        overflow: hidden !important;
    }

    .result-item img {
        width: 100% !important;
        height: auto !important;
        min-height: 0 !important;
        max-height: none !important;
        display: block !important;
        vertical-align: top !important;
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

    button#submit-btn,
    #submit-btn {
        position: sticky !important;
        bottom: calc(90px + env(safe-area-inset-bottom)) !important;
        z-index: 6450 !important;
        width: 100% !important;
        margin-top: 12px !important;
        padding: 8px 0 !important;
        background: linear-gradient(to top, rgba(254,253,251,0.92) 70%, transparent) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
    }

    button#submit-btn,
    #submit-btn button,
    #submit-btn > button {
        overflow: hidden;
        display: flex !important;
        align-items: center;
        justify-content: center;
        margin: 0 auto !important;
        white-space: nowrap;
        min-height: 52px !important;
        border-radius: 999px !important;
        width: auto !important;
        min-width: min(280px, calc(100vw - 20px)) !important;
        max-width: calc(100vw - 20px) !important;
        padding: 0 50px !important;
        border: 1px solid rgba(255, 210, 170, 0.55) !important;
        background: linear-gradient(
            150deg,
            rgba(255, 255, 255, 0.22),
            rgba(255, 200, 140, 0.18) 55%,
            rgba(255, 180, 100, 0.16)
        ) !important;
        backdrop-filter: blur(40px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(40px) saturate(180%) !important;
        box-shadow: 0 8px 20px rgba(224, 128, 27, 0.10), inset 0 1px 0 rgba(255, 255, 255, 0.62), inset 0 -1px 0 rgba(200, 120, 40, 0.12) !important;
        color: rgba(155, 65, 8, 0.92) !important;
        font-weight: 500 !important;
        text-shadow: 0 1px 0 rgba(255, 255, 255, 0.60) !important;
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
        background: rgba(255, 255, 255, 0.985) !important;
        border: none !important;
        box-shadow: 0 -6px 14px rgba(95, 54, 28, 0.10) !important;
        color: #db5f37 !important;
        font-family: "Fraunces", "Space Grotesk", serif !important;
        font-weight: 500 !important;
        font-size: 15px !important;
        letter-spacing: 0.02em !important;
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

    #admin-secret-link,
    #admin-secret-link button,
    button#admin-secret-link {
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

function updateMobileUploadOffset() {
    document.documentElement.style.removeProperty('--mobile-upload-offset');
}

function isMainAppVisible() {
    const main = document.getElementById('main-app-screen');
    if (!main) return false;
    const style = window.getComputedStyle(main);
    if (style.display === 'none' || style.visibility === 'hidden') return false;
    const rect = main.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
}

function getVisibleResultCount() {
    const container = document.getElementById('results-grid-container');
    if (!container) return 0;
    const items = [...container.querySelectorAll('.result-item')];
    return items.filter((item) => {
        if (!item) return false;
        const style = window.getComputedStyle(item);
        if (style.display === 'none' || style.visibility === 'hidden') return false;
        if (item.getAttribute('aria-hidden') === 'true') return false;
        return item.getClientRects().length > 0;
    }).length;
}

function scrollToResultsStart() {
    if (!isCompactLayout()) return;
    if (!isMainAppVisible()) return;
    const container = document.getElementById('results-grid-container');
    if (!container) return;

    const uploadCol = document.getElementById('upload-col');
    const uploadHeight = Math.ceil(uploadCol?.getBoundingClientRect().height || 0);
    const containerTop = container.getBoundingClientRect().top + window.scrollY;
    const targetTop = Math.max(0, containerTop - uploadHeight - 6);
    window.scrollTo({ top: targetTop, behavior: 'auto' });
}

function dockMobileSubmitButton() {
    const submitHost = document.getElementById('submit-btn');
    if (!submitHost) return;
    const submitHostIsButton = submitHost.matches('button');
    const mainVisible = isMainAppVisible();
    const shouldDockOnCompact = true;

    const submitBtn = submitHostIsButton
        ? submitHost
        : (submitHost.querySelector('button') || null);

    if (!mainVisible) {
        submitHost.style.setProperty('display', 'none', 'important');
        return;
    }

    if (!window.__dressaSubmitOriginalParent) {
        window.__dressaSubmitOriginalParent = submitHost.parentElement || null;
        window.__dressaSubmitOriginalNextSibling = submitHost.nextSibling || null;
    }

    if (isCompactLayout() && shouldDockOnCompact) {
        if (submitHost.parentElement !== document.body) {
            document.body.appendChild(submitHost);
        }

        const dockBottom = 'calc(90px + env(safe-area-inset-bottom))';
        submitHost.style.setProperty('position', 'fixed', 'important');
        submitHost.style.setProperty('top', 'auto', 'important');
        submitHost.style.setProperty('bottom', dockBottom, 'important');
        submitHost.style.setProperty('left', '50%', 'important');
        submitHost.style.setProperty('right', 'auto', 'important');
        submitHost.style.setProperty('transform', 'translateX(-50%)', 'important');
        submitHost.style.setProperty('z-index', '6505', 'important');
        submitHost.style.setProperty('margin', '0', 'important');
        submitHost.style.setProperty('width', 'auto', 'important');
        submitHost.style.setProperty('text-align', 'center', 'important');
        submitHost.style.setProperty('overflow', 'visible', 'important');
        submitHost.style.setProperty('display', 'inline-flex', 'important');
        submitHost.style.setProperty('justify-content', 'center', 'important');
        submitHost.style.setProperty('align-items', 'center', 'important');
        submitHost.style.setProperty('pointer-events', 'auto', 'important');
        submitHost.style.setProperty('padding', '0 50px', 'important');
        submitHost.style.setProperty('max-width', 'calc(100vw - 20px)', 'important');
        submitHost.style.setProperty('text-align', 'center', 'important');
        submitHost.dataset.mobileDocked = '1';

        if (submitBtn && submitBtn !== submitHost) {
            submitBtn.style.setProperty('position', 'relative', 'important');
            submitBtn.style.setProperty('top', 'auto', 'important');
            submitBtn.style.setProperty('left', 'auto', 'important');
            submitBtn.style.setProperty('right', 'auto', 'important');
            submitBtn.style.setProperty('bottom', 'auto', 'important');
            submitBtn.style.setProperty('transform', 'none', 'important');
            submitBtn.style.setProperty('margin', '0 auto', 'important');
            submitBtn.style.setProperty('width', 'auto', 'important');
            submitBtn.style.setProperty('max-width', 'calc(100vw - 20px)', 'important');
            submitBtn.style.setProperty('padding', '0 50px', 'important');
            submitBtn.style.setProperty('pointer-events', 'auto', 'important');
            submitBtn.style.setProperty('z-index', '6506', 'important');
        }
    } else {
        if (window.__dressaSubmitOriginalParent && submitHost.parentElement !== window.__dressaSubmitOriginalParent) {
            const parent = window.__dressaSubmitOriginalParent;
            const sibling = window.__dressaSubmitOriginalNextSibling;
            if (sibling && sibling.parentNode === parent) {
                parent.insertBefore(submitHost, sibling);
            } else {
                parent.appendChild(submitHost);
            }
        }

        if (submitHost.dataset.mobileDocked === '1') {
            submitHost.style.removeProperty('position');
            submitHost.style.removeProperty('top');
            submitHost.style.removeProperty('bottom');
            submitHost.style.removeProperty('left');
            submitHost.style.removeProperty('right');
            submitHost.style.removeProperty('transform');
            submitHost.style.removeProperty('z-index');
            submitHost.style.removeProperty('margin');
            submitHost.style.removeProperty('width');
            submitHost.style.removeProperty('text-align');
            submitHost.style.removeProperty('display');
            submitHost.style.removeProperty('justify-content');
            submitHost.style.removeProperty('align-items');
            submitHost.style.removeProperty('padding');
            submitHost.style.removeProperty('max-width');
            submitHost.style.removeProperty('overflow');
            submitHost.style.removeProperty('pointer-events');

            if (submitBtn && submitBtn !== submitHost) {
                submitBtn.style.removeProperty('position');
                submitBtn.style.removeProperty('top');
                submitBtn.style.removeProperty('left');
                submitBtn.style.removeProperty('right');
                submitBtn.style.removeProperty('bottom');
                submitBtn.style.removeProperty('transform');
                submitBtn.style.removeProperty('margin');
                submitBtn.style.removeProperty('width');
                submitBtn.style.removeProperty('max-width');
                submitBtn.style.removeProperty('padding');
                submitBtn.style.removeProperty('pointer-events');
                submitBtn.style.removeProperty('z-index');
            }

            delete submitHost.dataset.mobileDocked;
        }
    }
}

window.__dressa_dock_mobile_submit = dockMobileSubmitButton;

function syncMobileLabels() {
    const resultCount = getVisibleResultCount();
    const mainVisible = isMainAppVisible();
    document.body.classList.toggle('dressa-main-active', mainVisible);
    document.body.classList.toggle('dressa-has-results', mainVisible && resultCount > 0);
    if (mainVisible && resultCount === 0 && !document.body.classList.contains('dressa-searching')) {
        document.body.classList.remove('dressa-has-results');
    }
    if (mainVisible && resultCount === 0) {
        const loader = document.getElementById('dressa-top-loader');
        const loaderActive = Boolean(loader && loader.classList.contains('active'));
        if (!loaderActive) {
            document.body.classList.remove('dressa-searching');
        }
    }
    if (!mainVisible) {
        document.body.classList.remove('dressa-searching');
    }
    if (resultCount === 0) {
        window.__dressaSelectionOrder = [];
    }
    const submitHost = document.getElementById('submit-btn');
    if (submitHost) {
        if (!mainVisible || resultCount === 0) {
            submitHost.style.setProperty('display', 'none', 'important');
        } else {
            submitHost.style.removeProperty('display');
        }
    }
    updateSelectionBadges();

    if (!isCompactLayout()) {
        updateMobileUploadOffset();
        dockMobileSubmitButton();
        return;
    }
    const searchBtn = document.querySelector('#search-btn button') || document.querySelector('#search-btn');
    if (searchBtn) {
        searchBtn.textContent = 'Find Similar Dresses';
    }
    const submitBtn = document.querySelector('#submit-btn button') || document.querySelector('#submit-btn');
    if (submitBtn) {
        const selectedCount = getSelectedIndices().length;
        submitBtn.textContent = `Submit (${selectedCount})`;
        submitBtn.disabled = resultCount === 0;
        submitBtn.setAttribute('aria-disabled', String(submitBtn.disabled));
    }
    updateMobileUploadOffset();
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
    const resultCount = getVisibleResultCount();
    const mainVisible = isMainAppVisible();
    document.body.classList.toggle('dressa-main-active', mainVisible);
    document.body.classList.toggle('dressa-has-results', mainVisible && resultCount > 0);
    if (resultCount === 0) {
        window.__dressaSelectionOrder = [];
    }
    const submitHost = document.getElementById('submit-btn');
    if (submitHost) {
        if (!mainVisible || resultCount === 0) {
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
        submitBtn.disabled = resultCount === 0;
        submitBtn.setAttribute('aria-disabled', String(submitBtn.disabled));
    }
    updateMobileUploadOffset();
    dockMobileSubmitButton();

    const countLabel = document.getElementById('selection-count');
    if (countLabel) {
        const total = getVisibleResultCount();
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
        const currentCount = getVisibleResultCount();
        if (currentCount === lastResultItemCount) {
            return;
        }
        lastResultItemCount = currentCount;
        window.__dressa_reset_selection_order?.();
        syncMobileLabels();
        if (currentCount > 0) {
            window.requestAnimationFrame(() => {
                updateMobileUploadOffset();
                scrollToResultsStart();
            });
        }
    });
    observer.observe(container, { childList: true, subtree: false });
    window.__dressaResultsObserverAttached = true;
}

function attachUploadColObserver() {
    if (window.__dressaUploadColObserverAttached) return;
    const uploadCol = document.getElementById('upload-col');
    if (!uploadCol) {
        setTimeout(attachUploadColObserver, 300);
        return;
    }

    if (typeof ResizeObserver === 'undefined') {
        window.__dressaUploadColObserverAttached = true;
        return;
    }

    const observer = new ResizeObserver(() => {
        updateMobileUploadOffset();
        syncMobileLabels();
    });
    observer.observe(uploadCol);
    window.__dressaUploadColObserverAttached = true;
}

function attachMainScreenObserver() {
    if (window.__dressaMainScreenObserverAttached) return;
    const mainScreen = document.getElementById('main-app-screen');
    if (!mainScreen) {
        setTimeout(attachMainScreenObserver, 300);
        return;
    }

    const observer = new MutationObserver(() => {
        window.requestAnimationFrame(() => {
            syncMobileLabels();
            updateMobileUploadOffset();
            dockMobileSubmitButton();
        });
    });
    observer.observe(mainScreen, { attributes: true, attributeFilter: ['style', 'class'] });
    window.__dressaMainScreenObserverAttached = true;
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
    document.addEventListener('DOMContentLoaded', attachUploadColObserver, { once: true });
    document.addEventListener('DOMContentLoaded', attachMainScreenObserver, { once: true });
    document.addEventListener('DOMContentLoaded', syncMobileLabels, { once: true });
    document.addEventListener('DOMContentLoaded', updateMobileUploadOffset, { once: true });
    document.addEventListener('DOMContentLoaded', dockMobileSubmitButton, { once: true });
} else {
    attachResultsObserver();
    attachUploadColObserver();
    attachMainScreenObserver();
    syncMobileLabels();
    updateMobileUploadOffset();
    dockMobileSubmitButton();
}

if (!window.__dressaMainPaddingWatcherAttached) {
    window.addEventListener('resize', enforceMobileMainPadding, { passive: true });
    window.addEventListener('orientationchange', enforceMobileMainPadding, { passive: true });
    window.addEventListener('resize', syncMobileLabels, { passive: true });
    window.addEventListener('orientationchange', syncMobileLabels, { passive: true });
    window.addEventListener('resize', updateMobileUploadOffset, { passive: true });
    window.addEventListener('orientationchange', updateMobileUploadOffset, { passive: true });
    window.addEventListener('resize', dockMobileSubmitButton, { passive: true });
    window.addEventListener('orientationchange', dockMobileSubmitButton, { passive: true });
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

        const resultCount = getVisibleResultCount();
        if (resultCount > 0 && selected.length === 0) {
            const proceed = window.confirm(
                'No similar dresses selected. Are you sure none of these dresses are similar and want to submit?'
            );
            if (!proceed) {
                event.preventDefault();
                event.stopPropagation();
                event.stopImmediatePropagation();
                return;
            }
        }

        window.__dressa_start_participant_loader?.();
    }, true);
    window.__dressaSubmitSyncAttached = true;
}

if (!window.__dressaSearchLoaderClickAttached) {
    document.addEventListener('click', function(event) {
        const searchHost = event.target.closest('#search-btn button, button#search-btn, #search-btn');
        if (!searchHost) return;
        const search = searchHost.matches('button') ? searchHost : (searchHost.querySelector('button') || searchHost);
        if (search && (search.disabled || search.getAttribute('aria-disabled') === 'true')) {
            return;
        }
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
        participant_view_state = gr.State(value="consent")
        prefetch_results_state = gr.State(value=None)  # (gallery_images, filtered_results, image_hash) or None

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
            """, elem_id="hero-wrap")

            # Progress tracker
            upload_progress = gr.Markdown("Uploads completed: 0 of 3-5 recommended", elem_id="upload-progress")

            # Main layout
            with gr.Row(elem_id="main-row"):
                # Left column: Upload
                with gr.Column(scale=1, min_width=0, elem_id="upload-col"):
                    gr.Markdown("### 1. Upload Your Dress")
                    upload_image = gr.Image(
                        label="Upload a dress photo",
                        show_label=False,
                        type="numpy",
                        sources=["upload"],
                        elem_id="upload-image"
                    )
                    upload_click_hint_mobile = gr.Markdown(
                        "Click to upload",
                        elem_id="upload-click-hint-mobile",
                        visible=True,
                    )
                    gr.HTML("""
                        <div class="mobile-step"><span class="mobile-step-num">1</span><span>Upload dress photo</span></div>
                        <div class="mobile-step"><span class="mobile-step-num">2</span><span>Select similar dresses</span></div>
                        <div class="mobile-step"><span class="mobile-step-num">3</span><span>Submit and continue</span></div>
                    """, elem_id="mobile-instructions")
                    gr.Markdown(
                        "Use gallery or camera. If you pick the wrong photo, use the X icon to remove and re-upload.",
                        elem_id="upload-helper",
                    )
                    search_btn = gr.Button(
                        "Find Similar Dresses",
                        variant="primary",
                        elem_id="search-btn",
                        interactive=False,
                    )
                    upload_progress_mobile = gr.Markdown(
                        "Uploads completed: 0 of 3-5 recommended",
                        elem_id="upload-progress-mobile"
                    )
                    status_text = gr.Markdown("", visible=False, elem_id="status-text")

                # Finish button (appears after minimum uploads)
                with gr.Row(elem_id="finish-row"):
                    finish_btn = gr.Button("Finish Study", variant="secondary", visible=False, elem_id="finish-btn")

                # Right column: Results
                with gr.Column(scale=2, min_width=0, elem_id="results-col"):
                    gr.Markdown("### 2. Select Similar Dresses")
                    progress_text = gr.Markdown("Upload a photo, then tap Find Similar Dresses.", elem_id="progress-text")

                    # Instructions for selection
                    selection_instructions = gr.Markdown(
                        "Select all images you think are similar. **It's okay to submit with 0 selected** — tap again to unselect.",
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
                    submit_status = gr.Markdown("", visible=False, elem_id="submit-status")

                    with gr.Column(elem_id="results-grid-stage", min_width=0):
                        results_grid_html = gr.HTML(value="", elem_id="results-grid-container")
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
        with gr.Column(visible=False, elem_id="admin-gate-screen") as admin_gate_screen:
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
            admin_back_btn_gate = gr.Button("Back to Study", variant="secondary", elem_id="admin-back-gate-btn")

        with gr.Column(visible=False, elem_id="admin-panel-screen") as admin_panel_screen:
            admin_back_btn_panel = gr.Button("Back to Study", variant="secondary", elem_id="admin-back-panel-btn")
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
                    with gr.Row():
                        leaderboard_csv_btn = gr.Button("Download leaderboard CSV", variant="secondary")
                        leaderboard_pdf_btn = gr.Button("Download leaderboard PDF", variant="secondary")
                    with gr.Row():
                        leaderboard_csv_file = gr.File(
                            label="Leaderboard CSV file",
                            interactive=False,
                            visible=False,
                        )
                        leaderboard_pdf_file = gr.File(
                            label="Leaderboard PDF file",
                            interactive=False,
                            visible=False,
                        )
                    leaderboard_export_status = gr.Markdown("")

                with gr.Tab("Live entries (tables + visuals)"):
                    with gr.Row():
                        max_uploads_dropdown = gr.Dropdown(
                            choices=[20, 50, 100, 200],
                            value=20,
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

                    gr.Markdown("### Downloads")
                    with gr.Row():
                        entries_csv_btn = gr.Button("Download live entries CSV", variant="secondary")
                        entries_pdf_btn = gr.Button("Download live entries PDF", variant="secondary")
                    with gr.Row():
                        uploads_zip_btn = gr.Button("Download full-size uploaded images ZIP", variant="secondary")
                        raw_csv_btn = gr.Button("Download raw data CSV", variant="secondary")
                    with gr.Row():
                        entries_csv_file = gr.File(
                            label="Live entries CSV file",
                            interactive=False,
                            visible=False,
                        )
                        entries_pdf_file = gr.File(
                            label="Live entries PDF file",
                            interactive=False,
                            visible=False,
                        )
                    with gr.Row():
                        uploads_zip_file = gr.File(
                            label="Full-size uploads ZIP file",
                            interactive=False,
                            visible=False,
                        )
                        raw_csv_file = gr.File(
                            label="Raw ratings CSV file",
                            interactive=False,
                            visible=False,
                        )
                    entries_export_status = gr.Markdown("")
                    raw_csv_status = gr.Markdown("")

        admin_secret_link = gr.Button("research tools", variant="secondary", size="sm", elem_id="admin-secret-link")

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
                "main",  # participant_view_state
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
                gr.update(value="Click to upload", visible=True),  # upload_click_hint_mobile
                gr.update(value="", visible=False),  # status_text
                gr.update(value="Upload a photo, then tap Find Similar Dresses."),  # progress_text
                gr.update(visible=False),  # selection_instructions
                gr.update(visible=False, value=""),  # selection_count
                gr.update(value="[]"),  # selected_indices_input
                gr.update(visible=False, interactive=False, value="Submit Similar Selections (0)"),  # submit_btn
                gr.update(value="", visible=False),  # submit_status
                gr.update(value=""),  # results_grid_html
                gr.update(value=reset_progress),  # upload_progress
                gr.update(value=reset_progress),  # upload_progress_mobile
                gr.update(visible=True, interactive=False),  # search_btn
                gr.update(visible=False),  # finish_btn
                "consent",  # participant_view_state
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

        def on_prefetch(image, user_id):
            """Silently run search on upload so results are ready when button is clicked."""
            if image is None:
                return None
            try:
                results = search_similar_dresses(image, user_id, "prefetch")
                filtered_results, gallery_images = filter_results_for_gallery(results)
                image_hash = compute_image_hash(_normalize_image_for_jpeg(image))
                return (gallery_images, filtered_results, image_hash)
            except Exception:
                logger.exception("Prefetch failed")
                return None

        def on_reveal_results(image, user_id, upload_id, upload_count, prefetch_state):
            """Handle search button click, using prefetch data when available."""
            if image is None:
                return (
                    user_id, upload_id, [], [], [], upload_count,
                    gr.update(visible=False, value=""),
                    "Upload a photo, then tap Find Similar Dresses.",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    "",
                    "[]",
                    gr.update(visible=False, interactive=False, value="Submit Similar Selections (0)"),
                    gr.update(visible=False, value=""),
                    f"Uploads completed: {upload_count} of 3-5 recommended",
                    f"Uploads completed: {upload_count} of 3-5 recommended",
                    gr.update(visible=True, interactive=False),
                    gr.update(visible=upload_count >= MIN_UPLOADS_FOR_DEBRIEF)
                )

            next_upload_id = upload_id
            next_upload_count = upload_count
            upload_record_created = False

            try:
                # Save uploaded image
                filepath = save_uploaded_image(image, user_id)
                next_upload_id = db.create_upload(user_id, filepath)
                next_upload_count = upload_count + 1
                upload_record_created = True
                logger.info(f"New upload: {next_upload_id} (count: {next_upload_count})")

                # Use prefetch results if available and image matches
                used_prefetch = False
                if prefetch_state is not None:
                    try:
                        prefetch_gallery, prefetch_filtered, prefetch_hash = prefetch_state
                        current_hash = compute_image_hash(_normalize_image_for_jpeg(image))
                        if current_hash == prefetch_hash and prefetch_gallery:
                            gallery_images = prefetch_gallery
                            filtered_results = prefetch_filtered
                            used_prefetch = True
                            logger.info("Using prefetch results (hash match)")
                    except Exception:
                        logger.warning("Prefetch state invalid, falling back to fresh search")

                if not used_prefetch:
                    # Fallback: run fresh search
                    results = search_similar_dresses(image, user_id, next_upload_id)
                    filtered_results, gallery_images = filter_results_for_gallery(results)

                if not gallery_images:
                    return (
                        user_id, next_upload_id, filtered_results, [], [], next_upload_count,
                        gr.update(visible=True, value="Search complete, but no corpus images were found."),
                        "No results found. Try another photo.",
                        gr.update(visible=False),
                        gr.update(visible=False),
                        "",
                        "[]",
                        gr.update(visible=False, interactive=False, value="Submit Similar Selections (0)"),
                        gr.update(visible=False, value=""),
                        f"Uploads completed: {next_upload_count} of 3-5 recommended",
                        f"Uploads completed: {next_upload_count} of 3-5 recommended",
                        gr.update(visible=True, interactive=True),
                        gr.update(visible=next_upload_count >= MIN_UPLOADS_FOR_DEBRIEF)
                    )

                progress_msg = (
                    f"Found **{len(gallery_images)}** similar dresses. Select all matching images, then submit. "
                    "If none are similar, submit with 0 selected."
                )
                selection_text = f"Selected: 0 of {len(gallery_images)}"
                grid_html = generate_results_grid_html(gallery_images, [])

                return (
                    user_id, next_upload_id, filtered_results, [], gallery_images, next_upload_count,
                    gr.update(visible=False, value=""),
                    progress_msg,
                    gr.update(visible=True),
                    gr.update(visible=True, value=selection_text),
                    grid_html,
                    "[]",
                    gr.update(visible=True, interactive=True, value="Submit Similar Selections (0)"),
                    gr.update(visible=False, value=""),
                    f"Uploads completed: {next_upload_count} of 3-5 recommended",
                    f"Uploads completed: {next_upload_count} of 3-5 recommended",
                    gr.update(visible=False, interactive=False),
                    gr.update(visible=next_upload_count >= MIN_UPLOADS_FOR_DEBRIEF)
                )
            except Exception:
                logger.exception(
                    "Search flow failed (user_id=%s, upload_id=%s, upload_count=%s, upload_record_created=%s)",
                    user_id,
                    upload_id,
                    upload_count,
                    upload_record_created,
                )
                effective_upload_id = next_upload_id if upload_record_created else upload_id
                effective_upload_count = next_upload_count if upload_record_created else upload_count
                progress_line = f"Uploads completed: {effective_upload_count} of 3-5 recommended"
                return (
                    user_id, effective_upload_id, [], [], [], effective_upload_count,
                    gr.update(visible=True, value="Search failed for this photo. Tap X to re-upload, then try Find Similar Dresses again."),
                    "Search failed to load results. Please retry.",
                    gr.update(visible=False),
                    gr.update(visible=False, value=""),
                    "",
                    "[]",
                    gr.update(visible=False, interactive=False, value="Submit Similar Selections (0)"),
                    gr.update(visible=False, value=""),
                    progress_line,
                    progress_line,
                    gr.update(visible=True, interactive=True),
                    gr.update(visible=effective_upload_count >= MIN_UPLOADS_FOR_DEBRIEF)
                )

        def on_upload_image_change(image):
            """Enable/disable search button and clear stale results when image removed."""
            has_image = image is not None
            if has_image:
                return (
                    gr.update(visible=True, interactive=True),
                    gr.update(value="Click to upload", visible=False),
                    gr.update(),  # results_grid_html unchanged
                    gr.update(),  # progress_text unchanged
                    gr.update(),  # submit_btn unchanged
                    gr.update(),  # selection_instructions unchanged
                    gr.update(),  # selection_count unchanged
                )
            return (
                gr.update(visible=True, interactive=False),
                gr.update(value="Click to upload", visible=True),
                "",  # clear results grid
                "Upload a photo, then tap Find Similar Dresses.",  # reset progress text
                gr.update(visible=False),  # hide submit button
                gr.update(visible=False),  # hide selection instructions
                gr.update(visible=False),  # hide selection count
            )

        def on_selection_change(selected_indices_json, gallery_images):
            """Handle selection change from JavaScript."""
            total = len(gallery_images) if gallery_images else 0
            selected_indices = _normalize_selected_indices(selected_indices_json, total)

            count = len(selected_indices)
            btn_text = f"Submit Similar Selections ({count})"
            count_text = f"Selected: {count} of {total}" if total else f"Selected: {count}"

            return (
                selected_indices,
                gr.update(value=btn_text, interactive=(total > 0)),
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
                    gr.update(visible=True, value="No results to rate."),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    [],
                    "[]",
                    gr.update(visible=False, interactive=False, value="Submit Similar Selections (0)"),
                    "",
                    "Upload a photo, then tap Find Similar Dresses.",
                    gr.update(visible=False, value=""),
                    gr.update(value=None),
                    gr.update(visible=True, interactive=False),
                    gr.update(value="Click to upload", visible=True),
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
                gr.update(visible=True, value=status),
                gr.update(visible=False),
                gr.update(visible=False, value=""),
                [],
                "[]",
                gr.update(visible=False, interactive=False, value="Submit Similar Selections (0)"),
                "",
                "Upload another photo, then tap Find Similar Dresses.",
                gr.update(visible=False, value=""),
                gr.update(value=None),
                gr.update(visible=True, interactive=False),
                gr.update(value="Click to upload", visible=True),
            )

        def on_finish(session_id):
            """Handle finish study button - show debrief screen."""
            return (
                gr.update(visible=False),  # Hide main app
                gr.update(visible=True),   # Show debrief
                session_id,  # Display session ID
                "debrief",  # participant_view_state
            )

        def on_close():
            """Handle close button on debrief screen."""
            return gr.update(value="Thank you for participating. You may close this window.")

        def on_open_admin():
            """Show admin access screen and hide participant UI screens."""
            return (
                gr.update(visible=False),  # consent_screen
                gr.update(visible=False),  # main_app_screen
                gr.update(visible=False),  # debrief_screen
                gr.update(visible=True),   # admin_gate_screen
                gr.update(visible=False),  # admin_panel_screen
                gr.update(value=""),       # admin_password_input
                gr.update(value=""),       # admin_unlock_status
            )

        def on_close_admin(participant_view):
            """Return from admin area to the last participant-facing screen."""
            view = (participant_view or "consent").strip().lower()
            show_consent = view not in {"main", "debrief"}
            show_main = view == "main"
            show_debrief = view == "debrief"
            return (
                gr.update(visible=show_consent),  # consent_screen
                gr.update(visible=show_main),     # main_app_screen
                gr.update(visible=show_debrief),  # debrief_screen
                gr.update(visible=False),         # admin_gate_screen
                gr.update(visible=False),         # admin_panel_screen
                gr.update(value=""),              # admin_password_input
                gr.update(value=""),              # admin_unlock_status
            )

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

            normalized_password = (input_password or "").strip()
            if not verify_admin_password(normalized_password):
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

            leaderboard_summary_text = "### Best Model Right Now\nLoading..."
            leaderboard_rows = []
            leaderboard_status_text = "Leaderboard not loaded yet."
            entries_status_text = "Admin unlocked. Tap Refresh live entries to load Table A and Table B."
            summary_rows = []
            visual_html = '<div class="admin-empty">Tap Refresh live entries to load data.</div>'

            try:
                leaderboard_summary_text, leaderboard_rows, leaderboard_status_text = load_model_leaderboard(
                    min_support=MIN_SUPPORT_FOR_BEST_MODEL
                )
            except Exception:
                logger.exception("Admin unlock: failed to auto-load leaderboard")
                leaderboard_summary_text = "### Best Model Right Now\nTemporarily unavailable."
                leaderboard_rows = []
                leaderboard_status_text = "Leaderboard failed to auto-load. Tap Refresh leaderboard."

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

        def on_download_leaderboard_csv():
            """Generate and return leaderboard CSV export."""
            filepath, status = export_leaderboard_csv(min_support=MIN_SUPPORT_FOR_BEST_MODEL)
            if filepath:
                return gr.update(value=filepath, visible=True), status
            return gr.update(value=None, visible=False), status

        def on_download_leaderboard_pdf():
            """Generate and return leaderboard PDF export."""
            filepath, status = export_leaderboard_pdf(min_support=MIN_SUPPORT_FOR_BEST_MODEL)
            if filepath:
                return gr.update(value=filepath, visible=True), status
            return gr.update(value=None, visible=False), status

        def on_download_live_entries_csv(max_uploads: int):
            """Generate and return live entries CSV export."""
            filepath, status = export_live_entries_csv(max_uploads=int(max_uploads))
            if filepath:
                return gr.update(value=filepath, visible=True), status
            return gr.update(value=None, visible=False), status

        def on_download_live_entries_pdf(max_uploads: int):
            """Generate and return live entries PDF export."""
            filepath, status = export_live_entries_pdf(max_uploads=int(max_uploads))
            if filepath:
                return gr.update(value=filepath, visible=True), status
            return gr.update(value=None, visible=False), status

        def on_download_upload_images_zip():
            """Generate and return full-resolution upload ZIP export."""
            filepath, status = export_upload_images_zip()
            if filepath:
                return gr.update(value=filepath, visible=True), status
            return gr.update(value=None, visible=False), status

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
                debrief_screen,
                participant_view_state,
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
                upload_click_hint_mobile,
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
                participant_view_state,
            ],
            show_progress="hidden",
        )

        # Enable/disable button immediately when image changes (must be defined first so it runs before prefetch)
        upload_change_dep = upload_image.change(
            fn=on_upload_image_change,
            inputs=[upload_image],
            outputs=[
                search_btn, upload_click_hint_mobile,
                results_grid_html, progress_text,
                submit_btn, selection_instructions, selection_count,
            ],
            show_progress="hidden",
        )

        # Prefetch triggered silently after button is enabled
        prefetch_dep = upload_image.change(
            fn=on_prefetch,
            inputs=[upload_image, user_id_state],
            outputs=[prefetch_results_state],
            show_progress="hidden",
        )
        upload_change_dep.success(
            fn=None,
            inputs=[],
            outputs=[],
            show_progress="hidden",
            queue=False,
            js="() => { window.__dressa_sync_mobile_labels?.(); }",
        )

        # Search events — use on_reveal_results which reads from prefetch state
        search_dep = search_btn.click(
            fn=on_reveal_results,
            inputs=[upload_image, user_id_state, upload_id_state, upload_count_state,
                    prefetch_results_state],
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
                progress_text, status_text, upload_image, search_btn, upload_click_hint_mobile
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
            outputs=[main_app_screen, debrief_screen, session_id_display, participant_view_state]
        )

        # Close events
        close_btn.click(
            fn=on_close,
            inputs=[],
            outputs=[close_message]
        )

        admin_secret_link.click(
            fn=on_open_admin,
            inputs=[],
            outputs=[
                consent_screen,
                main_app_screen,
                debrief_screen,
                admin_gate_screen,
                admin_panel_screen,
                admin_password_input,
                admin_unlock_status,
            ],
            show_progress="hidden",
        )

        admin_back_btn_gate.click(
            fn=on_close_admin,
            inputs=[participant_view_state],
            outputs=[
                consent_screen,
                main_app_screen,
                debrief_screen,
                admin_gate_screen,
                admin_panel_screen,
                admin_password_input,
                admin_unlock_status,
            ],
            show_progress="hidden",
        )

        admin_back_btn_panel.click(
            fn=on_close_admin,
            inputs=[participant_view_state],
            outputs=[
                consent_screen,
                main_app_screen,
                debrief_screen,
                admin_gate_screen,
                admin_panel_screen,
                admin_password_input,
                admin_unlock_status,
            ],
            show_progress="hidden",
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
        leaderboard_csv_btn.click(
            fn=on_download_leaderboard_csv,
            inputs=[],
            outputs=[leaderboard_csv_file, leaderboard_export_status],
        )
        leaderboard_pdf_btn.click(
            fn=on_download_leaderboard_pdf,
            inputs=[],
            outputs=[leaderboard_pdf_file, leaderboard_export_status],
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
        entries_csv_btn.click(
            fn=on_download_live_entries_csv,
            inputs=[max_uploads_dropdown],
            outputs=[entries_csv_file, entries_export_status],
        )
        entries_pdf_btn.click(
            fn=on_download_live_entries_pdf,
            inputs=[max_uploads_dropdown],
            outputs=[entries_pdf_file, entries_export_status],
        )
        uploads_zip_btn.click(
            fn=on_download_upload_images_zip,
            inputs=[],
            outputs=[uploads_zip_file, entries_export_status],
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
