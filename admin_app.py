"""
admin_app.py - Simple visual viewer for user_study.db

Launches a separate Gradio UI where you can:
- Pick a table (users / uploads / ratings / evaluation_ratings)
- Choose how many recent rows to load
- View them in an interactive table (sortable / scrollable)
"""

import sqlite3
import json
from pathlib import Path

import gradio as gr
import pandas as pd

from database import DEFAULT_DB_PATH


# Use the same database file as the main app
DB_PATH = DEFAULT_DB_PATH


def load_table(table_name: str, max_rows: int):
    """Load up to max_rows from the selected table into a DataFrame."""
    if not DB_PATH.exists():
        return (
            pd.DataFrame({"error": [f"Database not found at {DB_PATH}"]}),
            f"❌ Database not found at {DB_PATH}",
        )

    # Safety: only allow known tables
    allowed_tables = {
        "users": "SELECT * FROM users ORDER BY created_at DESC LIMIT ?",
        "uploads": "SELECT * FROM uploads ORDER BY uploaded_at DESC LIMIT ?",
        "ratings": "SELECT * FROM ratings ORDER BY timestamp DESC LIMIT ?",
        "evaluation_ratings": "SELECT * FROM evaluation_ratings ORDER BY timestamp DESC LIMIT ?",
    }

    if table_name not in allowed_tables:
        return (
            pd.DataFrame({"error": [f"Unknown table: {table_name}"]}),
            f"❌ Unknown table: {table_name}",
        )

    query = allowed_tables[table_name]

    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(query, conn, params=(int(max_rows),))
    finally:
        conn.close()

    n_rows = len(df)
    if n_rows == 0:
        status = f"⚠️ Table `{table_name}` has 0 rows."
    else:
        status = f"✅ Loaded {n_rows} rows from `{table_name}`."

    return df, status


def compute_model_metrics():
    """
    Compute simple per-model metrics from evaluation_ratings:
    - total_recommendations: number of times model recommended an image
    - similar_count: how many of those users marked as 'similar'
    - hit_rate: similar_count / total_recommendations
    """
    if not DB_PATH.exists():
        return pd.DataFrame({"error": [f"Database not found at {DB_PATH}"]})

    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT rating, provenance FROM evaluation_ratings",
            conn,
        )
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame({"info": ["evaluation_ratings has 0 rows yet. Collect more study data."]})

    stats: dict[str, dict[str, float]] = {}

    for _, row in df.iterrows():
        rating = row["rating"]
        try:
            prov = json.loads(row["provenance"])
        except Exception:
            continue

        for model_name in prov.keys():
            if model_name not in stats:
                stats[model_name] = {
                    "total_recommendations": 0,
                    "similar_count": 0,
                }
            stats[model_name]["total_recommendations"] += 1
            if rating == "similar":
                stats[model_name]["similar_count"] += 1

    rows = []
    for model_name, s in stats.items():
        total = s["total_recommendations"]
        similar = s["similar_count"]
        hit_rate = similar / total if total > 0 else 0.0
        rows.append(
            {
                "model": model_name,
                "total_recommendations": total,
                "similar_count": similar,
                "hit_rate": round(hit_rate, 3),
            }
        )

    return pd.DataFrame(rows)


with gr.Blocks(title="Dressa - Database Viewer") as admin_app:
    gr.Markdown("### Dressa Database & Model Viewer")

    with gr.Tabs():
        with gr.Tab("Browse raw tables"):
            gr.Markdown(
                "**Note:** The app uses `evaluation_ratings` table (not `ratings`). "
                "All user actions are saved there with timestamps."
            )
            gr.Markdown("Select a table and click **Load** to see recent rows.")

            with gr.Row():
                table_dropdown = gr.Dropdown(
                    choices=["evaluation_ratings", "ratings", "uploads", "users"],
                    value="evaluation_ratings",
                    label="Table",
                )
                max_rows_slider = gr.Slider(
                    minimum=10,
                    maximum=1000,
                    value=200,
                    step=10,
                    label="Max rows to show",
                )
                load_button = gr.Button("Load", variant="primary")

            status_md = gr.Markdown("", label="Status")

            # Let Gradio infer columns from the returned pandas DataFrame
            results_df = gr.Dataframe(
                label="Results",
                interactive=False,
                wrap=True,
            )

            load_button.click(
                fn=load_table,
                inputs=[table_dropdown, max_rows_slider],
                outputs=[results_df, status_md],
            )

        with gr.Tab("Model performance"):
            gr.Markdown(
                "**Note:** If all hit rates are 0, it means users haven't selected any items as 'similar' yet. "
                "Check the logs - if you see 'Selected indices JSON: []', the selection UI isn't capturing clicks."
            )
            gr.Markdown(
                "Click **Compute metrics** to see, for each model, "
                "how often its recommendations were marked *similar*."
            )
            compute_btn = gr.Button("Compute metrics", variant="primary")
            metrics_df = gr.Dataframe(
                label="Per-model metrics",
                interactive=False,
                wrap=True,
            )

            compute_btn.click(
                fn=compute_model_metrics,
                inputs=[],
                outputs=[metrics_df],
            )

        # ========== Danger zone: clear all data ==========

        def clear_database():
            """Delete all rows from all main tables. Keeps schema."""
            if not DB_PATH.exists():
                return f"❌ Database not found at {DB_PATH}"

            conn = sqlite3.connect(DB_PATH)
            try:
                cur = conn.cursor()
                # Order matters due to foreign keys
                cur.execute("DELETE FROM evaluation_ratings")
                cur.execute("DELETE FROM ratings")
                cur.execute("DELETE FROM uploads")
                cur.execute("DELETE FROM users")
                conn.commit()
            finally:
                conn.close()

            return "✅ All data cleared from users, uploads, ratings, and evaluation_ratings."

        with gr.Tab("Danger: clear database"):
            gr.Markdown(
                "**Warning:** This will permanently delete **all** study data "
                "from `users`, `uploads`, `ratings`, and `evaluation_ratings` "
                "in the current `user_study.db` file. The table structure will remain."
            )
            confirm_box = gr.Checkbox(
                label="I understand this cannot be undone.",
                value=False,
            )
            clear_btn = gr.Button("Clear all data", variant="stop")
            clear_status = gr.Markdown("")

            def clear_if_confirmed(confirmed: bool):
                if not confirmed:
                    return "⚠️ Tick the checkbox above to confirm before clearing."
                return clear_database()

            clear_btn.click(
                fn=clear_if_confirmed,
                inputs=[confirm_box],
                outputs=[clear_status],
            )


def main():
    """Run the admin DB viewer."""
    admin_app.queue()
    admin_app.launch(server_name="0.0.0.0", server_port=7861, share=False)


if __name__ == "__main__":
    main()

