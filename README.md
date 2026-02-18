---
title: Dressa User Study
colorFrom: yellow
colorTo: blue
sdk: gradio
app_file: app.py
python_version: "3.11"
pinned: false
---

# Dressa User Study

Public Gradio app for a dress-similarity user study.

## Runtime Environment Variables

- `DRESSA_DB_PATH` (recommended on Spaces: `/data/user_study.db`)
- `DRESSA_UPLOADS_DIR` (recommended on Spaces: `/data/uploads`)
- `DRESSA_IMAGES_DIR` (default: `./dress_images`)
- `DRESSA_EMBEDDINGS_DIR` (default: `./embeddings`)
- `DRESSA_PRELOAD_MODELS` (`0` or `1`, default `0`)
- `DRESSA_ENABLE_CORPUS_GROWTH` (`0` or `1`, default `0`)
- `DRESSA_ADMIN_PASSWORD` (set as Space Secret to unlock admin analytics panel)
