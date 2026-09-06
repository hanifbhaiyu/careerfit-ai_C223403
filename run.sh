#!/usr/bin/env bash
# Start the backend and frontend together. Ctrl+C stops both.
set -e

if [ ! -f .env ]; then
  echo "No .env found. Run: cp .env.example .env   then add your keys."
  exit 1
fi

if [ ! -d data/faiss_index ]; then
  echo "Building the vector index (first run only)..."
  python -m backend.rag.ingest --force
fi

echo "Starting API on http://127.0.0.1:8000 ..."
uvicorn backend.main:app --reload --port 8000 &
API_PID=$!

trap "kill $API_PID 2>/dev/null" EXIT
sleep 3

echo "Starting UI on http://localhost:8501 ..."
streamlit run frontend/app.py
