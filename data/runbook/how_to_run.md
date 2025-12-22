# Legal AI PoC Runbook

## Prerequisites
- Ensure Python 3.10+ is installed and a virtual environment is activated for dependencies.
- Install required packages: `pip install -r requirements.txt`.
- Confirm Ollama is installed locally and models are downloaded as needed.

## Start Ollama
- Start the Ollama service (daemon): `ollama serve`.
- Verify available models: `ollama list`.
- Health check (HTTP): `curl http://127.0.0.1:11434/api/tags`.

## Start Backend API
- Command (from repo root): `uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload`.
- Default port: 8000. Change with `--port <port>`.

## Start Streamlit UI
- Command (from repo root): `streamlit run apps/ui/streamlit_app.py --server.port 8501`.
- Default port: 8501. Change with `--server.port <port>`.

## Health Checks
- API health: `curl http://127.0.0.1:8000/health`.
- UI availability: open `http://127.0.0.1:8501` in a browser.
- Ollama tags: `curl http://127.0.0.1:11434/api/tags`.

## Common Errors and Fixes
- Port already in use: stop the other process or change ports (e.g., `--port 8001` for API, `--server.port 8502` for UI).
- Virtual environment missing: create and activate venv (`python -m venv .venv` then `source .venv/bin/activate` or `.venv\Scripts\activate`).
- Model not found in Ollama: pull the model before running (`ollama pull <model-name>`).
