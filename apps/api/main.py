from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.routes.health import router as health_router
from apps.api.routes.chat import router as chat_router
from apps.api.routes.citations import router as citations_router

app = FastAPI(title="Legal AI API", version="0.2.0")

# Streamlit runs on 8501, API on 8000. Allow local calls.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(citations_router)
