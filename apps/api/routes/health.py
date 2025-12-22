from fastapi import APIRouter

from apps.api.config import (
    OLLAMA_CONNECT_TIMEOUT_SECS,
    OLLAMA_MODEL_LEGAL,
    OLLAMA_MODEL_SYS,
    OLLAMA_READ_TIMEOUT_SECS,
    OLLAMA_URL,
)

router = APIRouter()


@router.get("/health")
def health():
    return {
        "ok": True,
        "ollama_url": OLLAMA_URL,
        "ollama_model_legal": OLLAMA_MODEL_LEGAL,
        "ollama_model_sys": OLLAMA_MODEL_SYS,
        "ollama_timeouts": {
            "connect_secs": OLLAMA_CONNECT_TIMEOUT_SECS,
            "read_secs": OLLAMA_READ_TIMEOUT_SECS,
        },
    }
