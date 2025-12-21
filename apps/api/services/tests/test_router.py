import pytest

from apps.api.services.router import route_domain

# --- System help routing ---


def test_route_domain_system_help_backend():
    assert route_domain("how to start backend and front end?") == "SYSTEM_HELP"


def test_route_domain_system_help_uvicorn():
    assert route_domain("uvicorn port already in use") == "SYSTEM_HELP"


# --- Legal routing ---


def test_route_domain_legal_coi_procedure():
    assert route_domain("COI procedure for recording evidence") == "LEGAL"


def test_route_domain_legal_disciplinary():
    assert route_domain("disciplinary action under service rules") == "LEGAL"


# --- Clarification routing ---


def test_route_domain_clarify_short_procedure():
    assert route_domain("procedure?") == "CLARIFY"


def test_route_domain_clarify_empty():
    assert route_domain("") == "CLARIFY"
