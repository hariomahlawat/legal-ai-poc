import pathlib
import sys

import pytest

ROOT_DIR = pathlib.Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from apps.api.services import synthesis


def test_validate_plan_valid():
    allowed_ids = {"CIT-1", "CIT-2"}
    plan = {
        "legal_object": "Court of Inquiry",
        "assumptions": ["None"],
        "steps": [
            {
                "title": "Scope",
                "points": [
                    {"text": "Define authority", "citations": ["CIT-1"]},
                    {"text": "Outline facts", "citations": ["CIT-2"]},
                ],
            }
        ],
    }

    ok, errors = synthesis._validate_plan(plan, allowed_ids)

    assert ok is True
    assert errors == []


def test_validate_plan_unknown_citation():
    allowed_ids = {"CIT-1"}
    plan = {
        "legal_object": "Court of Inquiry",
        "assumptions": [],
        "steps": [
            {
                "title": "Scope",
                "points": [{"text": "Define authority", "citations": ["CIT-2"]}],
            }
        ],
    }

    ok, errors = synthesis._validate_plan(plan, allowed_ids)

    assert ok is False
    assert any("unknown citation ID" in e for e in errors)


def test_validate_plan_missing_citations():
    allowed_ids = {"CIT-1"}
    plan = {
        "legal_object": "Court of Inquiry",
        "assumptions": [],
        "steps": [
            {
                "title": "Scope",
                "points": [{"text": "Define authority"}],
            }
        ],
    }

    ok, errors = synthesis._validate_plan(plan, allowed_ids)

    assert ok is False
    assert any("missing citations list" in e for e in errors)


def test_validate_plan_wrong_types():
    allowed_ids = {"CIT-1"}
    plan = {
        "legal_object": 123,
        "assumptions": "not-a-list",
        "steps": "invalid",
    }

    ok, errors = synthesis._validate_plan(plan, allowed_ids)

    assert ok is False
    assert len(errors) >= 3
