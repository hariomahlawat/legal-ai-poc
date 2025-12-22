from apps.api.services.grounding_verify import verify_grounding


def _sample_citations():
    return [
        {
            "citation_id": "CIT-1",
            "text": "File the complaint within thirty days to the commanding officer and include all relevant facts.",
        },
        {
            "citation_id": "CIT-2",
            "text": "Counsel must inform the accused of the hearing date and provide written notice.",
        },
        {
            "citation_id": "CIT-3",
            "text": "Witnesses are to be notified at least seven days prior to proceedings with formal notice.",
        },
    ]


def test_supported_bullet_passes():
    answer = (
        "Applicable provisions:\n"
        "- Placeholder [CIT-1]\n\n"
        "Step-by-step procedure:\n"
        "- File the complaint within thirty days with supporting facts. [CIT-1]\n\n"
        "Common mistakes to avoid:\n"
        "- None [CIT-1]\n\n"
        "If facts are missing:\n"
        "- Provide missing details. [CIT-1]"
    )

    ok, failures = verify_grounding(answer, _sample_citations())

    assert ok is True
    assert failures == []


def test_unsupported_bullet_fails():
    answer = (
        "Applicable provisions:\n"
        "- Placeholder [CIT-1]\n\n"
        "Step-by-step procedure:\n"
        "- Initiate mediation with local authority immediately. [CIT-1]\n\n"
        "Common mistakes to avoid:\n"
        "- None [CIT-1]\n\n"
        "If facts are missing:\n"
        "- Provide missing details. [CIT-1]"
    )

    ok, failures = verify_grounding(answer, _sample_citations())

    assert ok is False
    assert failures[0]["bullet_index"] == 0
    assert failures[0]["citation_ids"] == ["CIT-1"]
    assert failures[0]["support"]["overlap"] < 3


def test_missing_citation_fails():
    answer = (
        "Applicable provisions:\n"
        "- Placeholder [CIT-1]\n\n"
        "Step-by-step procedure:\n"
        "- File the complaint within thirty days with supporting facts.\n\n"
        "Common mistakes to avoid:\n"
        "- None [CIT-1]\n\n"
        "If facts are missing:\n"
        "- Provide missing details. [CIT-1]"
    )

    ok, failures = verify_grounding(answer, _sample_citations())

    assert ok is False
    assert failures[0]["citation_ids"] == []


def test_multiple_citations_one_supports():
    answer = (
        "Applicable provisions:\n"
        "- Placeholder [CIT-1]\n\n"
        "Step-by-step procedure:\n"
        "- Notify witnesses at least seven days before proceedings. [CIT-2][CIT-3]\n\n"
        "Common mistakes to avoid:\n"
        "- None [CIT-1]\n\n"
        "If facts are missing:\n"
        "- Provide missing details. [CIT-1]"
    )

    ok, failures = verify_grounding(answer, _sample_citations())

    assert ok is True
    assert failures == []


def test_short_bullet_uses_small_overlap_threshold():
    answer = (
        "Applicable provisions:\n"
        "- Placeholder [CIT-2]\n\n"
        "Step-by-step procedure:\n"
        "- Inform accused. [CIT-2]\n\n"
        "Common mistakes to avoid:\n"
        "- None [CIT-1]\n\n"
        "If facts are missing:\n"
        "- Provide missing details. [CIT-1]"
    )

    ok, failures = verify_grounding(answer, _sample_citations())

    assert ok is True
    assert failures == []
