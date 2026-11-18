import json

import pytest

from clipper.campaign import CampaignBrief, deterministic_issues, load, review
from clipper.detect import Highlight
from clipper.transcribe import Transcript, Word


def _highlight(start=0, end=20):
    return Highlight(start, end, 80, "Título", "Gancho", "Motivo")


def test_load_campaign_and_validate_literal_requirements(tmp_path):
    path = tmp_path / "brief.json"
    path.write_text(json.dumps({"name": "Marca", "objective": "Ensinar", "min_duration": 15,
                                "max_duration": 30, "required_terms": ["produto"],
                                "forbidden_terms": ["garantido"]}), encoding="utf-8")
    brief = load(path)
    transcript = Transcript("pt", 30, [Word("Produto", 0, 1), Word("garantido", 2, 3)])
    issues = deterministic_issues(brief, _highlight(0, 20), transcript)
    assert "contains forbidden term: garantido" in issues
    assert not any("missing" in issue for issue in issues)


def test_load_campaign_requires_name_and_objective(tmp_path):
    path = tmp_path / "brief.json"
    path.write_text('{"name": "x"}', encoding="utf-8")
    with pytest.raises(ValueError, match="name and objective"):
        load(path)


def test_review_combines_llm_and_deterministic_checks_and_caches(tmp_path):
    brief = CampaignBrief("Marca", "Ensinar", required_terms=("produto",))
    transcript = Transcript("pt", 30, [Word("Produto", 0, 1), Word("claro.", 1, 2)])
    calls = []

    def agent(system, user, schema):
        calls.append(user)
        return {"reviews": [{"index": 0, "approved": True, "score": 92, "reasons": ["Completo"],
                              "suggested_title": "Título", "caption": "Legenda", "hashtags": ["#marca"]}]}

    cache = tmp_path / "review.json"
    result = review(brief, [_highlight()], transcript, agent, cache)
    assert result[0]["approved"] is True
    assert result[0]["score"] == 92
    assert review(brief, [_highlight()], transcript, agent, cache) == result
    assert len(calls) == 1
