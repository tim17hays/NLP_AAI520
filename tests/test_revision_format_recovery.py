"""Replay the actual failed Colab response without loading a model."""
import json
from pathlib import Path

import pytest

from src import evaluation_memory as em


RAW = (Path(__file__).parent / 'fixtures/qwen_literal_quote_revision.txt').read_text()


def test_actual_colab_revision_preserves_prose_and_logs_normalization(monkeypatch):
    entry = {'task': 'revise', 'response_text': RAW}
    monkeypatch.setattr(em, 'MODEL_CALL_LOG', [entry])
    result = em.parse_qwen_response(RAW, 'revise')
    prefix = '{\n  "revised_analysis": "'
    assert result['revised_analysis'] == RAW[len(prefix):-3]
    assert entry['response_text'] == RAW
    assert entry['normalized_response'] == result
    assert entry['format_normalization']
    # Formatting recovery must not hide the observed financial contradiction.
    root = Path(__file__).parents[1]
    context = json.loads((root / 'data/examples/controlled_demo_inputs.json').read_text())['SAMPLE_CONTEXT']
    context['verification_profile'] = 'frozen_demo_v1'
    assert not em.check_citations(result['revised_analysis'], context['evidence'])['passed']
    assert 'does not provide specific details on YoY revenue growth' in result['revised_analysis']
    # The narrow benchmark checker misses this paraphrase; do not claim
    # formatting recovery supplies a comprehensive factual check.


@pytest.mark.parametrize('raw', [
    '{"revised_analysis": "unfinished',
    '{"revised_analysis": "Hello "quoted"", "passed": true}',
    '{"revised_analysis": "Hello "quoted""} trailing',
    '{"revised_analysis": "Nested {"claim":"value"}"}',
    '{"revised_analysis": "Hello "quoted" \\q"}',
])
def test_ambiguous_incomplete_or_extra_field_revisions_remain_errors(raw):
    with pytest.raises(ValueError):
        em.parse_qwen_response(raw, 'revise')


def test_revision_recovery_never_applies_to_evaluator_json():
    with pytest.raises(ValueError):
        em.parse_qwen_response(RAW, 'evaluate')
