"""Keep final claims tied to preserved model output and source fingerprints."""
import hashlib
import json
from pathlib import Path

from src.evidence_verification import verify_demonstration

ROOT = Path(__file__).parents[1]


def test_final_model_record_matches_sources_without_changing_acceptance():
    report = json.loads((ROOT / 'data/examples/final_grounded_demonstration.json').read_text())
    for name, digest in report['runtime']['source_hashes'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    assert report['runtime']['model'] == 'Qwen/Qwen2.5-0.5B-Instruct'
    assert all(call['response_text'] == str(call['selected_score'])
               for call in report['model_calls'])
    assert all(call['selected_score'] in (0, 1, 2) for call in report['model_calls'])


def test_evidence_policy_benefit_is_separate_from_failed_model_acceptance():
    report = json.loads((ROOT / 'data/examples/final_grounded_demonstration.json').read_text())
    baseline = json.loads((ROOT / 'data/examples/grounded_baseline_input.json').read_text())
    recorded = json.loads((ROOT / 'data/examples/final_grounded_verification.json').read_text())
    actual = verify_demonstration(report, baseline)
    assert actual['all_checks_passed']
    assert actual['unchanged_model_acceptance'] == [run['passed'] for run in report['results']]
    assert actual['checks'] == recorded['checks']
    assert hashlib.sha256((ROOT / 'src/evidence_verification.py').read_bytes()).hexdigest() == recorded['verifier_sha256']


def test_real_finbert_record_has_all_five_stages_and_no_source_errors():
    record = json.loads((ROOT / 'data/examples/final_finbert_demonstration.json').read_text())
    assert record['execution_mode'] == 'fresh_local_finbert'
    result = record['result']
    assert len(result['workflow']) == 5 and not result['errors']
    assert result['trace']['ingest']['input_count'] == 2
    assert len(result['items']) == 1
    assert result['items'][0]['sentiment'] == 'negative'
