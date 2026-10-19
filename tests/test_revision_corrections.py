"""Regression checks for supported revision guidance, not model quality."""
import copy
import json
from pathlib import Path

from src import evaluation_memory as em


def context():
    path = Path(__file__).parents[1] / 'data/examples/controlled_demo_inputs.json'
    return json.loads(path.read_text())['SAMPLE_CONTEXT']


def revision(context, draft):
    return em.build_revision_payload(context, draft, {
        'feedback': ['Correct the observed problems.'],
        'critical_issues': [],
        'scores': dict.fromkeys(em.CRITERIA, 1),
    }, [])


def test_revision_carries_observed_citation_failure_and_supported_calculations():
    ctx = context()
    ctx["verification_profile"] = "frozen_demo_v1"
    before = copy.deepcopy(ctx)
    payload = revision(ctx, 'Revenue growth and operating margin are unavailable.')
    checklist = payload['correction_checklist']
    assert any('citation' in item['observed_failure'].lower() for item in checklist)
    assert any('unavailable' in item['observed_failure'] for item in checklist)
    metrics = {item['metric']: item for item in payload['computed_metrics']}
    assert metrics['revenue_change_pct']['value'] == 20
    assert metrics['operating_margin_pct']['value'] == 10
    assert '[evidence.earnings]' in metrics['revenue_change_pct']['required_correction']
    assert ctx == before


def test_revision_metrics_follow_changed_inputs_instead_of_demo_constants():
    ctx = context()
    ctx['evidence']['earnings']['data']['current_revenue'] = 150
    ctx['evidence']['earnings']['data']['operating_income'] = 30
    metrics = {item['metric']: item for item in revision(ctx, 'Draft')['computed_metrics']}
    assert metrics['revenue_change_pct']['value'] == 50
    assert metrics['operating_margin_pct']['value'] == 20
    assert '50%' in metrics['revenue_change_pct']['required_correction']


def test_missing_inputs_do_not_create_calculations_or_invent_corrections():
    ctx = context()
    ctx['evidence']['earnings']['data'] = {}
    ctx['evidence']['market']['data'] = {}
    payload = revision(ctx, 'The financial inputs are missing.')
    assert payload['computed_metrics'] == []
    assert payload['derived_calculations'] == []
    assert not any('Use revenue_change_pct' in item['required_correction']
                   for item in payload['correction_checklist'])
