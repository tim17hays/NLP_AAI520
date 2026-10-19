"""Reproduce malformed five-criterion output without any model/GPU calls."""
import copy
import json

import pytest

from src.evaluation_memory import CRITERIA, parse_qwen_response
from src.model_transport import with_schema_retry


def valid_evaluation():
    return {"scores": dict(zip(CRITERIA, (1, 2, 1, 2, 2))),
            "feedback": ["Check the supplied sources."],
            "critical_issues": [], "lesson": ""}


def scripted_transport(responses, log, received):
    responses = iter(responses)

    def once(instructions, payload):
        received.append((instructions, copy.deepcopy(payload)))
        raw = next(responses)
        entry = {"task": "evaluate", "response_text": raw, "status": "started"}
        log.append(entry)
        try:
            result = parse_qwen_response(raw, "evaluate")
        except ValueError as exc:
            entry.update(status="error", error=str(exc))
            raise
        entry["status"] = "validated"
        return result

    return once


def test_missing_criterion_is_retried_and_both_raw_responses_are_kept():
    good = valid_evaluation()
    bad = copy.deepcopy(good)
    bad["scores"].pop("clarity")
    raw_bad, raw_good = json.dumps(bad), json.dumps(good)
    log, received = [], []
    once = scripted_transport([raw_bad, raw_good], log, received)
    call = with_schema_retry(once, log, CRITERIA)
    payload = {"task": "evaluate", "draft_analysis": "Actual draft", "evidence": {}}
    before = copy.deepcopy(payload)
    assert call("Evaluate only the evidence.", payload) == good
    assert payload == before
    assert len(log) == 2
    assert log[0]["response_text"] == raw_bad and log[0]["status"] == "error"
    assert log[1]["response_text"] == raw_good and log[1]["status"] == "validated"
    assert received[1][1]["response_format_retry"]["previous_invalid_response"] == raw_bad
    assert "All five criteria" in received[1][1]["response_format_retry"]["validation_error"]
    assert log[0]["structured_response_request_id"] == log[1]["structured_response_request_id"]
    assert [entry["structured_response_attempt"] for entry in log] == [1, 2]


def test_repeated_invalid_scores_fail_without_defaulting_or_unbounded_calls():
    invalid = valid_evaluation()
    invalid["scores"]["clarity"] = 5
    raw = json.dumps(invalid)
    log, received = [], []
    call = with_schema_retry(scripted_transport([raw, raw], log, received), log, CRITERIA)
    with pytest.raises(ValueError, match="after 2 attempts"):
        call("Evaluate", {"task": "evaluate"})
    assert len(log) == 2
    assert all(entry["status"] == "error" for entry in log)
    assert not log[-1]["schema_retry_scheduled"]
    assert all(json.loads(entry["response_text"])["scores"]["clarity"] == 5 for entry in log)


def test_valid_first_response_does_not_trigger_retry():
    log, received = [], []
    good = valid_evaluation()
    call = with_schema_retry(scripted_transport([json.dumps(good)], log, received), log, CRITERIA)
    assert call("Evaluate", {"task": "evaluate"}) == good
    assert len(received) == 1


@pytest.mark.parametrize("message", ["CUDA out of memory", "Qwen reached its output limit."])
def test_runtime_and_truncation_errors_are_not_retried(message):
    log, received = [], []

    def failure(instructions, payload):
        received.append(payload)
        log.append({"status": "error", "response_text": "partial", "error": message})
        if "output limit" in message:
            raise ValueError(message)
        raise RuntimeError(message)

    call = with_schema_retry(failure, log, CRITERIA)
    with pytest.raises((ValueError, RuntimeError), match=message):
        call("Evaluate", {"task": "evaluate"})
    assert len(received) == 1
