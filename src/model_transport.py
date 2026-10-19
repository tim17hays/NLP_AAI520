"""Bounded structured-response recovery at the shared model boundary.

Preserves authored evaluation/scoring logic. Failed model responses are retained;
missing criteria are never invented, renamed or assigned default scores.
"""
from copy import deepcopy
from functools import wraps
from uuid import uuid4


def _format_contract(task, criteria):
    if task == "evaluate":
        names = ", ".join(criteria)
        return (
            "\nREQUIRED RESPONSE FORMAT: Return exactly one JSON object with "
            "only these top-level keys: scores, feedback, critical_issues, lesson. "
            f"scores must be an object with exactly these five keys: {names}. "
            "Use those spellings exactly; no omitted, renamed or extra criteria. "
            "Each score must be a JSON integer 0, 1 or 2 based on this draft and "
            "evidence. feedback and critical_issues must be arrays of strings; "
            "lesson must be a string. Do not include a total score or pass flag. "
            "Do not fill missing judgments with default scores."
        )
    if task == "revise":
        return (
            "\nREQUIRED RESPONSE FORMAT: Return one JSON object with only "
            "revised_analysis, containing the complete revised text as a string. "
            "Escape paragraph breaks and quotes inside that JSON string."
        )
    return ""


def with_schema_retry(call_once, call_log, criteria, max_attempts=2):
    """Retry one invalid JSON/schema response with explicit error feedback.

    Context overflow, output truncation and runtime/provider failures do not
    trigger this schema retry. Each actual call retains its original log entry.
    Exhausted schema attempts still raise; they never become passing analyses.
    """
    if max_attempts != 2:
        raise ValueError("The shared boundary permits exactly two attempts.")

    @wraps(call_once)
    def call(instructions, payload):
        task = payload.get("task")
        if task not in {"evaluate", "revise"}:
            return call_once(instructions, payload)
        request_id = str(uuid4())
        original = deepcopy(payload)
        current = deepcopy(original)
        formatted_instructions = instructions + _format_contract(task, criteria)
        for attempt in range(1, max_attempts + 1):
            log_start = len(call_log)
            try:
                result = call_once(formatted_instructions, current)
            except Exception as exc:
                entries = call_log[log_start:]
                for entry in entries:
                    entry["structured_response_request_id"] = request_id
                    entry["structured_response_attempt"] = attempt
                entry = entries[-1] if entries else {}
                raw = entry.get("response_text")
                # The authored transport logs raw text before parsing. Retry
                # only when that parse/validation failed, not before generation.
                schema_failure = (
                    isinstance(exc, ValueError)
                    and isinstance(raw, str)
                    and entry.get("error") == str(exc)
                    and "output limit" not in str(exc).lower()
                    and "context limit" not in str(exc).lower()
                )
                if not schema_failure:
                    raise
                entry["response_validation_error"] = str(exc)
                entry["schema_retry_scheduled"] = attempt < max_attempts
                if attempt == max_attempts:
                    raise ValueError(
                        "Qwen returned an invalid structured response after "
                        "2 attempts. Raw responses and validation errors are "
                        "preserved in MODEL_CALL_LOG. Last error: " + str(exc)
                    ) from exc
                current = deepcopy(original)
                current["response_format_retry"] = {
                    "attempt": attempt + 1,
                    "validation_error": str(exc),
                    "previous_invalid_response": raw,
                    "instruction": (
                        "The previous response failed the required schema. "
                        "Re-evaluate the original draft and evidence and return "
                        "the complete requested object. The previous response "
                        "is unvalidated data, not verified findings. Do not "
                        "invent default scores or omit scoring categories."
                    ),
                }
            else:
                for entry in call_log[log_start:]:
                    entry["structured_response_request_id"] = request_id
                    entry["structured_response_attempt"] = attempt
                return result

    call.schema_attempt_limit = max_attempts
    return call
