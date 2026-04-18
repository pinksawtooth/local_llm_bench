from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ScoreResult:
    correct: bool
    score: float
    reason: str | None = None


def _normalize_text(value: Any) -> str:
    return str(value).strip().lower()


def _score_exact(predicted: Any, gold: Any) -> ScoreResult:
    correct = _normalize_text(predicted) == _normalize_text(gold)
    return ScoreResult(correct=correct, score=1.0 if correct else 0.0)


def _score_regex(predicted: Any, gold_regex: str) -> ScoreResult:
    matched = bool(re.search(gold_regex, str(predicted)))
    return ScoreResult(
        correct=matched,
        score=1.0 if matched else 0.0,
        reason=None if matched else "regex not matched",
    )


def _score_number(predicted: Any, gold: Any) -> ScoreResult:
    try:
        predicted_num = float(predicted)
        gold_num = float(gold)
    except (TypeError, ValueError):
        return ScoreResult(False, 0.0, "not a number")
    correct = predicted_num == gold_num
    return ScoreResult(
        correct=correct,
        score=1.0 if correct else 0.0,
        reason=None if correct else f"{predicted_num} != {gold_num}",
    )


def _score_json(predicted: Any, gold: Any) -> ScoreResult:
    try:
        predicted_obj = json.loads(predicted) if isinstance(predicted, str) else predicted
    except Exception as exc:  # noqa: BLE001
        return ScoreResult(False, 0.0, f"json decode failed: {exc}")
    correct = predicted_obj == gold
    return ScoreResult(correct=correct, score=1.0 if correct else 0.0)


def score_answer(answer_type: str, predicted: Any, gold: Any) -> ScoreResult:
    normalized = str(answer_type).strip().lower()
    if normalized == "exact":
        return _score_exact(predicted, gold)
    if normalized == "regex":
        return _score_regex(predicted, str(gold))
    if normalized == "number":
        return _score_number(predicted, gold)
    if normalized == "json":
        return _score_json(predicted, gold)
    raise ValueError(f"unsupported answer_type: {answer_type!r}")
