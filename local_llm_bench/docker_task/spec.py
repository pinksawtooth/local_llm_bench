from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

SUPPORTED_ANSWER_TYPES = {"exact", "regex", "number", "json"}


class SpecValidationError(ValueError):
    pass


@dataclass
class Question:
    id: str
    prompt: str
    answer_type: str
    gold_answer: Any
    binary_path: Path | None = None
    binary_ref: str | None = None
    category: str | None = None
    difficulty: str | None = None
    tags: List[str] = field(default_factory=list)
    description: str | None = None


@dataclass
class BenchmarkSpec:
    id: str
    title: str
    description: str | None = None
    questions: List[Question] = field(default_factory=list)


def _normalize_tags(raw: Any, *, index: int) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        tag = raw.strip()
        if not tag:
            raise SpecValidationError(f"questions[{index}].tags は空文字を含められません")
        return [tag]
    if not isinstance(raw, list):
        raise SpecValidationError(f"questions[{index}].tags は文字列または文字列配列である必要があります")
    normalized: List[str] = []
    for tag_index, tag_raw in enumerate(raw):
        if not isinstance(tag_raw, str) or not tag_raw.strip():
            raise SpecValidationError(
                f"questions[{index}].tags[{tag_index}] は空でない文字列である必要があります"
            )
        normalized.append(tag_raw.strip())
    return normalized


def _default_answer_key_path(spec_path: Path) -> Path:
    return spec_path.with_name(f"{spec_path.stem}.answers{spec_path.suffix}")


def _load_answer_key(answer_key_path: Path) -> Dict[str, Any]:
    if not answer_key_path.exists():
        raise SpecValidationError(f"answer key が存在しません: {answer_key_path}")
    data = yaml.safe_load(answer_key_path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SpecValidationError("answer key ファイルは辞書形式である必要があります")
    answers = data.get("answers", data)
    if not isinstance(answers, dict):
        raise SpecValidationError("answer key の answers は辞書である必要があります")
    normalized: Dict[str, Any] = {}
    for key, value in answers.items():
        if not isinstance(key, str) or not key.strip():
            raise SpecValidationError("answer key の question id は空でない文字列である必要があります")
        normalized[key.strip()] = value
    return normalized


def _resolve_repo_relative_path(spec_path: Path, raw_value: str) -> Path:
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate.resolve()

    search_roots = [spec_path.parent, *spec_path.parents]
    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    return (spec_path.parent / candidate).resolve()


def load_spec(spec_path: str | Path, answer_key_path: str | Path | None = None) -> BenchmarkSpec:
    path = Path(spec_path).expanduser().resolve()
    if not path.exists():
        raise SpecValidationError(f"spec ファイルが存在しません: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SpecValidationError("spec ファイルは辞書形式である必要があります")
    raw_questions = data.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise SpecValidationError("questions は1件以上の配列である必要があります")

    resolved_answer_key_path = (
        Path(answer_key_path).expanduser().resolve()
        if answer_key_path is not None
        else _default_answer_key_path(path)
    )
    answer_key = _load_answer_key(resolved_answer_key_path)

    questions: List[Question] = []
    seen_ids: set[str] = set()
    missing_answers: List[str] = []
    for index, raw_question in enumerate(raw_questions):
        if not isinstance(raw_question, dict):
            raise SpecValidationError(f"questions[{index}] は辞書である必要があります")
        question_id = str(raw_question.get("id") or "").strip()
        prompt = str(raw_question.get("prompt") or "").strip()
        answer_type = str(raw_question.get("answer_type") or "").strip().lower()
        if not question_id:
            raise SpecValidationError(f"questions[{index}].id は空でない文字列である必要があります")
        if not prompt:
            raise SpecValidationError(f"questions[{index}].prompt は空でない文字列である必要があります")
        if answer_type not in SUPPORTED_ANSWER_TYPES:
            raise SpecValidationError(
                f"questions[{index}].answer_type は {sorted(SUPPORTED_ANSWER_TYPES)} のいずれかである必要があります"
            )
        if question_id in seen_ids:
            raise SpecValidationError(f"question id が重複しています: {question_id}")
        seen_ids.add(question_id)

        gold_answer = answer_key.get(question_id)
        if gold_answer is None:
            missing_answers.append(question_id)

        binary_ref_raw = raw_question.get("binary_path")
        binary_ref = str(binary_ref_raw).strip() if binary_ref_raw is not None else None
        binary_path = (
            _resolve_repo_relative_path(path, binary_ref)
            if binary_ref
            else None
        )
        if binary_path is not None and not binary_path.exists():
            raise SpecValidationError(f"binary_path が存在しません: {binary_ref}")

        questions.append(
            Question(
                id=question_id,
                prompt=prompt,
                answer_type=answer_type,
                gold_answer=gold_answer,
                binary_path=binary_path,
                binary_ref=binary_ref,
                category=str(raw_question.get("category") or "").strip() or None,
                difficulty=str(raw_question.get("difficulty") or "").strip() or None,
                tags=_normalize_tags(raw_question.get("tags"), index=index),
                description=str(raw_question.get("description") or "").strip() or None,
            )
        )

    extra_answer_ids = sorted(set(answer_key.keys()) - seen_ids)
    if extra_answer_ids:
        raise SpecValidationError(
            "answer key に spec に存在しない question id があります: "
            + ", ".join(extra_answer_ids)
        )
    if missing_answers:
        raise SpecValidationError(
            "answer key でも gold_answer が不足しています: " + ", ".join(missing_answers)
        )

    spec_id = str(data.get("id") or path.stem).strip()
    title = str(data.get("title") or spec_id).strip()
    return BenchmarkSpec(
        id=spec_id,
        title=title,
        description=str(data.get("description") or "").strip() or None,
        questions=questions,
    )
