from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = ("id", "category", "difficulty", "prompt", "validation")
V2_FIELDS = (
    "tags",
    "expected_tools",
    "failure_modes",
    "strictness_level",
    "anti_overfit_markers",
)
DETERMINISTIC_CATEGORIES = {"mathematics", "coding", "reasoning_traps", "self_consistency"}


def _validation_is_weak(text: str) -> bool:
    value = (text or "").strip()
    if len(value) < 80:
        return True
    must_count = value.lower().count("must")
    return must_count < 2


def check_dataset(path: str, *, min_total: int, min_category_count: int) -> tuple[int, dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list):
        raise ValueError("Dataset must be a JSON list.")

    duplicate_ids: list[str] = []
    missing_required: list[str] = []
    weak_validation_ids: list[str] = []
    weak_deterministic_ids: list[str] = []
    missing_v2_ids: list[str] = []

    seen: set[str] = set()
    category_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()

    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            missing_required.append(f"non_dict_case_{index}")
            continue

        case_id = str(item.get("id") or "").strip() or f"case_{index}"

        for field in REQUIRED_FIELDS:
            if not str(item.get(field) or "").strip():
                missing_required.append(f"{case_id}:{field}")

        if case_id in seen:
            duplicate_ids.append(case_id)
        seen.add(case_id)

        category = str(item.get("category") or "uncategorized")
        difficulty = str(item.get("difficulty") or "unspecified")
        category_counts[category] += 1
        difficulty_counts[difficulty] += 1

        validation = str(item.get("validation") or "")
        if _validation_is_weak(validation):
            weak_validation_ids.append(case_id)

        if category in DETERMINISTIC_CATEGORIES and not str(item.get("answer") or "").strip():
            weak_deterministic_ids.append(case_id)

        missing_v2 = [field for field in V2_FIELDS if field not in item]
        if missing_v2:
            missing_v2_ids.append(f"{case_id}:{','.join(missing_v2)}")

    sparse_categories = [cat for cat, count in category_counts.items() if count < min_category_count]

    errors = 0
    if len(payload) < min_total:
        errors += 1
    if duplicate_ids:
        errors += 1
    if missing_required:
        errors += 1
    if sparse_categories:
        errors += 1
    if weak_validation_ids:
        errors += 1
    if weak_deterministic_ids:
        errors += 1
    if missing_v2_ids:
        errors += 1

    report: dict[str, Any] = {
        "path": path,
        "cases_total": len(payload),
        "category_counts": dict(sorted(category_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "duplicate_ids": duplicate_ids[:30],
        "missing_required": missing_required[:50],
        "sparse_categories": sparse_categories,
        "weak_validation_ids": weak_validation_ids[:50],
        "weak_deterministic_ids": weak_deterministic_ids[:50],
        "missing_v2_ids": missing_v2_ids[:50],
        "error_count": errors,
    }
    return errors, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QA checks for prompt_train_cases V2 dataset")
    parser.add_argument("--dataset", default="data/prompt_train_cases.json")
    parser.add_argument("--min-total", type=int, default=140)
    parser.add_argument("--min-category-count", type=int, default=8)
    args = parser.parse_args(argv)

    errors, report = check_dataset(
        args.dataset,
        min_total=max(1, int(args.min_total)),
        min_category_count=max(1, int(args.min_category_count)),
    )
    print(json.dumps(report, indent=2))
    if errors > 0:
        print(f"dataset_qa_failed: {errors} failing check group(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
