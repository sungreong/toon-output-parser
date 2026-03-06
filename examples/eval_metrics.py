from __future__ import annotations

from typing import Any


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if value == "":
        return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        if not value:
            out[prefix or "$"] = value
            return out
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            out.update(_flatten(child, child_prefix))
        return out
    if isinstance(value, list):
        if not value:
            out[prefix or "$"] = value
            return out
        for idx, child in enumerate(value):
            child_prefix = f"{prefix}[{idx}]"
            out.update(_flatten(child, child_prefix))
        return out
    out[prefix or "$"] = value
    return out


def evaluate_output(actual: dict[str, Any], expected: dict[str, Any] | None = None) -> dict[str, Any]:
    flat_actual = _flatten(actual)
    total_leaf = len(flat_actual)
    filled_leaf = sum(0 if _is_empty(v) else 1 for v in flat_actual.values())
    completeness = (filled_leaf / total_leaf) if total_leaf else 1.0

    expected_total = 0
    expected_matched = 0
    if expected is not None:
        flat_expected = _flatten(expected)
        expected_total = len(flat_expected)
        for key, expected_value in flat_expected.items():
            if key in flat_actual and flat_actual[key] == expected_value:
                expected_matched += 1

    exact_match_ratio = (expected_matched / expected_total) if expected_total else None

    return {
        "total_leaf_fields": total_leaf,
        "filled_leaf_fields": filled_leaf,
        "completeness_ratio": completeness,
        "expected_total_fields": expected_total,
        "expected_matched_fields": expected_matched,
        "expected_match_ratio": exact_match_ratio,
    }


def print_evaluation(title: str, actual: dict[str, Any], expected: dict[str, Any] | None = None) -> None:
    metrics = evaluate_output(actual, expected)
    print(f"\n=== {title} ===")
    if expected is not None:
        print("EXPECTED:")
        print(expected)
    print("METRICS:")
    print(
        f"- completeness: {metrics['filled_leaf_fields']}/{metrics['total_leaf_fields']} "
        f"({metrics['completeness_ratio'] * 100:.1f}%)"
    )
    if metrics["expected_match_ratio"] is not None:
        print(
            f"- expected-match: {metrics['expected_matched_fields']}/{metrics['expected_total_fields']} "
            f"({metrics['expected_match_ratio'] * 100:.1f}%)"
        )
