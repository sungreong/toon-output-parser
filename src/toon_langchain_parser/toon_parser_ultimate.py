# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import io
import json
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

try:
    from pydantic import BaseModel
except Exception as e:  # pragma: no cover
    raise ImportError("pydantic(v2) is required. Install: pip install pydantic>=2") from e


class ToonParserError(ValueError):
    pass


class ToonDecodeError(ToonParserError):
    pass


class PolicyViolationError(ToonParserError):
    pass


class SchemaViolationError(ToonParserError):
    pass


class ModelComplexityError(ToonParserError):
    pass


@dataclass(frozen=True)
class ParserConfig:
    indent_step: int = 2
    complexity_threshold: int = 3
    protect_string_ids: bool = True
    strict_schema: bool = True
    strict_count: bool = False
    allow_tabular_for_flat_objects: bool = True
    instructions_mode: str = "adaptive"  # adaptive | minimal | json
    auto_fallback_to_json: bool = True
    strict_minimal_validation: bool = False
    max_repair_attempts: int = 1
    repair_excerpt_lines: int = 12
    allow_dotted_paths: bool = True
    coerce_object_list_from_inline_scalars: bool = True


@dataclass(frozen=True)
class ComplexityLimits:
    max_nesting_depth: int = 2
    max_array_nesting: int = 1
    max_union_types: int = 1
    max_object_fields: int = 12
    max_array_item_fields: int = 6
    allow_nested_unions: bool = False
    allow_recursive_refs: bool = False
    allow_dynamic_object_keys: bool = False
    allow_array_of_arrays: bool = False


@dataclass
class ComplexityMetrics:
    max_nesting_depth: int = 0
    max_array_nesting: int = 0
    union_type_count: int = 0
    max_object_fields: int = 0
    max_array_item_fields: int = 0
    largest_list_object_width: int = 0
    has_nested_unions: bool = False
    has_recursive_refs: bool = False
    has_dynamic_object_keys: bool = False
    has_array_of_arrays: bool = False
    has_union_inside_array: bool = False
    has_union_inside_object: bool = False
    violation_reasons: List[str] = field(default_factory=list)

    def add_violation(self, reason: str) -> None:
        self.violation_reasons.append(reason)

    def is_within_limits(self, limits: ComplexityLimits) -> bool:
        self.violation_reasons.clear()

        if self.max_nesting_depth > limits.max_nesting_depth:
            self.add_violation(
                f"nesting depth {self.max_nesting_depth} exceeds {limits.max_nesting_depth}"
            )
        if self.max_array_nesting > limits.max_array_nesting:
            self.add_violation(
                f"array nesting {self.max_array_nesting} exceeds {limits.max_array_nesting}"
            )
        if self.union_type_count > limits.max_union_types:
            self.add_violation(
                f"union count {self.union_type_count} exceeds {limits.max_union_types}"
            )
        if self.max_object_fields > limits.max_object_fields:
            self.add_violation(
                f"object field count {self.max_object_fields} exceeds {limits.max_object_fields}"
            )
        if self.max_array_item_fields > limits.max_array_item_fields:
            self.add_violation(
                f"array item field count {self.max_array_item_fields} exceeds {limits.max_array_item_fields}"
            )
        if self.has_nested_unions and not limits.allow_nested_unions:
            self.add_violation("nested unions are not allowed")
        if self.has_recursive_refs and not limits.allow_recursive_refs:
            self.add_violation("recursive references are not allowed")
        if self.has_dynamic_object_keys and not limits.allow_dynamic_object_keys:
            self.add_violation("dynamic object keys are not allowed")
        if self.has_array_of_arrays and not limits.allow_array_of_arrays:
            self.add_violation("array-of-array schemas are not allowed")

        return not self.violation_reasons

    def get_violation_summary(self) -> str:
        if not self.violation_reasons:
            return "No complexity violations"
        return "\n".join(f"- {reason}" for reason in self.violation_reasons)


class ModelComplexityAnalyzer:
    def __init__(self, model: Type[BaseModel]):
        self.model = model
        self.schema = model.model_json_schema()
        self._defs: Dict[str, Any] = {}
        if isinstance(self.schema, dict):
            if isinstance(self.schema.get("$defs"), dict):
                self._defs.update(self.schema["$defs"])
            if isinstance(self.schema.get("definitions"), dict):
                self._defs.update(self.schema["definitions"])
        self.metrics = ComplexityMetrics()
        self._ref_stack: set[str] = set()

    def analyze(self) -> ComplexityMetrics:
        self._analyze_schema(self.schema, depth=0, array_depth=0, path="$", in_union=False, in_array=False)
        return self.metrics

    def recommended_mode(self) -> str:
        metrics = self.analyze()
        if metrics.has_recursive_refs:
            return "json"
        if metrics.has_dynamic_object_keys:
            return "json"
        if metrics.has_nested_unions:
            return "json"
        if metrics.has_union_inside_array:
            return "json"
        if metrics.has_array_of_arrays:
            return "json"
        if metrics.max_nesting_depth >= 4:
            return "json"
        if metrics.largest_list_object_width >= 8:
            return "json"
        return "adaptive"

    def _resolve_ref(self, ref: str) -> Dict[str, Any]:
        if not ref.startswith("#/"):
            return {}
        parts = ref.lstrip("#/").split("/")
        if len(parts) == 2 and parts[0] in ("$defs", "definitions"):
            return self._defs.get(parts[1], {})
        return {}

    def _resolve_schema(self, schema: Any) -> Any:
        cur = schema
        seen: set[str] = set()
        while isinstance(cur, dict) and "$ref" in cur:
            ref = cur.get("$ref")
            if not isinstance(ref, str):
                break
            if ref in seen:
                self.metrics.has_recursive_refs = True
                return cur
            seen.add(ref)
            base = {k: v for k, v in cur.items() if k != "$ref"}
            resolved = dict(self._resolve_ref(ref))
            if not resolved:
                return cur
            if base:
                resolved.update(base)
            cur = resolved
        return cur

    def _analyze_schema(
        self,
        schema: Dict[str, Any],
        depth: int,
        array_depth: int,
        path: str,
        in_union: bool,
        in_array: bool,
    ) -> None:
        if not isinstance(schema, dict):
            return

        if "$ref" in schema:
            ref = schema["$ref"]
            if ref in self._ref_stack:
                self.metrics.has_recursive_refs = True
                return
            self._ref_stack.add(ref)
            self._analyze_schema(self._resolve_ref(ref), depth, array_depth, f"{path}.$ref", in_union, in_array)
            self._ref_stack.discard(ref)
            return

        if "anyOf" in schema or "oneOf" in schema:
            union_items = schema.get("anyOf", schema.get("oneOf", []))
            self.metrics.union_type_count += 1
            if in_union:
                self.metrics.has_nested_unions = True
            if in_array:
                self.metrics.has_union_inside_array = True
            if depth > 0:
                self.metrics.has_union_inside_object = True
            for idx, item in enumerate(union_items):
                if isinstance(item, dict):
                    self._analyze_schema(item, depth, array_depth, f"{path}.union[{idx}]", True, in_array)
            return

        resolved = self._resolve_schema(schema)
        schema_type = resolved.get("type")

        if schema_type == "array":
            items_schema = resolved.get("items", {}) or {}
            next_array_depth = array_depth + 1
            self.metrics.max_array_nesting = max(self.metrics.max_array_nesting, next_array_depth)
            items_resolved = self._resolve_schema(items_schema)
            if isinstance(items_resolved, dict) and items_resolved.get("type") == "array":
                self.metrics.has_array_of_arrays = True
            self._analyze_schema(items_schema, depth, next_array_depth, f"{path}[]", in_union, True)
            return

        if schema_type == "object" or "properties" in resolved or "additionalProperties" in resolved:
            props = resolved.get("properties", {}) or {}
            field_count = len(props)
            if array_depth > 0:
                self.metrics.max_array_item_fields = max(self.metrics.max_array_item_fields, field_count)
                self.metrics.largest_list_object_width = max(self.metrics.largest_list_object_width, field_count)
            else:
                self.metrics.max_object_fields = max(self.metrics.max_object_fields, field_count)
            self.metrics.max_nesting_depth = max(self.metrics.max_nesting_depth, depth)

            additional = resolved.get("additionalProperties")
            if additional is True or isinstance(additional, dict):
                self.metrics.has_dynamic_object_keys = True
                if isinstance(additional, dict):
                    self._analyze_schema(additional, depth + 1, array_depth, f"{path}.*", in_union, in_array)

            for field_name, field_schema in props.items():
                self._analyze_schema(field_schema, depth + 1, array_depth, f"{path}.{field_name}", in_union, False)

    @staticmethod
    def validate_for_minimal_mode(
        model: Type[BaseModel], limits: Optional[ComplexityLimits] = None
    ) -> Tuple[bool, ComplexityMetrics]:
        if limits is None:
            limits = ComplexityLimits()
        analyzer = ModelComplexityAnalyzer(model)
        metrics = analyzer.analyze()
        return metrics.is_within_limits(limits), metrics


class ToonIntelligence:
    @staticmethod
    def _resolve_ref(ref: str, defs: Dict[str, Any]) -> Dict[str, Any]:
        if not ref.startswith("#/"):
            return {}
        parts = ref.lstrip("#/").split("/")
        if len(parts) == 2 and parts[0] in ("$defs", "definitions"):
            return defs.get(parts[1], {})
        return {}

    @staticmethod
    def _resolve_schema(schema: Any, defs: Dict[str, Any], seen: Optional[set[str]] = None) -> Any:
        cur = schema
        seen = seen or set()
        while isinstance(cur, dict) and "$ref" in cur:
            ref = cur.get("$ref")
            if not isinstance(ref, str) or ref in seen:
                return cur
            seen.add(ref)
            base = {k: v for k, v in cur.items() if k != "$ref"}
            resolved = dict(ToonIntelligence._resolve_ref(ref, defs))
            if not resolved:
                return cur
            if base:
                resolved.update(base)
            cur = resolved
        return cur

    @staticmethod
    def _unwrap_nullable(schema: Any, defs: Dict[str, Any]) -> Any:
        schema = ToonIntelligence._resolve_schema(schema, defs)
        if isinstance(schema, dict):
            for key in ("anyOf", "oneOf"):
                if key in schema and isinstance(schema[key], list):
                    non_null = [
                        item
                        for item in schema[key]
                        if not (isinstance(item, dict) and item.get("type") == "null")
                    ]
                    if len(non_null) == 1:
                        return ToonIntelligence._resolve_schema(non_null[0], defs)
        return schema

    @staticmethod
    def _items(schema: Dict[str, Any], defs: Dict[str, Any]) -> Dict[str, Any]:
        resolved = ToonIntelligence._unwrap_nullable(schema, defs)
        return ToonIntelligence._unwrap_nullable(resolved.get("items", {}) or {}, defs)

    @staticmethod
    def _scalar_type_name(schema: Dict[str, Any], defs: Dict[str, Any]) -> str:
        resolved = ToonIntelligence._unwrap_nullable(schema, defs)
        return resolved.get("type", "string")

    @staticmethod
    def _dummy_scalar(schema: Dict[str, Any], defs: Dict[str, Any]) -> str:
        value_type = ToonIntelligence._scalar_type_name(schema, defs)
        if value_type == "integer":
            return "1"
        if value_type == "number":
            return "0.1"
        if value_type == "boolean":
            return "true"
        return "text"

    @staticmethod
    def is_complex_array(field_schema: Dict[str, Any], threshold: int, defs: Optional[Dict[str, Any]] = None) -> bool:
        defs = defs or {}
        resolved = ToonIntelligence._unwrap_nullable(field_schema, defs)
        if resolved.get("type") != "array":
            return False
        items = ToonIntelligence._items(resolved, defs)
        if items.get("type") == "array":
            return True
        if "anyOf" in items or "oneOf" in items or "$ref" in items:
            return True
        props = items.get("properties", {}) or {}
        if len(props) > threshold:
            return True
        for value in props.values():
            child = ToonIntelligence._unwrap_nullable(value, defs)
            if child.get("type") in ("object", "array") or "$ref" in child or "anyOf" in child or "oneOf" in child:
                return True
        return False

    @staticmethod
    def is_flat_object_array(field_schema: Dict[str, Any], defs: Optional[Dict[str, Any]] = None) -> bool:
        defs = defs or {}
        resolved = ToonIntelligence._unwrap_nullable(field_schema, defs)
        if resolved.get("type") != "array":
            return False
        items = ToonIntelligence._items(resolved, defs)
        props = items.get("properties", {}) or {}
        if not props:
            return False
        for value in props.values():
            child = ToonIntelligence._unwrap_nullable(value, defs)
            if child.get("type") not in ("string", "integer", "number", "boolean"):
                return False
        return True

    @staticmethod
    def _schema_summary(model: Type[BaseModel]) -> List[str]:
        schema = model.model_json_schema()
        defs: Dict[str, Any] = {}
        if isinstance(schema.get("$defs"), dict):
            defs.update(schema["$defs"])
        if isinstance(schema.get("definitions"), dict):
            defs.update(schema["definitions"])

        props = schema.get("properties", {}) or {}
        required = set(schema.get("required", []))

        lines: List[str] = []
        for name, field_schema in list(props.items())[:8]:
            resolved = ToonIntelligence._unwrap_nullable(field_schema, defs)
            kind = resolved.get("type", "object" if "properties" in resolved else "any")
            suffix = " required" if name in required else " optional"
            if kind == "array":
                item = ToonIntelligence._items(resolved, defs)
                item_kind = item.get("type", "object" if "properties" in item else "any")
                lines.append(f"- {name}: array<{item_kind}>{suffix}")
            elif kind == "object" or "properties" in resolved:
                child_keys = ", ".join(list((resolved.get("properties", {}) or {}).keys())[:3])
                if child_keys:
                    lines.append(f"- {name}: object<{child_keys}>{suffix}")
                else:
                    lines.append(f"- {name}: object{suffix}")
            else:
                lines.append(f"- {name}: {kind}{suffix}")
        return lines

    @staticmethod
    def _build_flat_example(schema: Dict[str, Any], defs: Dict[str, Any]) -> List[str]:
        props = schema.get("properties", {}) or {}
        required = list(schema.get("required", []))
        names = required[:3] or list(props.keys())[:3]
        lines: List[str] = []
        for name in names:
            field_schema = ToonIntelligence._unwrap_nullable(props.get(name, {}), defs)
            if field_schema.get("type") == "array":
                item = ToonIntelligence._items(field_schema, defs)
                if item.get("type") in ("string", "integer", "number", "boolean"):
                    item_value = ToonIntelligence._dummy_scalar(item, defs)
                    lines.append(f"{name}[2]: {item_value},{item_value}")
                else:
                    lines.append(f"{name}: []")
            else:
                lines.append(f"{name}: {ToonIntelligence._dummy_scalar(field_schema, defs)}")
        return lines or ["name: text"]

    @staticmethod
    def _find_nested_object_field(schema: Dict[str, Any], defs: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
        for name, field_schema in (schema.get("properties", {}) or {}).items():
            resolved = ToonIntelligence._unwrap_nullable(field_schema, defs)
            if resolved.get("type") == "object" or "properties" in resolved:
                return name, resolved
        return None

    @staticmethod
    def _find_object_list_field(schema: Dict[str, Any], defs: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
        for name, field_schema in (schema.get("properties", {}) or {}).items():
            resolved = ToonIntelligence._unwrap_nullable(field_schema, defs)
            if resolved.get("type") != "array":
                continue
            items = ToonIntelligence._items(resolved, defs)
            if items.get("type") == "object" or "properties" in items or "$ref" in items:
                return name, items
        return None

    @staticmethod
    def _find_recursive_field(model: Type[BaseModel], schema: Dict[str, Any]) -> Optional[str]:
        model_name = model.__name__
        for name, field_schema in (schema.get("properties", {}) or {}).items():
            if field_schema.get("type") != "array":
                continue
            items = field_schema.get("items", {}) or {}
            ref = items.get("$ref")
            if isinstance(ref, str) and ref.endswith(f"/{model_name}"):
                return name
        return None

    @staticmethod
    def _build_nested_example(field_name: str, field_schema: Dict[str, Any], defs: Dict[str, Any]) -> List[str]:
        lines = [f"{field_name}:"]
        child_props = field_schema.get("properties", {}) or {}
        for child_name, child_schema in list(child_props.items())[:2]:
            resolved = ToonIntelligence._unwrap_nullable(child_schema, defs)
            lines.append(f"  {child_name}: {ToonIntelligence._dummy_scalar(resolved, defs)}")
        return lines

    @staticmethod
    def _build_object_list_example(field_name: str, item_schema: Dict[str, Any], defs: Dict[str, Any]) -> List[str]:
        lines = [f"{field_name}:"]
        child_props = item_schema.get("properties", {}) or {}
        first = True
        for child_name, child_schema in list(child_props.items())[:3]:
            resolved = ToonIntelligence._unwrap_nullable(child_schema, defs)
            prefix = "  - " if first else "    "
            if resolved.get("type") == "array":
                item = ToonIntelligence._items(resolved, defs)
                if item.get("type") in ("string", "integer", "number", "boolean"):
                    item_value = ToonIntelligence._dummy_scalar(item, defs)
                    lines.append(f"{prefix}{child_name}[2]: {item_value},{item_value}")
                else:
                    lines.append(f"{prefix}{child_name}: []")
            else:
                lines.append(f"{prefix}{child_name}: {ToonIntelligence._dummy_scalar(resolved, defs)}")
            first = False
        if len(lines) == 1:
            lines.extend(["  - name: text", "    value: text"])
        return lines

    @staticmethod
    def _build_recursive_example(field_name: str) -> List[str]:
        return [
            f"{field_name}:",
            "  - name: text",
            "    value: text",
            f"    {field_name}: []",
        ]

    @staticmethod
    def _build_invalid_examples() -> List[str]:
        return [
            "Invalid patterns (do not copy):",
            "- line without colon",
            "- list item without a space after '-'",
            "- JSON object literal inside TOON value",
        ]

    @staticmethod
    def _is_nullable_schema(schema: Dict[str, Any], defs: Dict[str, Any]) -> bool:
        resolved = ToonIntelligence._resolve_schema(schema, defs)
        if not isinstance(resolved, dict):
            return False
        if resolved.get("type") == "null":
            return True
        for key in ("anyOf", "oneOf"):
            union_items = resolved.get(key)
            if isinstance(union_items, list):
                for item in union_items:
                    if isinstance(item, dict) and ToonIntelligence._resolve_schema(item, defs).get("type") == "null":
                        return True
        return False

    @staticmethod
    def _build_non_null_required_rules(schema: Dict[str, Any], defs: Dict[str, Any]) -> List[str]:
        props = schema.get("properties", {}) or {}
        required = set(schema.get("required", []))
        lines: List[str] = []
        for name in props.keys():
            if name not in required:
                continue
            field_schema = ToonIntelligence._resolve_schema(props.get(name, {}), defs)
            field_type = field_schema.get("type")
            if field_type not in ("string", "integer", "number", "boolean"):
                continue
            if ToonIntelligence._is_nullable_schema(props.get(name, {}), defs):
                continue
            lines.append(f"- {name}: must not be null")
        return lines

    @staticmethod
    def _empty_literal(schema: Dict[str, Any], defs: Dict[str, Any]) -> str:
        resolved = ToonIntelligence._unwrap_nullable(schema, defs)
        value_type = resolved.get("type")
        if value_type == "string":
            return '""'
        if value_type in ("integer", "number", "boolean"):
            return "null"
        if value_type == "array":
            return "[]"
        if value_type == "object" or "properties" in resolved:
            return "{}"
        return "null"

    @staticmethod
    def _build_typed_empty_examples(schema: Dict[str, Any], defs: Dict[str, Any]) -> List[str]:
        props = schema.get("properties", {}) or {}
        required = set(schema.get("required", []))
        optional_fields = [name for name in props.keys() if name not in required]
        lines: List[str] = []
        for name in optional_fields[:6]:
            literal = ToonIntelligence._empty_literal(props.get(name, {}), defs)
            lines.append(f"- {name}: {literal}")
        return lines

    @staticmethod
    def build_schema_aware_prompt(
        model: Type[BaseModel], cfg: ParserConfig = ParserConfig(), concise: bool = False
    ) -> str:
        schema = model.model_json_schema()
        defs: Dict[str, Any] = {}
        if isinstance(schema.get("$defs"), dict):
            defs.update(schema["$defs"])
        if isinstance(schema.get("definitions"), dict):
            defs.update(schema["definitions"])

        analyzer = ModelComplexityAnalyzer(model)
        metrics = analyzer.analyze()

        sections: List[str] = [
            "Return TOON only.",
            "Rules:",
            "- every line must contain a colon",
            "- use 2 spaces for indentation; never use tabs",
            "- arrays of scalars use field[N]: a,b,c",
            "- arrays of objects use dash list notation under the field",
            "- do not emit JSON objects, JSON arrays, markdown, or prose",
            "- required and optional fields must be emitted; do not omit fields",
            "- use typed empty placeholders when value is missing",
            '- string -> ""',
            "- number/integer/boolean -> null",
            "- array -> []",
            "- object -> {}",
            "",
            "Schema summary:",
        ]
        sections.extend(ToonIntelligence._schema_summary(model))

        typed_empty_examples = ToonIntelligence._build_typed_empty_examples(schema, defs)
        if typed_empty_examples:
            sections.extend(["", "Typed empty examples for optional fields:"])
            sections.extend(typed_empty_examples)

        non_null_rules = ToonIntelligence._build_non_null_required_rules(schema, defs)
        if non_null_rules:
            sections.extend(["", "Required non-null scalar fields:"])
            sections.extend(non_null_rules)

        sections.extend(["", "Flat example:"])
        sections.extend(ToonIntelligence._build_flat_example(schema, defs))

        nested = ToonIntelligence._find_nested_object_field(schema, defs)
        if nested is not None:
            sections.extend(["", "Nested object example:"])
            sections.extend(ToonIntelligence._build_nested_example(nested[0], nested[1], defs))

        object_list = ToonIntelligence._find_object_list_field(schema, defs)
        if object_list is not None:
            sections.extend(["", "List-of-object example:"])
            sections.extend(ToonIntelligence._build_object_list_example(object_list[0], object_list[1], defs))

        recursive_field = ToonIntelligence._find_recursive_field(model, schema)
        if recursive_field is not None:
            sections.extend(["", "Recursive children pattern:"])
            sections.extend(ToonIntelligence._build_recursive_example(recursive_field))

        if not concise:
            sections.extend(["", *ToonIntelligence._build_invalid_examples()])
            sections.extend(
                [
                    "",
                    "Checklist:",
                    "- every required top-level field is present",
                    "- list items keep consistent indentation",
                    "- object fields stay nested; they are not JSON strings",
                ]
            )

        if metrics.has_recursive_refs and not concise:
            sections.extend(["", "Note:", "- recursive schemas may fall back to JSON when TOON is not cost-effective"])

        return "\n".join(sections).strip()

    @staticmethod
    def build_adaptive_prompt(model: Type[BaseModel], cfg: ParserConfig = ParserConfig()) -> str:
        return ToonIntelligence.build_schema_aware_prompt(model, cfg, concise=False)

    @staticmethod
    def build_minimal_prompt(model: Type[BaseModel], cfg: ParserConfig = ParserConfig()) -> str:
        return ToonIntelligence.build_schema_aware_prompt(model, cfg, concise=True)


class ToonParser:
    _TABULAR_HEADER_RE = re.compile(r"^(?P<name>[A-Za-z0-9_]+)\[(?P<n>\d+),(?:#)?\]\{(?P<cols>[^}]+)\}:\s*$")
    _SCALAR_LIST_RE = re.compile(r"^(?P<name>[A-Za-z0-9_]+)\[(?P<n>\d+)\]:\s*(?P<body>.*)$")
    _INDEXED_ITEM_RE = re.compile(r"^(?P<name>[A-Za-z0-9_]+)\[(?P<idx>\d+)\]:(?:\s*)$")
    _CODE_FENCE_RE = re.compile(r"```(?:json|toon)?\s*(.*?)\s*```", flags=re.DOTALL)

    def __init__(self, model: Type[BaseModel], cfg: ParserConfig = ParserConfig()):
        self.model = model
        self.cfg = cfg
        self.schema = model.model_json_schema()
        self.root_schema = self.schema
        self._defs: Dict[str, Any] = {}
        if isinstance(self.root_schema, dict):
            if isinstance(self.root_schema.get("$defs"), dict):
                self._defs.update(self.root_schema["$defs"])
            if isinstance(self.root_schema.get("definitions"), dict):
                self._defs.update(self.root_schema["definitions"])
        self._complexity_metrics = ModelComplexityAnalyzer(model).analyze()
        self._effective_mode, self._mode_reason = self._select_mode()
        if self.cfg.instructions_mode == "minimal":
            self._validate_minimal_mode_compatibility()

    def _select_mode(self) -> Tuple[str, str]:
        requested = self.cfg.instructions_mode
        analyzer = ModelComplexityAnalyzer(self.model)
        recommended = analyzer.recommended_mode()
        metrics = analyzer.metrics

        if requested == "json":
            return "json", "explicit json mode"
        if requested == "minimal":
            return "minimal", "explicit minimal mode"
        if recommended == "json":
            reason_bits: List[str] = []
            if metrics.has_recursive_refs:
                reason_bits.append("recursive refs")
            if metrics.has_dynamic_object_keys:
                reason_bits.append("dynamic object keys")
            if metrics.has_nested_unions or metrics.has_union_inside_array:
                reason_bits.append("nested unions")
            if metrics.has_array_of_arrays:
                reason_bits.append("array-of-arrays")
            if metrics.max_nesting_depth >= 4:
                reason_bits.append(f"depth={metrics.max_nesting_depth}")
            if metrics.largest_list_object_width >= 8:
                reason_bits.append(f"wide list objects={metrics.largest_list_object_width}")
            return "json", ", ".join(reason_bits) or "schema too complex for TOON"
        return "adaptive", "schema is TOON-compatible"

    def _validate_minimal_mode_compatibility(self) -> None:
        limits = ComplexityLimits(
            max_nesting_depth=2,
            max_array_nesting=1,
            max_union_types=1,
            max_object_fields=12,
            max_array_item_fields=6,
            allow_nested_unions=False,
            allow_recursive_refs=False,
            allow_dynamic_object_keys=False,
            allow_array_of_arrays=False,
        )
        is_valid, metrics = ModelComplexityAnalyzer.validate_for_minimal_mode(self.model, limits)
        if is_valid:
            return

        summary = metrics.get_violation_summary()
        if self.cfg.strict_minimal_validation:
            raise ModelComplexityError(
                f"minimal mode is not compatible with {self.model.__name__}:\n{summary}"
            )

        if self.cfg.auto_fallback_to_json:
            self._effective_mode = "json"
            self._mode_reason = "minimal mode auto-fallback: " + ", ".join(metrics.violation_reasons)
            warnings.warn(
                f"{self.model.__name__} exceeded minimal-mode limits; falling back to JSON.\n{summary}",
                UserWarning,
                stacklevel=3,
            )

    def get_effective_mode(self) -> str:
        return self._effective_mode

    def get_mode_reason(self) -> str:
        return self._mode_reason

    def get_format_instructions(self) -> str:
        if self._effective_mode == "minimal":
            return ToonIntelligence.build_minimal_prompt(self.model, self.cfg)
        if self._effective_mode == "json":
            return self._get_json_schema_instructions()
        return ToonIntelligence.build_adaptive_prompt(self.model, self.cfg)

    def get_toon_format_instructions(self) -> str:
        concise = self.cfg.instructions_mode == "minimal"
        return ToonIntelligence.build_schema_aware_prompt(self.model, self.cfg, concise=concise)

    def _get_json_schema_instructions(self) -> str:
        schema_str = json.dumps(self.model.model_json_schema(), ensure_ascii=False, indent=2)
        return (
            "You must output ONLY a valid JSON object. Your response must start with { and end with }.\n\n"
            f"Schema:\n{schema_str}\n\n"
            "Rules:\n"
            "- do not use code fences\n"
            "- do not add any explanation\n"
            "- output pure JSON only"
        )

    def build_repair_prompt(self, raw_output: str, error: Exception | str) -> str:
        excerpt_lines = self._normalize_lines(raw_output)[: self.cfg.repair_excerpt_lines]
        excerpt = "\n".join(excerpt_lines) if excerpt_lines else "<empty output>"
        error_text = self._compact_error(error)
        return (
            "Repair the TOON output. Return corrected TOON only.\n\n"
            f"Parser error: {error_text}\n\n"
            "Bad output excerpt:\n"
            f"```toon\n{excerpt}\n```\n\n"
            "Target format:\n"
            f"{self.get_toon_format_instructions()}"
        )

    def build_json_retry_prompt(self, error: Exception | str) -> str:
        error_text = self._compact_error(error)
        return (
            "TOON parsing was not reliable for this output. Return valid JSON only.\n\n"
            f"Previous TOON parse error: {error_text}\n\n"
            f"{self._get_json_schema_instructions()}"
        )

    def parse_with_recovery(
        self,
        text: str,
        repair_callback: Optional[Callable[[str], str]] = None,
        json_callback: Optional[Callable[[str], str]] = None,
    ) -> BaseModel:
        try:
            return self.parse(text)
        except Exception as first_error:
            if self._effective_mode == "json":
                raise

            repaired_error = first_error
            if repair_callback and self.cfg.max_repair_attempts > 0:
                repair_prompt = self.build_repair_prompt(text, first_error)
                repaired_text = repair_callback(repair_prompt)
                try:
                    return self.parse(repaired_text)
                except Exception as second_error:
                    repaired_error = second_error

            if self.cfg.auto_fallback_to_json and json_callback is not None:
                json_prompt = self.build_json_retry_prompt(repaired_error)
                json_text = json_callback(json_prompt)
                return self._validate_model(self._decode_json(json_text))

            raise repaired_error

    def decode(self, text: str) -> Dict[str, Any]:
        return self._decode_to_obj(text)

    def parse(self, text: str) -> BaseModel:
        return self._validate_model(self._decode_to_obj(text))

    def _validate_model(self, obj: Dict[str, Any]) -> BaseModel:
        return self.model.model_validate(obj)

    def _decode_to_obj(self, text: str) -> Dict[str, Any]:
        if self._effective_mode == "json":
            return self._decode_json(text)

        lines = self._normalize_lines(text)
        if not lines:
            raise ToonParserError("Empty input. Model produced no parsable TOON.")
        obj, next_i = self._parse_object(lines, 0, 0, self.schema, path="$")
        if next_i < len(lines):
            rest = [line for line in lines[next_i:] if line.strip()]
            if rest:
                raise ToonParserError(f"Unparsed tail remains at line {next_i + 1}: {rest[0]!r}")
        return obj

    def _decode_json(self, text: str) -> Dict[str, Any]:
        payload = self._extract_payload_text(text)
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError as e:
            raise ToonDecodeError(f"JSON parsing failed: {e}") from e
        if not isinstance(obj, dict):
            raise ToonDecodeError("JSON root must be an object")
        return obj

    def _normalize_lines(self, text: str) -> List[str]:
        payload = self._extract_payload_text(text)
        raw = payload.splitlines()
        return [line.rstrip() for line in raw if line.strip()]

    def _extract_payload_text(self, text: str) -> str:
        payload = (text or "").strip()
        lines = payload.splitlines()

        # Unwrap top/bottom fence lines repeatedly (handles nested duplicated fences).
        for _ in range(4):
            while lines and not lines[0].strip():
                lines.pop(0)
            while lines and not lines[-1].strip():
                lines.pop()
            if not lines:
                break
            if lines[0].strip().startswith("```") and lines[-1].strip().startswith("```"):
                lines = lines[1:-1]
                continue
            break

        cleaned: List[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                continue
            cleaned.append(line)

        while cleaned and cleaned[0].strip().lower() in ("toon", "json"):
            cleaned.pop(0)
        while cleaned and cleaned[-1].strip().lower() in ("toon", "json"):
            cleaned.pop()

        return "\n".join(cleaned).strip()

    def _count_indent(self, line: str) -> int:
        if line.startswith("\t"):
            raise ToonParserError("Tabs are not allowed. Use spaces only.")
        indent = len(line) - len(line.lstrip(" "))
        if "\t" in line[:indent]:
            raise ToonParserError("Tabs are not allowed. Use spaces only.")
        return indent

    def _split_kv(self, stripped: str) -> Tuple[str, Optional[str]]:
        if ":" not in stripped:
            raise ToonParserError(f"Every TOON line must contain ':': {stripped!r}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        if not key:
            raise ToonParserError(f"Missing key before ':': {stripped!r}")
        value = value.strip()
        return key, value if value != "" else None

    def _parse_scalar(self, raw: str, expected_schema: Optional[Dict[str, Any]] = None) -> Any:
        value = raw.strip()
        if value == "null":
            return None
        if value == "{}":
            return {}
        if value == "[]":
            return []
        if len(value) >= 2 and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
            value = value[1:-1]

        expected_type = (expected_schema or {}).get("type")
        if expected_type == "array":
            items_schema = self._resolve_nullable((expected_schema or {}).get("items", {}) or {})
            item_type = (items_schema or {}).get("type")
            if item_type in ("string", "integer", "number", "boolean", None):
                parts = [part.strip() for part in self._csv_row(value)]
                parts = [part for part in parts if part != ""]
                if not parts:
                    return []
                return [self._parse_scalar(part, items_schema) for part in parts]
            if item_type == "object" and self.cfg.coerce_object_list_from_inline_scalars:
                object_key = self._best_inline_object_key(items_schema)
                if object_key is None:
                    raise ToonParserError(
                        "Inline scalar value cannot be coerced to list[object]. Use dash list notation."
                    )
                parts = [part.strip() for part in self._csv_row(value)]
                parts = [part for part in parts if part != ""]
                if not parts:
                    return []
                key_schema = self._resolve_nullable((items_schema.get("properties", {}) or {}).get(object_key))
                return [{object_key: self._parse_scalar(part, key_schema)} for part in parts]
            raise ToonParserError(
                "Inline scalar value cannot be coerced to list[object]. Use dash list notation."
            )

        if expected_type == "string" and self.cfg.protect_string_ids:
            return value
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        if re.fullmatch(r"-?\d+", value):
            if expected_type == "string":
                return value
            if expected_type == "number":
                return float(value)
            return int(value)
        if re.fullmatch(r"-?\d+\.\d+", value):
            return float(value)
        return value

    def _empty_value_for_schema(self, field_schema: Optional[Dict[str, Any]]) -> Any:
        schema = self._resolve_nullable(field_schema or {})
        schema_type = (schema or {}).get("type")
        if schema_type == "array":
            return []
        if schema_type == "object" or self._schema_allows_additional(schema):
            return {}
        if schema_type == "string":
            return ""
        # For non-string scalars, keep None so Pydantic validation decides.
        return None

    def _best_inline_object_key(self, item_schema: Dict[str, Any]) -> Optional[str]:
        props = self._get_props(item_schema)
        if not props:
            return None

        preferred = ("name", "value", "title", "text", "summary", "label", "id")
        for key in preferred:
            if key in props:
                key_schema = self._resolve_nullable(props[key])
                if (key_schema or {}).get("type") in ("string", None):
                    return key

        scalar_candidates: List[str] = []
        for key, key_schema_raw in props.items():
            key_schema = self._resolve_nullable(key_schema_raw)
            if (key_schema or {}).get("type") in ("string", "integer", "number", "boolean", None):
                scalar_candidates.append(key)
        if len(scalar_candidates) == 1:
            return scalar_candidates[0]
        return None

    def _csv_row(self, row_line: str) -> List[str]:
        return next(csv.reader(io.StringIO(row_line), skipinitialspace=True), [])

    def _resolve_ref(self, ref: str) -> Dict[str, Any]:
        if not ref.startswith("#/"):
            raise ToonDecodeError(f"Unsupported $ref: {ref}")
        parts = ref.lstrip("#/").split("/")
        if len(parts) == 2 and parts[0] in ("$defs", "definitions"):
            target = self._defs.get(parts[1])
            if not isinstance(target, dict):
                raise ToonDecodeError(f"$ref not found: {ref}")
            return target
        raise ToonDecodeError(f"Unsupported $ref path: {ref}")

    def _resolve_schema(self, schema: Any) -> Any:
        cur = schema
        seen: set[str] = set()
        while isinstance(cur, dict) and "$ref" in cur:
            ref = cur.get("$ref")
            if not isinstance(ref, str):
                break
            if ref in seen:
                raise ToonDecodeError(f"Cyclic $ref detected: {ref}")
            seen.add(ref)
            base = {k: v for k, v in cur.items() if k != "$ref"}
            resolved = dict(self._resolve_ref(ref))
            if base:
                resolved.update(base)
            cur = resolved
        return cur

    def _unwrap_nullable(self, schema: Any) -> Any:
        schema = self._resolve_schema(schema)
        if isinstance(schema, dict):
            for key in ("anyOf", "oneOf"):
                if key in schema and isinstance(schema[key], list):
                    non_null = [
                        item
                        for item in schema[key]
                        if not (isinstance(item, dict) and item.get("type") == "null")
                    ]
                    if len(non_null) == 1:
                        return self._resolve_schema(non_null[0])
        return schema

    def _resolve_nullable(self, schema: Any) -> Any:
        return self._unwrap_nullable(schema)

    def _get_props(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        return schema.get("properties", {}) or {}

    def _schema_allows_additional(self, schema: Optional[Dict[str, Any]]) -> bool:
        schema = schema or {}
        if schema.get("type") != "object":
            return False
        additional = schema.get("additionalProperties")
        if additional is True or isinstance(additional, dict):
            return True
        if additional is None and not (schema.get("properties") or {}):
            return True
        return False

    def _apply_dotted_path(
        self,
        out: Dict[str, Any],
        dotted_key: str,
        value: Optional[str],
        props: Dict[str, Any],
        allow_additional: bool,
        path: str,
    ) -> bool:
        if "." not in dotted_key or not self.cfg.allow_dotted_paths:
            return False

        segments = [seg.strip() for seg in dotted_key.split(".") if seg.strip()]
        if len(segments) < 2:
            return False

        current_out = out
        current_props = props
        current_allow_additional = allow_additional
        current_path = path

        for idx, seg in enumerate(segments):
            is_last = idx == len(segments) - 1
            seg_schema = self._resolve_nullable(current_props.get(seg)) if current_props else None

            if seg_schema is None and self.cfg.strict_schema and not current_allow_additional:
                raise SchemaViolationError(f"Unknown field '{dotted_key}' at {path}")

            if is_last:
                if value is None:
                    current_out[seg] = self._empty_value_for_schema(seg_schema)
                else:
                    current_out[seg] = self._parse_scalar(value, seg_schema)
                return True

            existing = current_out.get(seg)
            if existing is None:
                current_out[seg] = {}
                existing = current_out[seg]
            if not isinstance(existing, dict):
                raise ToonParserError(
                    f"Cannot expand dotted path '{dotted_key}' because {current_path}.{seg} is not an object"
                )

            current_out = existing
            current_path = f"{current_path}.{seg}"

            if isinstance(seg_schema, dict):
                current_props = self._get_props(seg_schema)
                current_allow_additional = self._schema_allows_additional(seg_schema)
            else:
                current_props = {}
                current_allow_additional = True

        return True

    def _parse_object(
        self, lines: List[str], i: int, indent: int, schema: Dict[str, Any], path: str
    ) -> Tuple[Dict[str, Any], int]:
        out: Dict[str, Any] = {}
        props = self._get_props(schema)
        allow_additional = self._schema_allows_additional(schema)
        indexed_buffers: Dict[str, Dict[int, Dict[str, Any]]] = {}

        while i < len(lines):
            line = lines[i]
            cur_indent = self._count_indent(line)
            if cur_indent < indent:
                break
            if cur_indent > indent:
                raise ToonParserError(f"Unexpected indent at {path} (line {i + 1})")

            stripped = line.strip()

            tabular_match = self._TABULAR_HEADER_RE.match(stripped)
            if tabular_match:
                field_name = tabular_match.group("name")
                field_schema = self._resolve_nullable(props.get(field_name))
                if field_schema is None and not allow_additional and self.cfg.strict_schema:
                    raise SchemaViolationError(f"Unknown field '{field_name}' at {path}")
                value, i = self._parse_tabular(lines, i, indent, field_name, field_schema, f"{path}.{field_name}")
                out[field_name] = value
                continue

            scalar_list_match = self._SCALAR_LIST_RE.match(stripped)
            if scalar_list_match:
                field_name = scalar_list_match.group("name")
                expected_count = int(scalar_list_match.group("n"))
                body = scalar_list_match.group("body")
                field_schema = self._resolve_nullable(props.get(field_name))
                if field_schema is None and not allow_additional and self.cfg.strict_schema:
                    raise SchemaViolationError(f"Unknown field '{field_name}' at {path}")
                items_schema = (field_schema or {}).get("items", {}) if (field_schema or {}).get("type") == "array" else None
                parts = [part.strip() for part in body.split(",")] if body else []
                parts = [part for part in parts if part != ""]
                if self.cfg.strict_count and len(parts) != expected_count:
                    raise SchemaViolationError(
                        f"{path}.{field_name}: expected {expected_count} items, got {len(parts)}"
                    )
                out[field_name] = [self._parse_scalar(part, items_schema) for part in parts]
                i += 1
                continue

            indexed_match = self._INDEXED_ITEM_RE.match(stripped)
            if indexed_match:
                field_name = indexed_match.group("name")
                idx = int(indexed_match.group("idx"))
                field_schema = self._resolve_nullable(props.get(field_name))
                if field_schema is None and not allow_additional and self.cfg.strict_schema:
                    raise SchemaViolationError(f"Unknown field '{field_name}' at {path}")
                if (field_schema or {}).get("type") != "array":
                    raise SchemaViolationError(f"{path}.{field_name}[{idx}] used indexed notation for a non-array field")
                items_schema = self._resolve_nullable((field_schema or {}).get("items", {}) or {})
                i += 1
                obj, i = self._parse_object(lines, i, indent + self.cfg.indent_step, items_schema, f"{path}.{field_name}[{idx}]")
                indexed_buffers.setdefault(field_name, {})[idx] = obj
                continue

            key, value = self._split_kv(stripped)
            if self._apply_dotted_path(out, key, value, props, allow_additional, path):
                i += 1
                continue

            field_schema = self._resolve_nullable(props.get(key))
            if field_schema is None and not allow_additional and self.cfg.strict_schema:
                raise SchemaViolationError(f"Unknown field '{key}' at {path}")

            if value is not None:
                out[key] = self._parse_scalar(value, field_schema)
                i += 1
                continue

            i += 1
            if i >= len(lines) or self._count_indent(lines[i]) <= indent:
                out[key] = self._empty_value_for_schema(field_schema)
                continue

            next_line = lines[i].strip()
            next_is_dash = next_line.startswith("-")

            if (field_schema or {}).get("type") == "array":
                if not next_is_dash:
                    raise ToonParserError(f"{path}.{key} expected an array block that starts with '- ' (line {i + 1})")
                arr, i = self._parse_array_block(lines, i, indent + self.cfg.indent_step, field_schema, f"{path}.{key}")
                out[key] = arr
                continue

            if (field_schema or {}).get("type") == "object" or self._schema_allows_additional(field_schema):
                obj, i = self._parse_object(lines, i, indent + self.cfg.indent_step, field_schema or {}, f"{path}.{key}")
                out[key] = obj
                continue

            if field_schema is None and allow_additional:
                obj, i = self._parse_object(lines, i, indent + self.cfg.indent_step, {}, f"{path}.{key}")
                out[key] = obj
                continue

            if (field_schema or {}).get("type") == "string" and next_is_dash:
                items, i = self._parse_scalar_dash_block(lines, i, indent + self.cfg.indent_step, f"{path}.{key}")
                out[key] = self._string_from_dash_items(items)
                continue

            raise ToonParserError(f"{path}.{key} is a scalar field but received a nested block")

        for field_name, buffer in indexed_buffers.items():
            if field_name in out:
                raise ToonParserError(f"{path}.{field_name} mixed indexed notation with another notation")
            max_idx = max(buffer.keys()) if buffer else -1
            values: List[Any] = [None] * (max_idx + 1)
            for idx, obj in buffer.items():
                values[idx] = obj
            out[field_name] = values

        return out, i

    def _parse_array_block(
        self, lines: List[str], i: int, indent: int, field_schema: Dict[str, Any], path: str
    ) -> Tuple[List[Any], int]:
        out: List[Any] = []
        items_schema = self._resolve_nullable(field_schema.get("items", {}) or {})
        item_props = self._get_props(items_schema)

        while i < len(lines):
            line = lines[i]
            cur_indent = self._count_indent(line)
            if cur_indent < indent:
                break
            if cur_indent > indent:
                raise ToonParserError(f"Unexpected indent inside array at {path} (line {i + 1})")

            stripped = line.strip()
            if stripped.startswith("-") and not stripped.startswith("- "):
                raise ToonParserError(f"List items must start with '- ' at {path} (line {i + 1})")
            if not stripped.startswith("- "):
                break
            item_text = stripped[2:].strip()

            if ":" in item_text:
                key, value = self._split_kv(item_text)
                obj: Dict[str, Any] = {}
                prop_schema = self._resolve_nullable(item_props.get(key))
                if value is None:
                    i += 1
                    nested, i = self._parse_value_block(lines, i, indent + self.cfg.indent_step, prop_schema, f"{path}[-].{key}")
                    obj[key] = nested
                else:
                    obj[key] = self._parse_scalar(value, prop_schema)
                    i += 1

                while i < len(lines):
                    next_line = lines[i]
                    next_indent = self._count_indent(next_line)
                    if next_indent < indent + self.cfg.indent_step:
                        break
                    if next_indent > indent + self.cfg.indent_step:
                        raise ToonParserError(f"Unexpected indent inside object item at {path} (line {i + 1})")
                    stripped_next = next_line.strip()
                    if stripped_next.startswith("-"):
                        break
                    child_key, child_value = self._split_kv(stripped_next)
                    child_schema = self._resolve_nullable(item_props.get(child_key))
                    if child_value is None:
                        i += 1
                        nested, i = self._parse_value_block(lines, i, indent + 2 * self.cfg.indent_step, child_schema, f"{path}[-].{child_key}")
                        obj[child_key] = nested
                    else:
                        obj[child_key] = self._parse_scalar(child_value, child_schema)
                        i += 1

                out.append(obj)
                continue

            if (items_schema or {}).get("type") == "object" and self.cfg.coerce_object_list_from_inline_scalars:
                object_key = self._best_inline_object_key(items_schema)
                if object_key is None:
                    raise ToonParserError(
                        f"{path}: scalar dash item cannot be coerced to list[object]; "
                        "use '- name: value' object notation"
                    )
                key_schema = self._resolve_nullable((item_props.get(object_key)))
                out.append({object_key: self._parse_scalar(item_text, key_schema)})
                i += 1
                continue

            out.append(self._parse_scalar(item_text, items_schema))
            i += 1

        return out, i

    def _parse_value_block(
        self, lines: List[str], i: int, indent: int, schema: Optional[Dict[str, Any]], path: str
    ) -> Tuple[Any, int]:
        schema = self._resolve_nullable(schema or {})
        if i >= len(lines):
            return {}, i
        next_indent = self._count_indent(lines[i])
        if next_indent < indent:
            return {}, i
        if schema.get("type") == "array":
            return self._parse_array_block(lines, i, indent, schema, path)
        if schema.get("type") == "object" or self._schema_allows_additional(schema) or not schema:
            return self._parse_object(lines, i, indent, schema or {}, path)
        raise ToonParserError(f"{path} expected a scalar value, not a nested block")

    def _parse_scalar_dash_block(self, lines: List[str], i: int, indent: int, path: str) -> Tuple[List[str], int]:
        out: List[str] = []
        while i < len(lines):
            line = lines[i]
            cur_indent = self._count_indent(line)
            if cur_indent < indent:
                break
            if cur_indent > indent:
                raise ToonParserError(f"Unexpected indent inside scalar list at {path} (line {i + 1})")
            stripped = line.strip()
            if stripped.startswith("-") and not stripped.startswith("- "):
                raise ToonParserError(f"List items must start with '- ' at {path} (line {i + 1})")
            if not stripped.startswith("- "):
                break
            out.append(str(self._parse_scalar(stripped[2:].strip(), {"type": "string"})))
            i += 1
        return out, i

    def _string_from_dash_items(self, items: List[str]) -> str:
        normalized: List[str] = []
        for item in items:
            text = item.strip()
            if not text:
                continue
            if text.startswith("- "):
                normalized.append(text)
            else:
                normalized.append(f"- {text}")
        return "\n".join(normalized)

    def _parse_tabular(
        self, lines: List[str], i: int, indent: int, field_name: str, field_schema: Optional[Dict[str, Any]], path: str
    ) -> Tuple[List[Dict[str, Any]], int]:
        header = lines[i].strip()
        match = self._TABULAR_HEADER_RE.match(header)
        assert match
        expected_rows = int(match.group("n"))
        columns = [column.strip() for column in match.group("cols").split(",") if column.strip()]

        field_schema = field_schema or {}
        if field_schema.get("type") != "array":
            raise SchemaViolationError(f"{path}: tabular notation used for a non-array field")
        if ToonIntelligence.is_complex_array(field_schema, self.cfg.complexity_threshold, self._defs):
            raise PolicyViolationError(f"{path}: tabular notation is only allowed for flat object arrays")
        if not self.cfg.allow_tabular_for_flat_objects:
            raise PolicyViolationError(f"{path}: tabular notation is disabled by configuration")

        items_schema = self._resolve_nullable(field_schema.get("items", {}) or {})
        item_props = items_schema.get("properties", {}) or {}
        if self.cfg.strict_schema and item_props:
            unknown = [column for column in columns if column not in item_props]
            if unknown:
                raise SchemaViolationError(f"{path}: unknown tabular columns: {unknown}")

        out: List[Dict[str, Any]] = []
        i += 1
        while i < len(lines):
            line = lines[i]
            cur_indent = self._count_indent(line)
            if cur_indent <= indent:
                break
            if cur_indent != indent + self.cfg.indent_step:
                raise ToonParserError(f"{path}: invalid tabular indentation at line {i + 1}")
            row = self._csv_row(line.strip())
            if len(row) != len(columns):
                raise SchemaViolationError(
                    f"{path}: expected {len(columns)} tabular columns, got {len(row)} at row {len(out)}"
                )
            item: Dict[str, Any] = {}
            for idx, column in enumerate(columns):
                item[column] = self._parse_scalar(row[idx], item_props.get(column))
            out.append(item)
            i += 1
            if self.cfg.strict_count and len(out) >= expected_rows:
                break

        if self.cfg.strict_count and len(out) != expected_rows:
            raise SchemaViolationError(f"{path}: expected {expected_rows} rows, got {len(out)}")

        return out, i

    def _compact_error(self, error: Exception | str) -> str:
        text = str(error).strip().replace("\n", " ")
        return text[:240]
