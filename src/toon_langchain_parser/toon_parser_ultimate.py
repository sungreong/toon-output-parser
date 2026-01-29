# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union

try:
    from pydantic import BaseModel
except Exception as e:  # pragma: no cover
    raise ImportError("pydantic(v2) is required. Install: pip install pydantic>=2") from e


# ============================================================
# Errors
# ============================================================
class ToonParserError(ValueError):
    pass


class ToonDecodeError(ToonParserError):
    pass


class PolicyViolationError(ToonParserError):
    pass


class SchemaViolationError(ToonParserError):
    pass


class ModelComplexityError(ToonParserError):
    """minimal 모드에서 지원하지 않는 복잡한 스키마일 때 발생하는 에러."""
    pass


# ============================================================
# Config
# ============================================================
@dataclass(frozen=True)
class ParserConfig:
    indent_step: int = 2
    complexity_threshold: int = 3  # item ?? ??? ? ? ??? complex? ??
    protect_string_ids: bool = True
    strict_schema: bool = True
    strict_count: bool = False
    allow_tabular_for_flat_objects: bool = True
    
    # Format instructions 스타일 선택
    instructions_mode: str = "adaptive"  # "adaptive" | "minimal" | "json"
    # - adaptive: 상세한 설명 (1762 chars, 손익분기 19,363 chars)
    # - minimal: Few-shot 스타일 (102 chars, 손익분기 1,121 chars) ⭐ 권장
    # - json: JSON 포맷 사용 (TOON 비활성화)
    
    # minimal 모드 동작 설정
    auto_fallback_to_json: bool = True  # minimal 모드에서 복잡한 스키마 감지 시 자동으로 json으로 폴백
    strict_minimal_validation: bool = False  # True시 복잡한 스키마에서 에러 발생 (폴백 안 함)


# ============================================================
# Model Complexity Validator
# ============================================================
@dataclass(frozen=True)
class ComplexityLimits:
    """minimal 모드에서 지원 가능한 스키마의 복잡도 제한.
    
    핵심 제약: depth(중첩 깊이)만 제한
    - Level 1 (flat): 필드 개수, 타입 복잡도 무관하게 모두 지원
    - Level 2 (1단계 중첩): 대부분 지원
    - Level 3+ (2단계 이상 중첩): LLM이 TOON 형식을 제대로 출력 못할 가능성 높음
    """
    max_nesting_depth: int = 2  # 최대 중첩 깊이 (객체 안의 객체) - 🎯 핵심 제약
    
    # 아래 제약들은 사실상 무제한 (매우 큰 값으로 설정)
    max_array_nesting: int = 999  # 배열 중첩은 제한 없음
    max_union_types: int = 999  # Union 타입 필드 개수 제한 없음
    max_object_fields: int = 999  # flat 객체 필드 개수 제한 없음
    max_array_item_fields: int = 999  # 배열 아이템 필드 개수 제한 없음
    allow_nested_unions: bool = True  # 중첩 Union 허용
    allow_recursive_refs: bool = False  # 재귀 $ref만 불허 (무한 루프 방지)


class ComplexityMetrics:
    """스키마의 복잡도를 측정하는 메트릭."""
    
    def __init__(self):
        self.max_nesting_depth = 0
        self.max_array_nesting = 0
        self.union_type_count = 0
        self.max_object_fields = 0
        self.max_array_item_fields = 0
        self.has_nested_unions = False
        self.has_recursive_refs = False
        self.violation_reasons: List[str] = []
    
    def add_violation(self, reason: str):
        """제약 위반 사항을 기록."""
        self.violation_reasons.append(reason)
    
    def is_within_limits(self, limits: ComplexityLimits) -> bool:
        """주어진 제약 내에 있는지 확인."""
        self.violation_reasons.clear()
        
        if self.max_nesting_depth > limits.max_nesting_depth:
            self.add_violation(
                f"중첩 깊이 초과: {self.max_nesting_depth} > {limits.max_nesting_depth} "
                f"(객체 안에 객체가 {self.max_nesting_depth}단계 중첩됨)"
            )
        
        if self.max_array_nesting > limits.max_array_nesting:
            self.add_violation(
                f"배열 중첩 깊이 초과: {self.max_array_nesting} > {limits.max_array_nesting} "
                f"(배열 안에 배열이 {self.max_array_nesting}단계 중첩됨)"
            )
        
        if self.union_type_count > limits.max_union_types:
            self.add_violation(
                f"Union 타입 개수 초과: {self.union_type_count} > {limits.max_union_types} "
                f"(Union 타입 필드가 너무 많음)"
            )
        
        if self.max_object_fields > limits.max_object_fields:
            self.add_violation(
                f"객체 필드 개수 초과: {self.max_object_fields} > {limits.max_object_fields} "
                f"(단일 객체에 필드가 너무 많음)"
            )
        
        if self.max_array_item_fields > limits.max_array_item_fields:
            self.add_violation(
                f"배열 아이템 필드 개수 초과: {self.max_array_item_fields} > {limits.max_array_item_fields} "
                f"(배열의 각 아이템 객체에 필드가 너무 많음)"
            )
        
        if self.has_nested_unions and not limits.allow_nested_unions:
            self.add_violation("중첩된 Union 타입 발견 (예: Union[List[Union[str, int]], Dict])")
        
        if self.has_recursive_refs and not limits.allow_recursive_refs:
            self.add_violation("재귀적 $ref 발견 (예: Node → children: List[Node])")
        
        return len(self.violation_reasons) == 0
    
    def get_violation_summary(self) -> str:
        """제약 위반 사항을 요약한 문자열 반환."""
        if not self.violation_reasons:
            return "제약 위반 없음"
        
        return "\n".join(f"  - {reason}" for reason in self.violation_reasons)


class ModelComplexityAnalyzer:
    """BaseModel의 복잡도를 분석하는 클래스."""
    
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
        self._visited_refs: set = set()
    
    def analyze(self) -> ComplexityMetrics:
        """스키마의 복잡도를 분석하고 메트릭을 반환."""
        self._analyze_schema(self.schema, depth=0, array_depth=0, path="$")
        return self.metrics
    
    def _resolve_ref(self, ref: str) -> Dict[str, Any]:
        """$ref를 해석하여 실제 스키마 반환."""
        if not ref.startswith("#/"):
            return {}
        parts = ref.lstrip("#/").split("/")
        if len(parts) == 2 and parts[0] in ("$defs", "definitions"):
            name = parts[1]
            return self._defs.get(name, {})
        return {}
    
    def _analyze_schema(
        self, 
        schema: Dict[str, Any], 
        depth: int, 
        array_depth: int, 
        path: str,
        in_union: bool = False
    ):
        """재귀적으로 스키마를 분석."""
        if not isinstance(schema, dict):
            return
        
        # $ref 처리
        if "$ref" in schema:
            ref = schema["$ref"]
            if ref in self._visited_refs:
                self.metrics.has_recursive_refs = True
                return
            self._visited_refs.add(ref)
            resolved = self._resolve_ref(ref)
            self._analyze_schema(resolved, depth, array_depth, f"{path}.$ref", in_union)
            self._visited_refs.discard(ref)
            return
        
        # Union 타입 처리
        if "anyOf" in schema or "oneOf" in schema:
            self.metrics.union_type_count += 1
            
            if in_union:
                self.metrics.has_nested_unions = True
            
            union_schemas = schema.get("anyOf", schema.get("oneOf", []))
            for idx, sub_schema in enumerate(union_schemas):
                if isinstance(sub_schema, dict):
                    self._analyze_schema(
                        sub_schema, 
                        depth, 
                        array_depth, 
                        f"{path}.union[{idx}]",
                        in_union=True
                    )
            return
        
        schema_type = schema.get("type")
        
        # 객체 타입
        if schema_type == "object" or "properties" in schema:
            props = schema.get("properties", {}) or {}
            field_count = len(props)
            
            # 객체 필드 개수 업데이트
            if array_depth > 0:
                # 배열 아이템인 경우
                self.metrics.max_array_item_fields = max(
                    self.metrics.max_array_item_fields, 
                    field_count
                )
            else:
                # 일반 객체인 경우
                self.metrics.max_object_fields = max(
                    self.metrics.max_object_fields, 
                    field_count
                )
            
            # 중첩 깊이 업데이트
            self.metrics.max_nesting_depth = max(self.metrics.max_nesting_depth, depth)
            
            # 각 필드에 대해 재귀 분석
            for field_name, field_schema in props.items():
                self._analyze_schema(
                    field_schema, 
                    depth + 1, 
                    array_depth, 
                    f"{path}.{field_name}",
                    in_union
                )
        
        # 배열 타입
        elif schema_type == "array":
            items_schema = schema.get("items", {}) or {}
            
            # 배열 중첩 깊이 업데이트
            new_array_depth = array_depth + 1
            self.metrics.max_array_nesting = max(
                self.metrics.max_array_nesting, 
                new_array_depth
            )
            
            # 배열 아이템에 대해 재귀 분석
            self._analyze_schema(
                items_schema, 
                depth, 
                new_array_depth, 
                f"{path}[]",
                in_union
            )
    
    @staticmethod
    def validate_for_minimal_mode(
        model: Type[BaseModel], 
        limits: Optional[ComplexityLimits] = None
    ) -> Tuple[bool, ComplexityMetrics]:
        """minimal 모드에 적합한 모델인지 검증.
        
        Args:
            model: 검증할 Pydantic BaseModel
            limits: 복잡도 제한 (None이면 기본값 사용)
        
        Returns:
            (is_valid, metrics) 튜플
        """
        if limits is None:
            limits = ComplexityLimits()
        
        analyzer = ModelComplexityAnalyzer(model)
        metrics = analyzer.analyze()
        is_valid = metrics.is_within_limits(limits)
        
        return is_valid, metrics


# ============================================================
# Adaptive Prompt Builder
# ============================================================
class ToonIntelligence:
    @staticmethod
    def _props(schema: Dict[str, Any]) -> Dict[str, Any]:
        return schema.get("properties", {}) or {}

    @staticmethod
    def _items(schema: Dict[str, Any]) -> Dict[str, Any]:
        return schema.get("items", {}) or {}

    @staticmethod
    def _is_scalar_type(t: str) -> bool:
        return t in ("string", "integer", "number", "boolean")

    @staticmethod
    def _dummy_scalar(t: str) -> str:
        if t == "string":
            return "text"
        if t == "integer":
            return "1"
        if t == "number":
            return "0.1"
        if t == "boolean":
            return "true"
        return "text"

    @staticmethod
    def is_complex_array(field_schema: Dict[str, Any], threshold: int) -> bool:
        if field_schema.get("type") != "array":
            return False
        items = ToonIntelligence._items(field_schema)

        # $ref / anyOf ?? ????? complex? ??
        if "$ref" in items or "anyOf" in items or "oneOf" in items or "allOf" in items:
            return True

        item_type = items.get("type")
        if item_type not in (None, "object"):
            return True

        props = items.get("properties", {}) or {}

        # ?? ??(??/??) ?? -> complex
        for v in props.values():
            t = v.get("type")
            if t in ("object", "array") or "$ref" in v or "anyOf" in v or "oneOf" in v or "allOf" in v:
                return True

        # ?? ?? ??? complex? ??
        return len(props) > threshold

    @staticmethod
    def is_flat_object_array(field_schema: Dict[str, Any]) -> bool:
        if field_schema.get("type") != "array":
            return False
        items = ToonIntelligence._items(field_schema)
        if items.get("type") not in (None, "object"):
            return False
        props = items.get("properties", {}) or {}
        if not props:
            return False
        return all(ToonIntelligence._is_scalar_type((v.get("type") or "string")) for v in props.values())

    @staticmethod
    def _detect_union_types(props: Dict[str, Any]) -> List[str]:
        """Union 타입을 가진 필드 감지."""
        union_fields = []
        for fname, fsch in props.items():
            # anyOf, oneOf 체크
            if "anyOf" in fsch or "oneOf" in fsch:
                union_fields.append(fname)
            # Python 3.10+ union syntax (str | int | None)
            elif isinstance(fsch.get("type"), list):
                union_fields.append(fname)
        return union_fields
    
    @staticmethod
    def _detect_string_fields(props: Dict[str, Any]) -> List[str]:
        """String 타입 필드 감지."""
        return [fname for fname, fsch in props.items() if fsch.get("type") == "string"]
    
    @staticmethod
    def _detect_special_char_fields(props: Dict[str, Any]) -> List[str]:
        """특수 문자가 포함될 가능성이 있는 필드 감지 (description에 특수문자 관련 키워드 포함)."""
        keywords = ["url", "email", "code", "json", "colon", ":", "-", "dash", "special"]
        special_fields = []
        for fname, fsch in props.items():
            desc = (fsch.get("description", "") or "").lower()
            if any(kw in desc or kw in fname.lower() for kw in keywords):
                special_fields.append(fname)
        return special_fields

    @staticmethod
    def build_adaptive_prompt(model: Type[BaseModel], cfg: ParserConfig = ParserConfig()) -> str:
        schema = model.model_json_schema()
        props = schema.get("properties", {}) or {}
        required_fields = set(schema.get("required", []))
        
        # 스키마 분석
        union_fields = ToonIntelligence._detect_union_types(props)
        string_fields = ToonIntelligence._detect_string_fields(props)
        special_char_fields = ToonIntelligence._detect_special_char_fields(props)

        out: List[str] = []
        out.append("### TOON FORMAT GUIDE (MUST FOLLOW EXACTLY)")
        out.append("")
        out.append("TOON is a simple, indentation-based format similar to YAML but simpler.")
        out.append("")
        out.append("#### CORE RULES:")
        out.append("1. Use 2 spaces for each indentation level (NEVER tabs)")
        out.append("2. Every line MUST have a colon (:) - format is always 'key: value' or 'key:' for nested")
        out.append("3. Output ONLY TOON format - NEVER mix with JSON, YAML, or plain text")
        out.append("4. ALL REQUIRED FIELDS MUST BE INCLUDED - validation will fail if any are missing")
        out.append("")
        
        # List required fields prominently
        if required_fields:
            out.append("#### REQUIRED FIELDS (MUST INCLUDE ALL):")
            for fname in sorted(required_fields):
                fsch = props.get(fname, {})
                ftype = fsch.get("type", "any")
                fdesc = fsch.get("description", "")
                out.append(f"✓ {fname}: {ftype}" + (f" - {fdesc}" if fdesc else ""))
            out.append("")
        
        out.append("")
        out.append("#### BASIC SYNTAX:")
        out.append("```")
        out.append("# Simple values (inline)")
        out.append("name: John")
        out.append("age: 30")
        out.append("active: true")
        out.append("score: 95.5")
        out.append("unknown: null")
        out.append("")
        out.append("# Nested object (colon, then indent 2 spaces)")
        out.append("address:")
        out.append("  street: Main St")
        out.append("  city: NYC")
        out.append("")
        out.append("# Empty object/array")
        out.append("empty_object: {}")
        out.append("empty_array: []")
        out.append("```")
        out.append("")
        out.append("#### NUMBER FORMATTING:")
        out.append("- NEVER use thousand separators: 25000 (NOT 25,000)")
        out.append("- Integers: 42")
        out.append("- Decimals: 3.14")
        out.append("")

        # ======= 동적 타입별 가이드 추가 =======
        
        # String 필드 경고
        if string_fields:
            out.append("#### ⚠️ CRITICAL: STRING FIELD RULES")
            out.append("")
            out.append("**String 타입 필드는 반드시 단일 값으로 출력:**")
            for fname in string_fields[:5]:  # 최대 5개만 표시
                out.append(f"- {fname}: MUST be single string value (NOT list/array)")
            out.append("")
            out.append("❌ WRONG (list format):")
            out.append(f"{string_fields[0]}:")
            out.append('  - "item 1"')
            out.append('  - "item 2"')
            out.append("")
            out.append("✅ CORRECT (single string):")
            out.append(f'{string_fields[0]}: "item 1, item 2"')
            out.append("")
        
        # Union 타입 가이드
        if union_fields:
            out.append("#### ⚠️ UNION TYPE FIELDS (Multiple Types Allowed)")
            out.append("")
            out.append("**다음 필드들은 여러 타입 중 하나를 선택:**")
            for fname in union_fields[:5]:
                fsch = props.get(fname, {})
                any_of = fsch.get("anyOf", fsch.get("oneOf", []))
                types = [s.get("type", "any") for s in any_of if isinstance(s, dict)]
                if types:
                    out.append(f"- {fname}: Choose one → {' | '.join(types)}")
            out.append("")
            out.append("**타입 선택 규칙:**")
            out.append("1. 문서에서 숫자면 → 숫자로 출력 (e.g., count: 30)")
            out.append("2. 문서에서 텍스트면 → 문자열로 출력 (e.g., value: \"text\")")
            out.append("3. 배열이면 → 콤마 형식 또는 대시 형식 (NOT JSON string like '[\"a\",\"b\"]')")
            out.append("4. 객체면 → 중첩 형식 (NOT JSON string like '{\"key\":\"value\"}')")
            out.append("")
            out.append("❌ WRONG (JSON string):")
            out.append('metadata: ["tag1", "tag2", "tag3"]')
            out.append("")
            out.append("✅ CORRECT (actual list):")
            out.append("metadata[3]: tag1,tag2,tag3")
            out.append("")
        
        # 특수 문자 필드 가이드
        if special_char_fields:
            out.append("#### ⚠️ SPECIAL CHARACTER HANDLING")
            out.append("")
            out.append("**다음 필드들은 특수 문자를 포함할 수 있습니다:**")
            for fname in special_char_fields[:5]:
                out.append(f"- {fname}: Use quotes if contains :, -, or special chars")
            out.append("")
            out.append("**따옴표 사용 규칙:**")
            out.append('✅ description: "This has: special chars"')
            out.append('✅ note: "- This starts with dash"')
            out.append('✅ code: "def func(x: int) -> str: return x"')
            out.append("✅ url: https://example.com (quotes optional for URLs)")
            out.append("")

        out.append("#### ARRAY FORMATS:")
        out.append("")
        out.append("**Option 1: Dash List (MOST COMMON - USE THIS FOR OBJECTS)**")
        out.append("```")
        out.append("items:")
        out.append("  - name: Item1")
        out.append("    value: 100")
        out.append("  - name: Item2")
        out.append("    value: 200")
        out.append("```")
        out.append("CRITICAL: Each item starts with '- ' (dash + space) at the same indent level")
        out.append("CRITICAL: Additional fields continue with 2 more spaces (no dash)")
        out.append("")
        out.append("**Option 2: Inline Scalar List (ONLY for simple values)**")
        out.append("```")
        out.append("tags[3]: red,green,blue")
        out.append("numbers[2]: 10,20")
        out.append("```")
        out.append("Use ONLY when items are simple strings/numbers, NOT objects")
        out.append("")
        out.append("**Option 3: Tabular (ONLY for flat objects with few fields)**")
        out.append("```")
        out.append("items[2,]{name,value}:")
        out.append("  Item1,100")
        out.append("  Item2,200")
        out.append("```")
        out.append("NEVER add row numbers like: 0,Item1,100 or 1,Item2,200")
        out.append("")
        out.append("#### SCHEMA-SPECIFIC ARRAY FORMATS:")
        
        for fname, fsch in props.items():
            if fsch.get("type") != "array":
                continue

            items = fsch.get("items", {}) or {}
            # scalar array
            if items.get("type") in ("string", "integer", "number", "boolean"):
                t = items.get("type")
                out.append(f"- {fname}: Use inline format -> {fname}[N]: value1,value2,value3")
                continue

            complex_ = ToonIntelligence.is_complex_array(fsch, cfg.complexity_threshold)
            flat_ = ToonIntelligence.is_flat_object_array(fsch)
            item_props = (items.get("properties", {}) or {})

            if complex_:
                out.append(f"- {fname}: MUST use dash list (complex objects)")
                keys = list(item_props.keys())[:3] or ["field1", "field2"]
                out.append(f"  {fname}:")
                out.append(f"    - {keys[0]}: example")
                if len(keys) > 1:
                    out.append(f"      {keys[1]}: example")
                if len(keys) > 2:
                    out.append(f"      {keys[2]}: example")
            else:
                out.append(f"- {fname}: Use dash list (recommended)")
                keys = list(item_props.keys())[:2] or ["field1", "field2"]
                out.append(f"  {fname}:")
                out.append(f"    - {keys[0]}: example")
                if len(keys) > 1:
                    out.append(f"      {keys[1]}: example")

        out.append("")
        out.append("#### COMMON MISTAKES TO AVOID:")
        out.append("")
        out.append("❌ WRONG: Standalone word without colon")
        out.append("TOON")
        out.append("items")
        out.append("")
        out.append("✅ CORRECT: Always use key: value")
        out.append("format: TOON")
        out.append("items: []")
        out.append("")
        out.append("❌ WRONG: Missing dash space in lists")
        out.append("items:")
        out.append("  -name: value")
        out.append("  - name value")
        out.append("")
        out.append("✅ CORRECT: Dash + space, then key: value")
        out.append("items:")
        out.append("  - name: value")
        out.append("")
        out.append("❌ WRONG: Row numbers in tabular")
        out.append("items[2,]{name,value}:")
        out.append("  0,Item1,100")
        out.append("  1,Item2,200")
        out.append("")
        out.append("✅ CORRECT: No row numbers")
        out.append("items[2,]{name,value}:")
        out.append("  Item1,100")
        out.append("  Item2,200")
        out.append("")
        out.append("❌ WRONG: Thousand separators")
        out.append("amount: 25,000")
        out.append("")
        out.append("✅ CORRECT: Plain numbers")
        out.append("amount: 25000")
        out.append("")
        
        # 최종 체크리스트 강화
        out.append("#### FINAL CHECKLIST BEFORE SUBMITTING:")
        out.append("✓ Every line has a colon (:)")
        out.append("✓ String fields = single value (NOT dash list)")
        if union_fields:
            out.append("✓ Union types = match document's actual type (NOT JSON string)")
        if special_char_fields:
            out.append("✓ Special characters = use quotes when needed")
        out.append("✓ Arrays = use [N]:item1,item2 or dash format (NOT JSON array string)")
        out.append("✓ Objects = nested key:value (NOT JSON object string)")
        out.append("✓ Indent with 2 spaces per level")
        out.append("✓ No thousand separators in numbers")
        out.append("✓ No row indices in tabular format")
        
        return "\n".join(out).strip()
    
    @staticmethod
    def _analyze_schema_structure(schema: Dict[str, Any]) -> Dict[str, Any]:
        """스키마 구조를 분석하여 필요한 가이드 타입을 결정합니다.
        
        Returns:
            dict: {
                'depth': int,  # 최대 중첩 깊이
                'has_arrays': bool,
                'has_nested_objects': bool,
                'has_union_types': bool,
                'field_count': int,
                'array_fields': List[str],
                'nested_fields': List[str],
                'union_fields': List[str],
            }
        """
        props = schema.get("properties", {}) or {}
        required = set(schema.get("required", []))
        
        info = {
            'depth': 0,
            'has_arrays': False,
            'has_nested_objects': False,
            'has_union_types': False,
            'field_count': len(props),
            'array_fields': [],
            'nested_fields': [],
            'union_fields': [],
        }
        
        for fname, fsch in props.items():
            ftype = fsch.get("type")
            
            # Array 감지
            if ftype == "array":
                info['has_arrays'] = True
                info['array_fields'].append(fname)
            
            # Nested object 감지
            elif ftype == "object" or "properties" in fsch:
                info['has_nested_objects'] = True
                info['nested_fields'].append(fname)
                info['depth'] = max(info['depth'], 1)
            
            # Union 타입 감지
            if "anyOf" in fsch or "oneOf" in fsch:
                info['has_union_types'] = True
                info['union_fields'].append(fname)
        
        return info
    
    @staticmethod
    def build_minimal_prompt(model: Type[BaseModel], cfg: ParserConfig = ParserConfig()) -> str:
        """스키마 구조에 따라 동적으로 최적화된 minimal instructions 생성.
        
        전략:
        - Flat object (depth=0): 초간결 가이드 (~80 chars)
        - 1-depth nested: 중간 가이드 (~150 chars)
        - 2-depth nested: 상세 가이드 (~250 chars)
        
        손익분기점: 출력이 1,121 chars 이상일 때 JSON 대비 효율적
        """
        schema = model.model_json_schema()
        required = schema.get("required", [])
        props = schema.get("properties", {}) or {}
        
        # 스키마 구조 분석
        struct_info = ToonIntelligence._analyze_schema_structure(schema)
        
        # ==========================================
        # CASE 1: Flat Object (depth=0, 필드 많음)
        # ==========================================
        if struct_info['depth'] == 0 and not struct_info['has_nested_objects']:
            example_lines = []
            
            # Required 필드 예시 (최대 5개)
            for fname in list(required)[:5]:
                fsch = props.get(fname, {})
                ftype = fsch.get("type", "string")
                
                if ftype == "string":
                    example_lines.append(f"{fname}: text")
                elif ftype == "integer":
                    example_lines.append(f"{fname}: 123")
                elif ftype == "array":
                    example_lines.append(f"{fname}[2]: a,b")
                else:
                    example_lines.append(f"{fname}: value")
            
            # Optional 필드 예시 1개 (생략 가능함을 보여주기)
            optional_shown = 0
            for fname, fsch in props.items():
                if fname not in required and optional_shown < 1:
                    ftype = fsch.get("type", "string")
                    if ftype == "string":
                        example_lines.append(f"{fname}: optional")
                    else:
                        example_lines.append(f"{fname}: value")
                    optional_shown += 1
                    break
            
            example = "\n".join(example_lines)
            required_str = ", ".join(list(required)[:10]) if required else "none"
            
            return f"""TOON format (flat, NO indent):

{example}

key: value | key[N]: a,b,c
Required: {required_str}"""
        
        # ==========================================
        # CASE 2: Has Arrays or Union Types (but still flat)
        # ==========================================
        if struct_info['depth'] == 0 and (struct_info['has_arrays'] or struct_info['has_union_types']):
            example_lines = []
            
            # Required 필드 예시
            for fname in list(required)[:5]:
                fsch = props.get(fname, {})
                ftype = fsch.get("type", "string")
                
                if ftype == "string":
                    example_lines.append(f"{fname}: text")
                elif ftype == "integer":
                    example_lines.append(f"{fname}: 123")
                elif ftype == "array":
                    example_lines.append(f"{fname}[2]: item1,item2")
                else:
                    example_lines.append(f"{fname}: value")
            
            example = "\n".join(example_lines)
            required_str = ", ".join(list(required)[:10]) if required else "none"
            
            # Array 또는 Union 가이드 추가
            extra_guide = ""
            if struct_info['has_arrays']:
                extra_guide += "\nArrays: field[N]: val1,val2,val3"
            if struct_info['has_union_types']:
                extra_guide += "\nOptional fields: skip if not needed"
            
            return f"""TOON format (flat):

{example}

Rules: key: value{extra_guide}
Required: {required_str}"""
        
        # ==========================================
        # CASE 3: Has Nested Objects (1-2 depth)
        # ==========================================
        if struct_info['has_nested_objects'] or struct_info['depth'] > 0:
            example_lines = []
            
            # Required 필드 예시
            shown_nested = False
            for fname in list(required)[:4]:
                fsch = props.get(fname, {})
                ftype = fsch.get("type", "string")
                
                if ftype == "object" or "properties" in fsch and not shown_nested:
                    # Nested object 예시 1개
                    example_lines.append(f"{fname}:")
                    nested_props = fsch.get("properties", {})
                    for nf in list(nested_props.keys())[:2]:
                        example_lines.append(f"  {nf}: value")
                    shown_nested = True
                elif ftype == "string":
                    example_lines.append(f"{fname}: text")
                elif ftype == "integer":
                    example_lines.append(f"{fname}: 123")
                elif ftype == "array":
                    example_lines.append(f"{fname}[2]: a,b")
                else:
                    example_lines.append(f"{fname}: value")
            
            example = "\n".join(example_lines)
            required_str = ", ".join(list(required)[:8]) if required else "none"
            
            return f"""TOON format:

{example}

Rules:
- key: value (inline)
- Nested: key: then indent 2 spaces
- Arrays: key[N]: a,b or dash list

Required: {required_str}"""
        
        # ==========================================
        # FALLBACK: Default minimal
        # ==========================================
        example_lines = []
        for fname in list(required)[:3]:
            fsch = props.get(fname, {})
            ftype = fsch.get("type", "string")
            
            if ftype == "string":
                example_lines.append(f"{fname}: text")
            elif ftype == "integer":
                example_lines.append(f"{fname}: 123")
            else:
                example_lines.append(f"{fname}: value")
        
        example = "\n".join(example_lines)
        required_str = ", ".join(required) if required else "none"
        
        return f"""TOON: key: value

{example}

Required: {required_str}"""


# ============================================================
# Core Parser
# ============================================================
class ToonParser:
    _TABULAR_HEADER_RE = re.compile(r"^(?P<name>[A-Za-z0-9_]+)\[(?P<n>\d+),(?:#)?\]\{(?P<cols>[^}]+)\}:\s*$")
    _SCALAR_LIST_RE = re.compile(r"^(?P<name>[A-Za-z0-9_]+)\[(?P<n>\d+)\]:\s*(?P<body>.*)$")
    _INDEXED_ITEM_RE = re.compile(r"^(?P<name>[A-Za-z0-9_]+)\[(?P<idx>\d+)\]:(?:\s*)$")
    _CODE_FENCE_RE = re.compile(r"```(?:toon)?\s*(.*?)\s*```", flags=re.DOTALL)

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
        
        # minimal 모드일 때 스키마 복잡도 검증 (자동 폴백 여부에 따라 동작)
        self._effective_mode = self.cfg.instructions_mode
        if self.cfg.instructions_mode == "minimal":
            self._validate_minimal_mode_compatibility()

    # ---------------- Validation ----------------
    def _validate_minimal_mode_compatibility(self):
        """minimal 모드에서 사용 가능한 스키마인지 검증.
        
        핵심 제약: depth(중첩 깊이)만 검증
        - max_nesting_depth=2: 객체 중첩 2단계까지 허용
        - 나머지 제약 (필드 개수, 타입 등)은 제한 없음
        
        복잡한 스키마일 경우:
        - auto_fallback_to_json=True: json 모드로 자동 폴백 (경고만 출력)
        - strict_minimal_validation=True: ModelComplexityError 발생
        
        Raises:
            ModelComplexityError: strict_minimal_validation=True이고 depth가 초과된 경우
        """
        # 복잡도 제한 정의 (minimal 모드 기준: depth만 제한)
        limits = ComplexityLimits(
            max_nesting_depth=2,  # 🎯 핵심: 최대 2단계 중첩 (예: user.job.company까지 허용)
            max_array_nesting=999,  # 배열 중첩 무제한
            max_union_types=999,  # Union 타입 무제한
            max_object_fields=999,  # flat 객체 필드 무제한
            max_array_item_fields=999,  # 배열 아이템 필드 무제한
            allow_nested_unions=True,  # 중첩 Union 허용
            allow_recursive_refs=False,  # 재귀 $ref만 불허 (무한 루프 방지)
        )
        
        # 복잡도 분석
        is_valid, metrics = ModelComplexityAnalyzer.validate_for_minimal_mode(
            self.model, limits
        )
        
        if not is_valid:
            model_name = self.model.__name__
            violation_summary = metrics.get_violation_summary()
            
            # strict 모드: 에러 발생
            if self.cfg.strict_minimal_validation:
                error_message = f"""
❌ minimal 모드는 depth가 깊은 스키마를 지원하지 않습니다.

모델: {model_name}
지원 불가 사유:
{violation_summary}

📋 minimal 모드 제약 사항 (depth만 제한):
  - 최대 중첩 깊이: {limits.max_nesting_depth}단계 (현재: {metrics.max_nesting_depth}단계) 🎯
  - 배열 중첩: 무제한 (현재: {metrics.max_array_nesting}단계)
  - Union 타입 필드: 무제한 (현재: {metrics.union_type_count}개)
  - 객체 필드: 무제한 (현재: {metrics.max_object_fields}개)
  - 배열 아이템 필드: 무제한 (현재: {metrics.max_array_item_fields}개)

💡 해결 방법:
  1. 자동 폴백 허용 (권장)
     cfg = ParserConfig(instructions_mode="minimal", auto_fallback_to_json=True)
  
  2. instructions_mode="json"으로 변경 (표준 JSON 사용)
     cfg = ParserConfig(instructions_mode="json")
  
  3. 스키마 중첩을 2단계 이하로 줄이기 (필드 개수는 상관 없음)
""".strip()
                
                raise ModelComplexityError(error_message)
            
            # 자동 폴백 모드: json으로 전환하고 경고 출력
            if self.cfg.auto_fallback_to_json:
                self._effective_mode = "json"
                
                import warnings
                warning_message = f"""
⚠️ minimal 모드는 '{model_name}' 스키마의 depth가 너무 깊어 자동으로 JSON 모드로 전환되었습니다.

지원 불가 사유:
{violation_summary}

📋 minimal 모드 제약 사항 (depth만 제한):
  - 최대 중첩 깊이: {limits.max_nesting_depth}단계 (현재: {metrics.max_nesting_depth}단계) 🎯
  - 배열 중첩: 무제한 (현재: {metrics.max_array_nesting}단계)
  - Union 타입 필드: 무제한 (현재: {metrics.union_type_count}개)
  - 객체 필드: 무제한 (현재: {metrics.max_object_fields}개)
  - 배열 아이템 필드: 무제한 (현재: {metrics.max_array_item_fields}개)

💡 JSON 모드는 표준 JSON 형식을 사용하므로 모든 복잡도를 지원합니다.
   TOON의 압축 효과는 없지만, 파싱 성공률이 높습니다.

이 경고를 끄려면:
  cfg = ParserConfig(instructions_mode="json")  # 처음부터 JSON 모드 사용
""".strip()
                
                warnings.warn(warning_message, UserWarning, stacklevel=3)
    
    # ---------------- Public ----------------
    def get_format_instructions(self) -> str:
        """Format instructions 반환 (설정에 따라 다른 스타일 사용)
        
        Note: minimal 모드에서 자동 폴백된 경우 _effective_mode가 "json"으로 변경됨
        """
        # 자동 폴백이 발생했을 경우 effective_mode 사용
        effective_mode = getattr(self, '_effective_mode', self.cfg.instructions_mode)
        
        if effective_mode == "minimal":
            # Few-shot 스타일 (85% 감축, 손익분기 1,121 chars)
            return ToonIntelligence.build_minimal_prompt(self.model, self.cfg)
        elif effective_mode == "json":
            # JSON 포맷 사용 (TOON 비활성화)
            return self._get_json_schema_instructions()
        else:  # "adaptive" (기본값)
            # 상세한 설명 (교육용, 디버깅용)
            return ToonIntelligence.build_adaptive_prompt(self.model, self.cfg)
    
    def _get_json_schema_instructions(self) -> str:
        """JSON 포맷 instructions (비교용)"""
        import json
        schema_str = json.dumps(self.model.model_json_schema(), ensure_ascii=False, indent=2)
        return f"""You must output ONLY a valid JSON object. Your response must start with {{ and end with }}.

Schema:
{schema_str}

CRITICAL RULES:
- DO NOT use ```json, ```toon, or ``` code fences
- DO NOT add any explanations before or after the JSON
- DO NOT add any markdown formatting
- Start directly with {{ and end with }}
- Output pure JSON only"""

    # ------------- Schema helpers (Pydantic $ref / Optional) -------------
    def _resolve_ref(self, ref: str) -> Dict[str, Any]:
        # Supported refs: #/$defs/Name or #/definitions/Name
        if not ref.startswith("#/"):
            raise ToonDecodeError(f"Unsupported $ref: {ref}")
        parts = ref.lstrip("#/").split("/")
        if len(parts) == 2 and parts[0] in ("$defs", "definitions"):
            name = parts[1]
            target = self._defs.get(name)
            if not isinstance(target, dict):
                raise ToonDecodeError(f"$ref not found: {ref}")
            return target
        raise ToonDecodeError(f"Unsupported $ref path: {ref}")

    def _resolve_schema(self, schema: Any) -> Any:
        # Resolve direct $ref chains. Merge sibling constraints with resolved schema.
        cur = schema
        seen = set()
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
                        s for s in schema[key]
                        if not (isinstance(s, dict) and s.get("type") == "null")
                    ]
                    if len(non_null) == 1:
                        return self._resolve_schema(non_null[0])
        return schema

    def _resolve_nullable(self, schema: Any) -> Any:
        return self._unwrap_nullable(schema)

    def decode(self, text: str) -> Dict[str, Any]:
        """TOON 텍스트를 딕셔너리로 디코딩 (Pydantic 검증 없이).
        
        Args:
            text: TOON 형식의 텍스트
            
        Returns:
            Dict[str, Any]: 디코딩된 딕셔너리 (Pydantic 검증 전)
        """
        # JSON 모드인 경우 (또는 자동 폴백된 경우) JSON으로 파싱
        effective_mode = getattr(self, '_effective_mode', self.cfg.instructions_mode)
        if effective_mode == "json":
            import json
            try:
                # JSON 추출 시도 (코드펜스 제거)
                s = (text or "").strip()
                # ```json ... ```, ```toon ... ```, ``` ... ``` 모두 제거
                if s.startswith("```") and s.endswith("```"):
                    lines = s.split("\n")
                    # 첫 줄이 ```json, ```toon, ``` 등인 경우 제거
                    s = "\n".join(lines[1:-1])
                return json.loads(s)
            except json.JSONDecodeError as e:
                raise ToonDecodeError(f"JSON parsing failed: {e}")
        
        # TOON 모드 파싱
        lines = self._normalize_lines(text)
        if not lines:
            raise ToonParserError("Empty input. Model produced no parsable TOON.")
        obj, next_i = self._parse_object(lines, 0, 0, self.schema, path="$")
        
        # remain check
        if next_i < len(lines):
            rest = [ln for ln in lines[next_i:] if ln.strip()]
            if rest:
                raise ToonParserError(f"Unparsed tail remains at line {next_i+1}: {rest[0]!r}")
        
        return obj

    def parse(self, text: str) -> BaseModel:
        # JSON 모드인 경우 (또는 자동 폴백된 경우) JSON으로 파싱
        effective_mode = getattr(self, '_effective_mode', self.cfg.instructions_mode)
        if effective_mode == "json":
            import json
            try:
                # JSON 추출 시도 (코드펜스 제거)
                s = (text or "").strip()
                # ```json ... ```, ```toon ... ```, ``` ... ``` 모두 제거
                if s.startswith("```") and s.endswith("```"):
                    lines = s.split("\n")
                    # 첫 줄이 ```json, ```toon, ``` 등인 경우 제거
                    s = "\n".join(lines[1:-1])
                obj = json.loads(s)
                return self.model.model_validate(obj)
            except json.JSONDecodeError as e:
                raise ToonDecodeError(f"JSON parsing failed: {e}")
        
        # TOON 모드 파싱
        lines = self._normalize_lines(text)
        if not lines:
            raise ToonParserError("Empty input. Model produced no parsable TOON.")
        obj, next_i = self._parse_object(lines, 0, 0, self.schema, path="$")

        # remain check
        if next_i < len(lines):
            rest = [ln for ln in lines[next_i:] if ln.strip()]
            if rest:
                raise ToonParserError(f"Unparsed tail remains at line {next_i+1}: {rest[0]!r}")

        return self.model.model_validate(obj)

    # ---------------- Normalization ----------------
    def _normalize_lines(self, text: str) -> List[str]:
        s = (text or "").strip()
        m = self._CODE_FENCE_RE.search(s)
        if m:
            s = m.group(1).strip()
        raw = s.splitlines()
        lines = [ln.rstrip() for ln in raw]
        lines = [ln for ln in lines if ln.strip()]
        return lines

    # ---------------- Helpers ----------------
    def _count_indent(self, line: str) -> int:
        if "\t" in line[: line.find(line.lstrip(" ")) if line.lstrip(" ") else 0]:
            raise ToonParserError("Tabs are not allowed. Use spaces only.")
        return len(line) - len(line.lstrip(" "))

    def _split_kv(self, stripped: str) -> Tuple[str, Optional[str]]:
        # split on first ":"
        if ":" not in stripped:
            # Be more tolerant: return None to indicate skip this line
            # This allows the parser to be more robust with malformed LLM output
            return "", None
        k, v = stripped.split(":", 1)
        k = k.strip()
        v = v.strip()
        if v == "":
            return k, None
        return k, v

    def _parse_scalar(self, raw: str, expected_schema: Optional[Dict[str, Any]] = None) -> Any:
        v = raw.strip()
        if v == "null":
            return None
        if v == "{}":
            return {}
        if v == "[]":
            return []
        # quoted
        if (len(v) >= 2) and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
            v = v[1:-1]

        expected_type = (expected_schema or {}).get("type")

        # preserve string if schema says string
        if expected_type == "string" and self.cfg.protect_string_ids:
            return v

        # boolean
        if v.lower() in ("true", "false"):
            return v.lower() == "true"

        # int / float
        # NOTE: no thousand separators supported
        if re.fullmatch(r"-?\d+", v):
            if expected_type == "number":
                return float(v)
            if expected_type == "integer":
                return int(v)
            if expected_type == "string":
                return v
            # heuristic: prefer int
            return int(v)

        if re.fullmatch(r"-?\d+\.\d+", v):
            if expected_type == "integer":
                # force int parse would fail; keep float
                return float(v)
            return float(v)

        return v

    def _csv_row(self, row_line: str) -> List[str]:
        # allow spaces; treat as CSV single line
        buf = io.StringIO(row_line)
        reader = csv.reader(buf, skipinitialspace=True)
        return next(reader, [])

    # ---------------- Schema helpers ----------------
    def _get_props(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        return schema.get("properties", {}) or {}

    def _schema_allows_additional(self, schema: Dict[str, Any]) -> bool:
        # dict[str, Any] or free-form object
        if schema.get("type") != "object":
            return False
        ap = schema.get("additionalProperties", None)
        if ap is True:
            return True
        # If additionalProperties is an object/schema (not just boolean), it's flexible
        if isinstance(ap, dict):
            return True
        # if no properties defined, treat as flexible
        if not (schema.get("properties") or {}):
            if ap is None:
                return True
        return False

    # ---------------- Parsers ----------------
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
                raise ToonParserError(f"Unexpected indent at {path} (line {i+1})")

            stripped = line.strip()

            # Tabular header
            tm = self._TABULAR_HEADER_RE.match(stripped)
            if tm:
                fname = tm.group("name")
                fsch = props.get(fname)
                if fsch is None and not allow_additional and self.cfg.strict_schema:
                    raise SchemaViolationError(f"Unknown field '{fname}' at {path}")
                val, i = self._parse_tabular(lines, i, indent, fname, fsch, path=f"{path}.{fname}")
                out[fname] = val
                continue

            # Scalar list
            sm = self._SCALAR_LIST_RE.match(stripped)
            if sm:
                fname = sm.group("name")
                n = int(sm.group("n"))
                body = sm.group("body")
                fsch = props.get(fname)
                if fsch is None and not allow_additional and self.cfg.strict_schema:
                    raise SchemaViolationError(f"Unknown field '{fname}' at {path}")
                items_schema = (fsch or {}).get("items", {}) if (fsch or {}).get("type") == "array" else None
                parts = [p.strip() for p in body.split(",")] if body else []
                parts = [p for p in parts if p != ""]
                if self.cfg.strict_count and len(parts) != n:
                    raise SchemaViolationError(f"{path}.{fname}: expected {n} items, got {len(parts)}")
                out[fname] = [self._parse_scalar(p, items_schema) for p in parts]
                i += 1
                continue

            # Indexed array item
            im = self._INDEXED_ITEM_RE.match(stripped)
            if im:
                fname = im.group("name")
                idx = int(im.group("idx"))
                fsch = props.get(fname)
                if fsch is None and not allow_additional and self.cfg.strict_schema:
                    raise SchemaViolationError(f"Unknown field '{fname}' at {path}")
                if (fsch or {}).get("type") != "array":
                    raise SchemaViolationError(f"{path}.{fname}[{idx}]: indexed notation used for non-array field")
                items_schema = (fsch or {}).get("items", {}) or {}
                # parse nested object block
                i += 1
                obj, i = self._parse_object(lines, i, indent + self.cfg.indent_step, items_schema, path=f"{path}.{fname}[{idx}]")
                indexed_buffers.setdefault(fname, {})[idx] = obj
                continue

            # key/value or block
            key, val = self._split_kv(stripped)
            
            # Skip invalid lines (no colon found)
            if key == "" and val is None:
                i += 1
                continue
            
            fsch = self._resolve_nullable(props.get(key))

            if fsch is None and not allow_additional and self.cfg.strict_schema:
                raise SchemaViolationError(f"Unknown field '{key}' at {path}")

            if val is not None:
                out[key] = self._parse_scalar(val, fsch)
                i += 1
                continue

            # block
            i += 1
            # empty block -> treat as {} or []
            if i >= len(lines) or self._count_indent(lines[i]) <= indent:
                if (fsch or {}).get("type") == "array":
                    out[key] = []
                else:
                    out[key] = {}
                continue

            # Check if next line starts with dash (array) regardless of schema
            next_line_is_dash = False
            if i < len(lines):
                next_stripped = lines[i].strip()
                next_line_is_dash = next_stripped.startswith("-")
            
            # decide array/object
            if (fsch or {}).get("type") == "array" or next_line_is_dash:
                # Parse as array if schema says array OR if content starts with dash
                arr, i = self._parse_array_block(lines, i, indent + self.cfg.indent_step, fsch or {"type": "array", "items": {}}, path=f"{path}.{key}")
                out[key] = arr
            elif (fsch or {}).get("type") == "object" or self._schema_allows_additional(fsch or {}) or not fsch:
                # Parse as object if schema says object, allows additional properties, or schema is undefined
                obj, i = self._parse_object(lines, i, indent + self.cfg.indent_step, fsch or {}, path=f"{path}.{key}")
                out[key] = obj
            else:
                # Scalar field with block content - be more tolerant
                # Try to parse as object anyway (LLM might have output wrong format)
                # If schema is not defined or ambiguous, treat as flexible object
                try:
                    obj, i = self._parse_object(lines, i, indent + self.cfg.indent_step, {}, path=f"{path}.{key}")
                    out[key] = obj
                except Exception:
                    # If parsing as object fails, skip the block and set to empty dict
                    while i < len(lines) and self._count_indent(lines[i]) > indent:
                        i += 1
                    out[key] = {}

        # finalize indexed buffers
        for fname, buf in indexed_buffers.items():
            if fname in out:
                raise ToonParserError(f"{path}.{fname}: both indexed notation and other notation used")
            max_idx = max(buf.keys()) if buf else -1
            lst: List[Any] = [None] * (max_idx + 1)
            for idx, obj in buf.items():
                lst[idx] = obj
            # compact: drop None tail? keep for schema? keep as-is; pydantic will raise if None not allowed
            out[fname] = lst

        return out, i

    def _parse_array_block(
        self, lines: List[str], i: int, indent: int, field_schema: Dict[str, Any], path: str
    ) -> Tuple[List[Any], int]:
        # Only dash lists are accepted in block arrays
        out: List[Any] = []
        items_schema = field_schema.get("items", {}) or {}

        while i < len(lines):
            line = lines[i]
            cur_indent = self._count_indent(line)
            if cur_indent < indent:
                break
            if cur_indent > indent:
                raise ToonParserError(f"Unexpected indent inside array at {path} (line {i+1})")

            stripped = line.strip()
            if not stripped.startswith("- "):
                # Also accept just "-" without space for robustness
                if not stripped.startswith("-"):
                    break
                else:
                    # Handle "- key: value" or "-key: value" (missing space after dash)
                    item_text = stripped[1:].strip()
            else:
                item_text = stripped[2:].strip()

            # object item " - key: value" or " - key:"
            if ":" in item_text:
                k, v = item_text.split(":", 1)
                k = k.strip()
                v = v.strip()
                obj: Dict[str, Any] = {}
                props = self._get_props(items_schema) if items_schema.get("type") in (None, "object") else {}

                # first key
                if v == "":
                    # nested block for this key
                    i += 1
                    nested, i = self._parse_value_block(lines, i, indent + self.cfg.indent_step, props.get(k), path=f"{path}[-].{k}")
                    obj[k] = nested
                else:
                    obj[k] = self._parse_scalar(v, props.get(k))

                i += 1
                # consume additional fields for this object at indent+step
                while i < len(lines):
                    ln = lines[i]
                    ind = self._count_indent(ln)
                    if ind < indent + self.cfg.indent_step:
                        break
                    # Don't break on deeper indents - they might be nested arrays/objects
                    if ind == indent + self.cfg.indent_step:
                        # Same level field
                        stripped_ln = ln.strip()
                        # Skip if it's a dash (part of nested array)
                        if stripped_ln.startswith("-"):
                            break
                        sk, sv = self._split_kv(stripped_ln)
                        # Skip invalid lines
                        if sk == "" and sv is None:
                            i += 1
                            continue
                        psch = props.get(sk)
                        if sv is None:
                            i += 1
                            # Resolve schema before passing to _parse_value_block
                            resolved_psch = self._resolve_nullable(psch) if psch else None
                            nested, i = self._parse_value_block(lines, i, indent + 2 * self.cfg.indent_step, resolved_psch, path=f"{path}[-].{sk}")
                            obj[sk] = nested
                        else:
                            obj[sk] = self._parse_scalar(sv, psch)
                            i += 1
                    else:
                        # Deeper indent - it's part of a nested structure, let recursion handle it
                        break

                out.append(obj)
                continue

            # scalar item "- value"
            out.append(self._parse_scalar(item_text, items_schema))
            i += 1

        return out, i

    def _parse_value_block(
        self, lines: List[str], i: int, indent: int, schema: Optional[Dict[str, Any]], path: str
    ) -> Tuple[Any, int]:
        schema = schema or {}
        # decide by next line: dash list or object
        if i >= len(lines):
            return {}, i
        next_indent = self._count_indent(lines[i])
        if next_indent < indent:
            return {}, i

        # Check actual content: if next line starts with dash, it's an array
        next_line_stripped = lines[i].strip()
        content_is_array = next_line_stripped.startswith("-")
        
        # Prefer content-based detection over schema
        if content_is_array:
            # Content shows it's an array, parse as array
            return self._parse_array_block(lines, i, indent, schema or {"type": "array", "items": {}}, path)
        
        if schema.get("type") == "array":
            return self._parse_array_block(lines, i, indent, schema, path)
        if schema.get("type") == "object" or self._schema_allows_additional(schema):
            return self._parse_object(lines, i, indent, schema, path)
        # fallback: treat as object
        return self._parse_object(lines, i, indent, schema, path)

    def _parse_tabular(
        self, lines: List[str], i: int, indent: int, fname: str, field_schema: Optional[Dict[str, Any]], path: str
    ) -> Tuple[List[Dict[str, Any]], int]:
        # header line already matched
        header = lines[i].strip()
        m = self._TABULAR_HEADER_RE.match(header)
        assert m
        n = int(m.group("n"))
        cols = [c.strip() for c in m.group("cols").split(",") if c.strip()]

        fsch = field_schema or {}
        if fsch.get("type") != "array":
            raise SchemaViolationError(f"{path}: tabular used for non-array field")

        # policy: forbid tabular for complex arrays
        if ToonIntelligence.is_complex_array(fsch, self.cfg.complexity_threshold):
            raise PolicyViolationError(
                f"{path}: tabular(items[N,]{{cols}}) is FORBIDDEN for complex arrays. Use Dash(-) notation."
            )

        items_schema = self._resolve_nullable(fsch.get("items", {}) or {})
        item_props = items_schema.get("properties", {}) or {}

        if self.cfg.strict_schema and item_props:
            unknown_cols = [c for c in cols if c not in item_props]
            if unknown_cols:
                raise SchemaViolationError(f"{path}: unknown tabular columns: {unknown_cols}")

        out: List[Dict[str, Any]] = []
        i += 1
        row_idx = 0
        while i < len(lines):
            line = lines[i]
            cur_indent = self._count_indent(line)
            if cur_indent <= indent:
                break
            if cur_indent != indent + self.cfg.indent_step:
                # tolerate deeper indent by trimming left, but keep simple rule
                if cur_indent < indent + self.cfg.indent_step:
                    break

            row = self._csv_row(line.strip())

            # Row Index Defense
            if len(row) == len(cols) + 1:
                c0 = (row[0] or "").strip()
                if c0.isdigit():
                    v = int(c0)
                    if v == row_idx or v == row_idx + 1:
                        row = row[1:]

            if self.cfg.strict_count and row_idx >= n:
                break

            # ensure length
            if len(row) < len(cols):
                # pad None
                row = row + ["null"] * (len(cols) - len(row))
            if len(row) > len(cols):
                # keep extra as error (unless it's benign whitespace)
                raise SchemaViolationError(f"{path}: too many columns in row {row_idx}: got {len(row)}, expected {len(cols)}")

            d: Dict[str, Any] = {}
            for j, c in enumerate(cols):
                expected = item_props.get(c) if item_props else None
                d[c] = self._parse_scalar(row[j], expected)
            out.append(d)

            row_idx += 1
            i += 1
            if self.cfg.strict_count and row_idx >= n:
                break

        # if strict_count, enforce n
        if self.cfg.strict_count and row_idx != n:
            raise SchemaViolationError(f"{path}: expected {n} rows, got {row_idx}")

        return out, i
