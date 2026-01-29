# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, Optional, Type

from pydantic import BaseModel

from .toon_parser_ultimate import ParserConfig, ToonParser


def build_toon_format_prompt(model: Type[BaseModel], cfg: Optional[ParserConfig] = None) -> str:
    """스키마 기반(Adaptive) TOON 포맷 지시문을 생성합니다.
    
    이제 ToonParser 내부에서 스키마 기반 동적 가이드를 생성하므로,
    여기서는 최소한의 추가 예시만 제공합니다.
    """
    parser = ToonParser(model=model, cfg=cfg or ParserConfig())
    instructions = parser.get_format_instructions()
    
    # 간단한 시각적 예시만 추가 (중복 제거)
    visual_example = """

### VISUAL LEARNING: COMPLETE EXAMPLE

```toon
name: John Doe
age: 30
active: true
score: 95.5

contact:
  email: john@example.com
  phone: 555-0123

tags[3]: developer,python,ai

projects:
  - title: Project A
    status: active
    budget: 50000
  - title: Project B
    status: completed
    budget: 75000

metadata:
  created_at: 2024-01-15
  updated_at: 2024-01-20
```

💡 **Tip:** When in doubt, use dash list format - it works for all arrays.
"""
    
    return instructions + visual_example


def _dummy_scalar(schema: Dict[str, Any]) -> Any:
    t = schema.get("type")
    if t == "string":
        return "example"
    if t == "integer":
        return 1
    if t == "number":
        return 0.1
    if t == "boolean":
        return True
    return None


def _dummy_from_schema(schema: Dict[str, Any], depth: int = 0, max_depth: int = 2) -> Any:
    if depth >= max_depth:
        return {} if schema.get("type") == "object" else None

    t = schema.get("type")
    if t in ("string", "integer", "number", "boolean"):
        return _dummy_scalar(schema)

    if t == "object":
        props = schema.get("properties") or {}
        out: Dict[str, Any] = {}
        for k, v in list(props.items())[:6]:
            out[k] = _dummy_from_schema(v, depth + 1, max_depth)
        return out

    if t == "array":
        item = schema.get("items") or {}
        return [
            _dummy_from_schema(item, depth + 1, max_depth),
            _dummy_from_schema(item, depth + 1, max_depth),
        ]

    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema and schema[key]:
            return _dummy_from_schema(schema[key][0], depth, max_depth)

    return None


def build_toon_example(model: Type[BaseModel], cfg: Optional[ParserConfig] = None) -> str:
    """구조 학습용 '작은' TOON 예시를 생성합니다."""
    from .simple_toon import SimpleToon
    
    parser = ToonParser(model=model, cfg=cfg or ParserConfig())
    schema = model.model_json_schema()
    props = schema.get("properties") or {}
    example_obj: Dict[str, Any] = {}
    for k, v in list(props.items())[:8]:
        example_obj[k] = _dummy_from_schema(v, 0, 2)
    
    encoder = SimpleToon()
    toon_output = encoder.encode(example_obj).rstrip()
    
    # Add helpful header
    header = "# Example TOON output for this schema:\n# Remember: key:value on every line, dash+space for arrays\n\n"
    
    return "```toon\n" + header + toon_output + "\n```\n"
