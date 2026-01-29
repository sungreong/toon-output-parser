# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

Scalar = Union[str, int, float, bool, None]

_TABULAR_HEADER_RE = re.compile(r"^(?P<name>[A-Za-z0-9_]+)?\[(?P<n>\d+),(?:#)?\]\{(?P<cols>[^}]+)\}:\s*$")
_LIST_INLINE_RE = re.compile(r"^(?P<name>[A-Za-z0-9_]+)\[(?P<n>\d+)\]:\s*(?P<vals>.*)$")

def _parse_scalar(raw: str) -> Scalar:
    s = raw.strip()
    if s == "":
        return ""
    low = s.lower()
    if low in {"null", "none"}:
        return None
    if low in {"true", "false"}:
        return low == "true"
    if re.fullmatch(r"[-+]?\d+", s):
        try:
            return int(s)
        except ValueError:
            return s
    if re.fullmatch(r"[-+]?\d+\.\d+", s):
        try:
            return float(s)
        except ValueError:
            return s
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s

@dataclass
class SimpleToon:
    """TOON의 빈번한 패턴만 커버하는 내장 fallback 파서.

    지원(부분):
    - 객체: `key: value` / 중첩은 2-space indent 기반
    - 리스트(인라인): `tags[3]: a,b,c`
    - 탭уляр 배열: `items[2,]{id,name}:` + 다음 줄 CSV row

    한계:
    - spec 전체를 100% 구현하지 않습니다.
    - 운영/프로덕션에서는 공식 백엔드(toon_format)를 설치하는 것을 강하게 권장합니다.
    """

    indent: int = 2

    def encode(self, value: Any) -> str:
        return self._encode_any(value, level=0).rstrip() + "\n"

    def decode(self, text: str) -> Any:
        lines = [ln.rstrip("\n") for ln in text.splitlines() if ln.strip() != ""]
        if not lines:
            return {}
        if _is_tabular_header(lines[0]):
            rows, _ = self._decode_tabular_block(lines, start_index=0, base_indent=0)
            return rows
        obj, _ = self._decode_object(lines, start_index=0, base_indent=0)
        return obj

    def _encode_any(self, value: Any, level: int) -> str:
        pad = " " * (self.indent * level)
        if isinstance(value, dict):
            out = []
            for k, v in value.items():
                if isinstance(v, dict):
                    out.append(f"{pad}{k}:")
                    out.append(self._encode_any(v, level + 1))
                elif isinstance(v, list):
                    if _is_uniform_dict_list(v):
                        out.append(self._encode_tabular(k, v, level))
                    else:
                        out.append(f"{pad}{k}[{len(v)}]: " + ",".join(_scalar_to_str(x) for x in v))
                else:
                    out.append(f"{pad}{k}: {_scalar_to_str(v)}")
            return "\n".join(out)
        if isinstance(value, list):
            if _is_uniform_dict_list(value):
                return self._encode_tabular(None, value, level)
            return pad + f"[{len(value)}]: " + ",".join(_scalar_to_str(x) for x in value)
        return pad + _scalar_to_str(value)

    def _encode_tabular(self, name: str | None, rows: List[Dict[str, Any]], level: int) -> str:
        pad = " " * (self.indent * level)
        cols = list(rows[0].keys()) if rows else []
        head_name = f"{name}" if name else ""
        header = f"{pad}{head_name}[{len(rows)},]" + "{" + ",".join(cols) + "}:"
        out = [header]
        for r in rows:
            out.append(pad + " " * self.indent + ",".join(_scalar_to_str(r.get(c)) for c in cols))
        return "\n".join(out)

    def _decode_object(self, lines: List[str], start_index: int, base_indent: int) -> Tuple[Dict[str, Any], int]:
        obj: Dict[str, Any] = {}
        indexed_fields: Dict[str, List[Any]] = {}  # Collect items[0], items[1], etc.
        i = start_index
        while i < len(lines):
            ln = lines[i]
            indent = _count_indent(ln)
            if indent < base_indent:
                break
            # indent > base_indent인 경우는 자식 객체나 리스트이므로 상위에서 처리
            if indent > base_indent:
                break

            stripped = ln.strip()

            m_list = _LIST_INLINE_RE.match(stripped)
            if m_list:
                key = m_list.group("name")
                vals_raw = m_list.group("vals").strip()
                vals = [] if vals_raw == "" else [v.strip() for v in vals_raw.split(",")]
                obj[key] = [_parse_scalar(v) for v in vals]
                i += 1
                continue

            # tabular object list (key inside header)
            if _is_tabular_header(stripped):
                rows, next_i = self._decode_tabular_block(lines, i, base_indent)
                # if header has name, decode_tabular_block returns rows; put under that name
                name = _tabular_name(stripped)
                obj[name or "items"] = rows
                i = next_i
                continue

            # Skip lines without colon (invalid TOON lines, but be more tolerant)
            if ":" not in stripped:
                # Could be a malformed line or comment - skip it
                i += 1
                continue

            key, rest = stripped.split(":", 1)
            key = key.strip()
            rest = rest.strip()
            
            # Detect indexed notation (items[0], items[1], etc.)
            indexed_match = re.match(r'^([A-Za-z0-9_]+)\[(\d+)\]$', key)
            if indexed_match:
                field_name = indexed_match.group(1)
                index = int(indexed_match.group(2))
                
                # Collect object
                if rest == "":
                    child, next_i = self._decode_object(lines, i + 1, base_indent + self.indent)
                    if field_name not in indexed_fields:
                        indexed_fields[field_name] = []
                    # Expand to fit index position
                    while len(indexed_fields[field_name]) <= index:
                        indexed_fields[field_name].append(None)
                    indexed_fields[field_name][index] = child
                    i = next_i
                else:
                    # Scalar value
                    if field_name not in indexed_fields:
                        indexed_fields[field_name] = []
                    while len(indexed_fields[field_name]) <= index:
                        indexed_fields[field_name].append(None)
                    indexed_fields[field_name][index] = _parse_scalar(rest)
                    i += 1
                continue
            
            # 빈 리스트 처리: `skills: []` 같은 경우
            if rest == "[]":
                obj[key] = []
                i += 1
                continue
            
            # 빈 딕셔너리 처리: `data: {}` 같은 경우
            if rest == "{}":
                obj[key] = {}
                i += 1
                continue
            
            if rest == "":
                # 다음 줄들을 확인하여 리스트인지 객체인지 판단
                if i + 1 < len(lines):
                    next_ln = lines[i + 1]
                    next_indent = _count_indent(next_ln)
                    next_stripped = next_ln.strip()
                    
                    # 다음 줄이 `-`로 시작하면 리스트
                    if next_stripped.startswith("-") and next_indent == base_indent + self.indent:
                        list_items, next_i = self._decode_list_items(lines, i + 1, base_indent + self.indent)
                        obj[key] = list_items
                        i = next_i
                        continue
                
                # 리스트가 아니면 객체로 처리
                child, next_i = self._decode_object(lines, i + 1, base_indent + self.indent)
                obj[key] = child
                i = next_i
            else:
                obj[key] = _parse_scalar(rest)
                i += 1
        
        # Convert indexed fields to regular lists
        for field_name, items_list in indexed_fields.items():
            obj[field_name] = [item for item in items_list if item is not None]
        
        return obj, i
    
    def _decode_list_items(self, lines: List[str], start_index: int, base_indent: int) -> Tuple[List[Any], int]:
        """`-`로 시작하는 리스트 항목들을 파싱. 스칼라와 객체 모두 지원."""
        items: List[Any] = []
        i = start_index
        while i < len(lines):
            ln = lines[i]
            indent = _count_indent(ln)
            # base_indent보다 작으면 리스트가 끝남
            if indent < base_indent:
                break
            
            stripped = ln.strip()
            # `-`로 시작하지 않으면 리스트가 끝남
            if not stripped.startswith("-"):
                if indent == base_indent:
                    break
                # 더 깊은 들여쓰기는 무시 (중첩된 구조)
                i += 1
                continue
            
            # 들여쓰기가 base_indent와 정확히 일치하는 경우만 처리
            if indent == base_indent:
                # `- ` 제거 후 값 파싱
                value_str = stripped[1:].strip()
                
                # value_str 자체가 `key: value` 형식이면 인라인 객체의 시작
                is_inline_object = ":" in value_str and not value_str.startswith(('"', "'"))
                
                # 다음 줄을 확인하여 객체가 계속되는지 확인
                has_next_object_line = False
                if i + 1 < len(lines):
                    next_ln = lines[i + 1]
                    next_indent = _count_indent(next_ln)
                    next_stripped = next_ln.strip()
                    
                    # 다음 줄이 더 깊은 들여쓰기이고 `:`를 포함하고 `-`로 시작하지 않으면 객체 계속
                    has_next_object_line = (next_indent == base_indent + self.indent and 
                                            ":" in next_stripped and 
                                            not next_stripped.startswith("-"))
                
                # 인라인 객체이거나 다음 줄이 객체인 경우
                if is_inline_object or has_next_object_line:
                    # 객체로 파싱
                    # 인라인 객체인 경우: 현재 줄부터 시작
                    if is_inline_object:
                        # 인라인 객체: `- name: value` 형식을 객체로 변환
                        # 객체 라인들을 수집
                        obj_lines = []
                        # 첫 줄: `- name: value` -> `name: value` (들여쓰기 제거)
                        obj_lines.append(value_str)
                        
                        # 다음 줄들도 추가 (객체가 계속되는 경우)
                        j = i + 1
                        while j < len(lines):
                            next_ln = lines[j]
                            next_indent = _count_indent(next_ln)
                            next_stripped = next_ln.strip()
                            
                            # base_indent보다 작으면 리스트 항목이 끝남
                            if next_indent < base_indent:
                                break
                            # base_indent와 같고 `-`로 시작하면 다음 리스트 항목
                            if next_indent == base_indent and next_stripped.startswith("-"):
                                break
                            # base_indent + self.indent이고 `:`를 포함하면 객체 필드
                            if next_indent == base_indent + self.indent:
                                if ":" in next_stripped:
                                    obj_lines.append(next_stripped)
                                    j += 1
                                else:
                                    break
                            # 더 깊은 들여쓰기 (중첩 객체/리스트) - 들여쓰기 조정 필요
                            elif next_indent > base_indent + self.indent:
                                # base_indent + self.indent만큼 들여쓰기 제거
                                indent_to_remove = base_indent + self.indent
                                adjusted_line = " " * (next_indent - indent_to_remove) + next_stripped
                                obj_lines.append(adjusted_line)
                                j += 1
                            else:
                                break
                        
                        # 객체 파싱 (들여쓰기 0 기준)
                        obj, _ = self._decode_object(obj_lines, 0, 0)
                        items.append(obj)
                        i = j
                    else:
                        # 다음 줄부터 객체 시작 (빈 `-` 다음에 객체)
                        obj, next_i = self._decode_object(lines, i + 1, base_indent + self.indent)
                        items.append(obj)
                        i = next_i
                    continue
                
                # 스칼라 값으로 파싱
                if value_str == "":
                    items.append("")
                else:
                    items.append(_parse_scalar(value_str))
                i += 1
            else:
                # 들여쓰기가 다르면 리스트가 끝남
                break
        
        return items, i

    def _decode_tabular_block(self, lines: List[str], start_index: int, base_indent: int) -> Tuple[List[Dict[str, Any]], int]:
        header = lines[start_index].strip()
        m = _TABULAR_HEADER_RE.match(header)
        if not m:
            raise ValueError(f"Invalid tabular header: {header}")
        n = int(m.group("n"))
        cols = [c.strip() for c in m.group("cols").split(",") if c.strip()]
        rows: List[Dict[str, Any]] = []
        i = start_index + 1
        for _ in range(n):
            if i >= len(lines):
                break
            ln = lines[i]
            if _count_indent(ln) < base_indent + self.indent:
                break
            row_str = ln.strip()
            row = _parse_csv_row(row_str)
            # Row-index defense: if the LLM prepended a numeric row number, drop it.
            if len(row) == len(cols) + 1 and str(row[0]).strip().isdigit():
                row = row[1:]
            # Safety: if row is still longer than cols, truncate to avoid column drift.
            if len(row) > len(cols):
                row = row[: len(cols)]
            d = {cols[j]: _parse_scalar(row[j]) if j < len(row) else None for j in range(len(cols))}
            rows.append(d)
            i += 1
        return rows, i

def _parse_csv_row(row_str: str) -> List[str]:
    f = io.StringIO(row_str)
    reader = csv.reader(f, delimiter=",", quotechar='"', skipinitialspace=False)
    return next(reader)

def _scalar_to_str(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)

def _count_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))

def _is_uniform_dict_list(v: Any) -> bool:
    if not isinstance(v, list) or not v:
        return False
    if not all(isinstance(x, dict) for x in v):
        return False
    keys = list(v[0].keys())
    return all(list(x.keys()) == keys for x in v)

def _is_tabular_header(line: str) -> bool:
    return bool(_TABULAR_HEADER_RE.match(line.strip()))

def _tabular_name(line: str) -> str | None:
    m = _TABULAR_HEADER_RE.match(line.strip())
    if not m:
        return None
    return m.group("name")
