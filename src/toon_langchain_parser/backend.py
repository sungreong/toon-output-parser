# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .simple_toon import SimpleToon

@runtime_checkable
class ToonBackend(Protocol):
    """TOON 인코딩/디코딩 백엔드 인터페이스."""
    def encode(self, value: Any) -> str: ...
    def decode(self, text: str) -> Any: ...

@dataclass(frozen=True)
class AutoToonBackend:
    """가능한 경우 공식/외부 라이브러리를 사용하고, 없으면 내장 fallback 사용."""
    _fallback: SimpleToon = field(default_factory=SimpleToon)

    def encode(self, value: Any) -> str:
        backend = _select_backend()
        return backend.encode(value)

    def decode(self, text: str) -> Any:
        backend = _select_backend()
        return backend.decode(text)

def _select_backend() -> ToonBackend:
    # 1) Official: toon-format/toon-python (모듈: toon_format)
    try:
        import toon_format  # type: ignore
        if hasattr(toon_format, "encode") and hasattr(toon_format, "decode"):
            return _ToonFormatBackend()
    except Exception:
        pass

    # 2) Legacy: python-toon (모듈: toon)
    try:
        import toon  # type: ignore
        if hasattr(toon, "encode") and hasattr(toon, "decode"):
            return _ToonModuleBackend()
    except Exception:
        pass

    # 3) Fallback: partial parser
    return SimpleToon()

class _ToonFormatBackend:
    def encode(self, value: Any) -> str:
        import toon_format  # type: ignore
        return toon_format.encode(value)

    def decode(self, text: str) -> Any:
        import toon_format  # type: ignore
        return toon_format.decode(text)

class _ToonModuleBackend:
    def encode(self, value: Any) -> str:
        import toon  # type: ignore
        return toon.encode(value)

    def decode(self, text: str) -> Any:
        import toon  # type: ignore
        return toon.decode(text)
