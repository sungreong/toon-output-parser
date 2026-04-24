# -*- coding: utf-8 -*-
from __future__ import annotations

from .toon_parser_ultimate import (
    PolicyViolationError,
    SchemaViolationError,
    ToonDecodeError,
    ToonParser,
    ToonParserError,
)


class ToonDecoder(ToonParser):
    """TOON decoder/validator boundary.

    The class intentionally subclasses ToonParser to preserve the existing
    public parser while exposing a decoder-focused import path.
    """


__all__ = [
    "PolicyViolationError",
    "SchemaViolationError",
    "ToonDecodeError",
    "ToonDecoder",
    "ToonParserError",
]
