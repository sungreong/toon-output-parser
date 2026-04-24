# -*- coding: utf-8 -*-
from __future__ import annotations

from .toon_parser_ultimate import ToonIntelligence


class ToonPromptBuilder(ToonIntelligence):
    """Schema-aware TOON prompt builder.

    Kept as a named boundary so applications can depend on prompt generation
    separately from decoding while the legacy ToonIntelligence alias remains.
    """


__all__ = ["ToonPromptBuilder"]
