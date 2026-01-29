# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[- ]?)?(?:\d{2,4}[- ]?)\d{3,4}[- ]?\d{4}\b")
_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

@dataclass(frozen=True)
class RawLogPolicy:
    """원문(LLM 출력) 로깅 정책.

    - 기본값: 원문 미로깅.
    - 개발 환경에서만 활성화 추천.
    """
    enabled: bool
    preview_chars: int = 200

    @staticmethod
    def from_env() -> "RawLogPolicy":
        enabled = os.getenv("TOON_PARSER_LOG_RAW", "false").lower() == "true"
        try:
            preview_chars = int(os.getenv("TOON_PARSER_LOG_PREVIEW_CHARS", "200"))
        except ValueError:
            preview_chars = 200
        return RawLogPolicy(enabled=enabled, preview_chars=max(0, preview_chars))

def mask_pii_text(text: str) -> str:
    """간단한 PII 마스킹(정규식 기반).

    한계:
    - 완전한 PII 탐지는 불가능(특히 자유형 텍스트).
    - 운영환경에서는 별도 DLP(Data Loss Prevention)와 병행 권장.
    """
    text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = _PHONE_RE.sub("[REDACTED_PHONE]", text)
    text = _CARD_RE.sub("[REDACTED_CARD]", text)
    return text

def safe_raw_preview(text: str, policy: Optional[RawLogPolicy] = None) -> str:
    """정책에 따라 원문 미리보기 문자열을 반환."""
    if policy is None:
        policy = RawLogPolicy.from_env()
    if not policy.enabled:
        return "REDACTED"
    preview = text[: policy.preview_chars]
    return mask_pii_text(preview)
