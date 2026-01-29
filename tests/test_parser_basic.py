from __future__ import annotations

import pytest
from pydantic import BaseModel, Field
from toon_langchain_parser import ToonOutputParser

class M(BaseModel):
    route: str
    confidence: float = Field(..., ge=0, le=1)
    reason: str

def test_parse_success_dict_to_model():
    parser = ToonOutputParser(model=M)
    out = parser.parse("route: search\nconfidence: 0.9\nreason: ok\n")
    assert out.route == "search"
    assert out.confidence == 0.9

def test_schema_mismatch_raises():
    parser = ToonOutputParser(model=M)
    with pytest.raises(Exception):
        parser.parse("route: search\nconfidence: not_a_number\nreason: ok\n")

def test_extract_from_code_fence():
    parser = ToonOutputParser(model=M)
    text = "```toon\nroute: faq\nconfidence: 0.5\nreason: x\n```"
    out = parser.parse(text)
    assert out.route == "faq"
