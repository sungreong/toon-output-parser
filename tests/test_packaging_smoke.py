from __future__ import annotations

from pydantic import BaseModel

from toon_langchain_parser import ParserConfig, ToonOutputParser, ToonParser


class SmokeModel(BaseModel):
    name: str
    age: int


def test_top_level_exports_are_core_safe():
    import toon_langchain_parser as pkg

    assert "ToonOutputParser" in pkg.__all__
    assert "ToonParser" in pkg.__all__
    assert "ToonBackend" not in pkg.__all__
    assert "AutoToonBackend" not in pkg.__all__


def test_toon_output_parser_basic_parse_without_langchain():
    parser = ToonOutputParser(model=SmokeModel, cfg=ParserConfig(instructions_mode="adaptive"))
    result = parser.parse("name: John\nage: 30")
    assert result.name == "John"
    assert result.age == 30


def test_toon_parser_basic_parse_without_langchain():
    parser = ToonParser(model=SmokeModel, cfg=ParserConfig(instructions_mode="adaptive"))
    result = parser.parse("name: Alice\nage: 28")
    assert result.name == "Alice"
    assert result.age == 28
