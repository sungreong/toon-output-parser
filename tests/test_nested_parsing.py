from __future__ import annotations

from pydantic import BaseModel, Field

from toon_langchain_parser import ToonOutputParser


class Trait(BaseModel):
    name: str
    level: str = ""


class Person(BaseModel):
    name: str
    traits: list[Trait] = Field(default_factory=list)


def test_nested_array_parsing():
    parser = ToonOutputParser(model=Person)
    out = parser.parse(
        "name: John\n"
        "traits:\n"
        "  - name: Brave\n"
        "    level: high\n"
        "  - name: Calm\n"
        "    level: medium\n"
    )
    assert out.name == "John"
    assert len(out.traits) == 2
    assert out.traits[0].name == "Brave"
    assert out.traits[1].level == "medium"
