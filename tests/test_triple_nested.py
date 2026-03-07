from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from toon_langchain_parser import ToonOutputParser


class PersonTrait(BaseModel):
    name: str
    level: str = ""
    description: str = ""


class Person(BaseModel):
    name: str
    age: int | None = None
    role: str = ""
    traits: list[PersonTrait] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    background: str = ""


class Team(BaseModel):
    team_name: str
    total_members: int
    members: list[Person] = Field(default_factory=list)


def test_triple_nested_array_parsing():
    parser = ToonOutputParser(model=Team)
    out = parser.parse(
        "team_name: Alpha Team\n"
        "total_members: 2\n"
        "members:\n"
        "  - name: Nori\n"
        "    age: 17\n"
        "    role: leader\n"
        "    traits:\n"
        "      - name: brave\n"
        "        level: high\n"
        "        description: handles risk\n"
        "    skills[2]: planning,coordination\n"
        "    background: experienced\n"
        "  - name: Mina\n"
        "    age: 14\n"
        "    role: analyst\n"
        "    traits:\n"
        "      - name: detail\n"
        "        level: medium\n"
        "        description: checks edge cases\n"
        "    skills: []\n"
        "    background: curious\n"
    )

    assert out.team_name == "Alpha Team"
    assert out.total_members == 2
    assert len(out.members) == 2
    assert out.members[0].traits[0].name == "brave"
    assert out.members[1].traits[0].description == "checks edge cases"


def test_triple_nested_requires_dash_for_object_array_items():
    parser = ToonOutputParser(model=Team)
    bad = (
        "team_name: Alpha Team\n"
        "total_members: 1\n"
        "members:\n"
        "  name: Nori\n"
    )
    with pytest.raises(ValueError):
        parser.parse(bad)
