from __future__ import annotations

import json

from pydantic import BaseModel, Field

from eval_metrics import print_evaluation
from toon_langchain_parser import ToonOutputParser
from toon_langchain_parser.toon_parser_ultimate import ParserConfig


class Personality(BaseModel):
    traits: list[str] = Field(default_factory=list)
    summary: str = ""


class Appearance(BaseModel):
    features: list[str] = Field(default_factory=list)
    summary: str = ""


class Details(BaseModel):
    personality: Personality = Field(default_factory=Personality)
    appearance: Appearance = Field(default_factory=Appearance)


class CharacterFeatures(BaseModel):
    name: str
    age: int | None = None
    details: Details = Field(default_factory=Details)


class Trait(BaseModel):
    name: str
    level: str = ""


class Member(BaseModel):
    name: str
    traits: list[Trait] = Field(default_factory=list)


class Team(BaseModel):
    team_name: str
    members: list[Member] = Field(default_factory=list)


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def run_character_case() -> None:
    print_header("CASE 1: dotted path + inline list[str] + empty scalar")
    parser = ToonOutputParser(model=CharacterFeatures, cfg=ParserConfig(instructions_mode="adaptive"))

    raw_toon = """name: Harry Potter
age: 17
details.personality.traits: brave, loyal, kind
details.personality.summary:
details.appearance.features: black hair, green eyes, lightning scar
details.appearance.summary:
"""

    result = parser.parse(raw_toon)
    expected = {
        "name": "Harry Potter",
        "age": 17,
        "details": {
            "personality": {"traits": ["brave", "loyal", "kind"], "summary": ""},
            "appearance": {"summary": ""},
        },
    }

    print("RAW:")
    print(raw_toon)
    print("PARSED:")
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    print_evaluation("QUALITY", result.model_dump(), expected)


def run_team_case() -> None:
    print_header("CASE 2: list[object] coercion from inline scalar")
    parser = ToonOutputParser(model=Team, cfg=ParserConfig(instructions_mode="adaptive"))

    raw_toon = """team_name: Team Atlas
members:
  - name: Minji
    traits: detail-oriented
  - name: Sora
    traits: fast at prototyping
"""

    result = parser.parse(raw_toon)
    expected = {
        "team_name": "Team Atlas",
        "members": [{"name": "Minji"}, {"name": "Sora"}],
    }

    print("RAW:")
    print(raw_toon)
    print("PARSED:")
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    print_evaluation("QUALITY", result.model_dump(), expected)


def run_nested_fence_case() -> None:
    print_header("CASE 3: nested code fence cleanup")
    parser = ToonOutputParser(model=CharacterFeatures, cfg=ParserConfig(instructions_mode="adaptive"))

    raw_toon = """```toon
```TOON
name: Alice
age: 30
details.personality.summary:
details.appearance.summary:
```
```"""

    result = parser.parse(raw_toon)
    expected = {
        "name": "Alice",
        "age": 30,
    }

    print("RAW:")
    print(raw_toon)
    print("PARSED:")
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    print_evaluation("QUALITY", result.model_dump(), expected)


def run_recovery_case() -> None:
    print_header("CASE 4: parse_with_recovery (TOON repair -> JSON fallback)")
    parser = ToonOutputParser(
        model=CharacterFeatures,
        cfg=ParserConfig(instructions_mode="adaptive", max_repair_attempts=1),
    )

    bad_raw = "name Alice\nage: seventeen"

    repaired = parser.parse_with_recovery(
        bad_raw,
        repair_callback=lambda _prompt: "name: Alice\nage: 30\ndetails.personality.summary:\ndetails.appearance.summary:",
    )
    fallback = parser.parse_with_recovery(
        bad_raw,
        repair_callback=lambda _prompt: "still bad output",
        json_callback=lambda _prompt: '{"name":"Jane","age":22,"details":{"personality":{"traits":[],"summary":""},"appearance":{"features":[],"summary":""}}}',
    )

    print("REPAIRED PARSED:")
    print(json.dumps(repaired.model_dump(), ensure_ascii=False, indent=2))
    print_evaluation("QUALITY REPAIRED", repaired.model_dump(), {"name": "Alice", "age": 30})

    print("JSON FALLBACK PARSED:")
    print(json.dumps(fallback.model_dump(), ensure_ascii=False, indent=2))
    print_evaluation("QUALITY FALLBACK", fallback.model_dump(), {"name": "Jane", "age": 22})


def main() -> None:
    run_character_case()
    run_team_case()
    run_nested_fence_case()
    run_recovery_case()
    print("\nALL REGRESSION CHECKS EXECUTED")


if __name__ == "__main__":
    main()
