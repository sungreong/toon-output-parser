from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from toon_langchain_parser import ParserConfig, ToonParser


class FlatModel(BaseModel):
    name: str
    age: int


class OptionalModel(BaseModel):
    name: str
    nickname: str = ""
    score: int | None = None
    tags: list[str] = Field(default_factory=list)


class Member(BaseModel):
    name: str
    role: str


class TeamModel(BaseModel):
    team_name: str
    members: list[Member] = Field(default_factory=list)


class RecursiveNode(BaseModel):
    name: str
    children: list[RecursiveNode] = Field(default_factory=list)


RecursiveNode.model_rebuild()


def test_auto_mode_for_flat_schema_is_adaptive():
    parser = ToonParser(model=FlatModel, cfg=ParserConfig(instructions_mode="adaptive"))
    assert parser.get_effective_mode() == "adaptive"


def test_auto_mode_for_recursive_schema_is_json():
    parser = ToonParser(model=RecursiveNode, cfg=ParserConfig(instructions_mode="adaptive"))
    assert parser.get_effective_mode() == "json"
    assert "recursive" in parser.get_mode_reason()


def test_schema_aware_prompt_contains_examples():
    parser = ToonParser(model=TeamModel, cfg=ParserConfig(instructions_mode="adaptive"))
    instructions = parser.get_format_instructions()

    assert "Rules:" in instructions
    assert "Flat example:" in instructions
    assert "List-of-object example:" in instructions


def test_schema_aware_prompt_includes_typed_empty_guidance():
    parser = ToonParser(model=OptionalModel, cfg=ParserConfig(instructions_mode="adaptive"))
    instructions = parser.get_format_instructions()

    assert "required and optional fields must be emitted" in instructions
    assert 'string -> ""' in instructions
    assert "- array -> []" in instructions
    assert "Typed empty examples for optional fields:" in instructions


def test_missing_colon_fails_strictly():
    parser = ToonParser(model=FlatModel, cfg=ParserConfig(instructions_mode="adaptive"))
    with pytest.raises(ValueError):
        parser.parse("name John\nage: 10")


def test_scalar_field_with_block_fails_strictly():
    parser = ToonParser(model=FlatModel, cfg=ParserConfig(instructions_mode="adaptive"))
    with pytest.raises(ValueError):
        parser.parse("name:\n  first: John\nage: 10")


def test_dotted_paths_are_supported_in_core_parser():
    parser = ToonParser(model=TeamModel, cfg=ParserConfig(instructions_mode="adaptive"))
    out = parser.parse(
        "team_name: Atlas\n"
        "members:\n"
        "  - name: Minji\n"
        "    role: Backend\n"
        "  - name: Sora\n"
        "    role: Design\n"
    )
    assert out.team_name == "Atlas"
    assert out.members[0].name == "Minji"


def test_dotted_paths_for_nested_objects():
    class Nested(BaseModel):
        traits: str
        summary: str

    class Root(BaseModel):
        details: Nested

    parser = ToonParser(model=Root, cfg=ParserConfig(instructions_mode="adaptive"))
    out = parser.parse(
        "details.traits: brave\n"
        "details.summary: concise\n"
    )
    assert out.details.traits == "brave"
    assert out.details.summary == "concise"


def test_dotted_paths_can_be_disabled():
    class Nested(BaseModel):
        traits: str

    class Root(BaseModel):
        details: Nested

    parser = ToonParser(
        model=Root,
        cfg=ParserConfig(instructions_mode="adaptive", allow_dotted_paths=False),
    )
    with pytest.raises(ValueError):
        parser.parse("details.traits: brave")


def test_inline_csv_is_coerced_for_list_fields():
    class Personality(BaseModel):
        traits: list[str] = Field(default_factory=list)

    class Root(BaseModel):
        details: Personality

    parser = ToonParser(model=Root, cfg=ParserConfig(instructions_mode="adaptive"))
    out = parser.parse("details.traits: brave, loyal, calm")
    assert out.details.traits == ["brave", "loyal", "calm"]


def test_empty_scalar_block_maps_to_empty_string_for_string_fields():
    class Inner(BaseModel):
        summary: str = ""

    class Root(BaseModel):
        details: Inner

    parser = ToonParser(model=Root, cfg=ParserConfig(instructions_mode="adaptive"))
    out = parser.parse("details:\n  summary:")
    assert out.details.summary == ""


def test_string_field_can_coerce_dash_list_block():
    class Root(BaseModel):
        note: str = ""

    parser = ToonParser(model=Root, cfg=ParserConfig(instructions_mode="adaptive"))
    out = parser.parse("note:\n  - first line\n  - second line")
    assert out.note == "- first line\n- second line"


def test_list_object_can_be_coerced_from_inline_scalar():
    class Trait(BaseModel):
        name: str
        level: str = ""

    class Member(BaseModel):
        traits: list[Trait] = Field(default_factory=list)

    parser = ToonParser(model=Member, cfg=ParserConfig(instructions_mode="adaptive"))
    out = parser.parse("traits: detail-oriented")
    assert len(out.traits) == 1
    assert out.traits[0].name == "detail-oriented"


def test_parse_with_recovery_via_repair_callback():
    parser = ToonParser(model=FlatModel, cfg=ParserConfig(instructions_mode="adaptive"))

    recovered = parser.parse_with_recovery(
        "name John\nage: ten",
        repair_callback=lambda _prompt: "name: John\nage: 10",
    )

    assert recovered.name == "John"
    assert recovered.age == 10


def test_parse_with_recovery_json_fallback():
    parser = ToonParser(model=FlatModel, cfg=ParserConfig(instructions_mode="adaptive"))

    recovered = parser.parse_with_recovery(
        "name John\nage ten",
        json_callback=lambda _prompt: '{"name": "Jane", "age": 22}',
    )

    assert recovered.name == "Jane"
    assert recovered.age == 22


def test_tabular_row_column_mismatch_fails_in_strict_mode():
    class Item(BaseModel):
        name: str
        price: int

    class Inventory(BaseModel):
        items: list[Item] = Field(default_factory=list)

    parser = ToonParser(model=Inventory, cfg=ParserConfig(instructions_mode="adaptive"))

    bad_toon = (
        "items[1,]{name,price}:\n"
        "  only_name_value\n"
    )
    with pytest.raises(ValueError):
        parser.parse(bad_toon)


def test_nested_code_fence_with_language_marker_is_parsed():
    parser = ToonParser(model=FlatModel, cfg=ParserConfig(instructions_mode="adaptive"))
    text = (
        "```toon\n"
        "```TOON\n"
        "name: Alice\n"
        "age: 30\n"
        "```\n"
        "```"
    )
    out = parser.parse(text)
    assert out.name == "Alice"
    assert out.age == 30
