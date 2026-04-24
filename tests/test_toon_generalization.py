from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from toon_langchain_parser import ModelComplexityAnalyzer, ParserConfig, ToonParser


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


def test_auto_mode_for_flat_schema_is_toon():
    parser = ToonParser(model=FlatModel, cfg=ParserConfig(instructions_mode="adaptive"))
    assert parser.get_effective_mode() == "toon"


def test_auto_mode_for_recursive_schema_is_json():
    parser = ToonParser(model=RecursiveNode, cfg=ParserConfig(instructions_mode="adaptive"))
    assert parser.get_effective_mode() == "json"
    assert "recursive" in parser.get_mode_reason()


def test_scalar_broad_union_stays_toon_but_complex_union_falls_back_to_json():
    class SafeUnion(BaseModel):
        value: str | int

    class ScalarBroadUnion(BaseModel):
        value: str | int | float

    class ComplexUnion(BaseModel):
        value: FlatModel | str

    safe_parser = ToonParser(model=SafeUnion, cfg=ParserConfig(instructions_mode="adaptive"))
    scalar_broad_parser = ToonParser(
        model=ScalarBroadUnion,
        cfg=ParserConfig(instructions_mode="adaptive"),
    )
    complex_parser = ToonParser(model=ComplexUnion, cfg=ParserConfig(instructions_mode="adaptive"))

    assert safe_parser.get_effective_mode() == "toon"
    assert scalar_broad_parser.get_effective_mode() == "toon"
    assert scalar_broad_parser.get_mode_decision()["soft_reasons"] == ["scalar union choices=3"]
    assert complex_parser.get_effective_mode() == "json"
    assert "complex union branches" in complex_parser.get_mode_reason()


def test_analyzer_repeated_analyze_does_not_accumulate_union_counts():
    class ScalarBroadUnion(BaseModel):
        value: str | int | float

    analyzer = ModelComplexityAnalyzer(ScalarBroadUnion)
    first = analyzer.analyze()
    second = analyzer.analyze()

    assert first.union_type_count == 1
    assert second.union_type_count == 1


def test_hard_fallbacks_still_use_json_in_toon_first_policy():
    class DynamicMap(BaseModel):
        values: dict[str, str]

    class Matrix(BaseModel):
        rows: list[list[str]]

    class UnionArray(BaseModel):
        values: list[str | int]

    for model in (RecursiveNode, DynamicMap, Matrix, UnionArray):
        parser = ToonParser(model=model, cfg=ParserConfig(instructions_mode="adaptive"))
        assert parser.get_effective_mode() == "json"
        assert parser.get_mode_decision()["hard_reasons"]


def test_soft_depth_risk_stays_toon_first_but_safe_policy_falls_back_to_json():
    class Level4(BaseModel):
        value: str

    class Level3(BaseModel):
        child: Level4

    class Level2(BaseModel):
        child: Level3

    class Level1(BaseModel):
        child: Level2

    class Root(BaseModel):
        child: Level1

    toon_first = ToonParser(model=Root, cfg=ParserConfig(instructions_mode="adaptive"))
    safe = ToonParser(
        model=Root,
        cfg=ParserConfig(instructions_mode="adaptive", fallback_policy="safe"),
    )

    assert toon_first.get_effective_mode() == "toon"
    assert "depth=4" in toon_first.get_mode_decision()["soft_reasons"]
    assert toon_first.get_mode_decision()["risk_score"] >= 12
    assert safe.get_effective_mode() == "json"


def test_wide_object_list_stays_toon_first():
    class WideItem(BaseModel):
        f1: str
        f2: str
        f3: str
        f4: str
        f5: str
        f6: str
        f7: str
        f8: str

    class WideList(BaseModel):
        items: list[WideItem]

    parser = ToonParser(model=WideList, cfg=ParserConfig(instructions_mode="adaptive"))
    decision = parser.get_mode_decision()

    assert parser.get_effective_mode() == "toon"
    assert "wide list objects=8" in decision["soft_reasons"]


def test_soft_fallback_threshold_can_force_json_from_soft_risk():
    class Level4(BaseModel):
        value: str

    class Level3(BaseModel):
        child: Level4

    class Level2(BaseModel):
        child: Level3

    class Level1(BaseModel):
        child: Level2

    class Root(BaseModel):
        child: Level1

    parser = ToonParser(
        model=Root,
        cfg=ParserConfig(instructions_mode="adaptive", soft_fallback_threshold=1),
    )

    assert parser.get_effective_mode() == "json"
    assert parser.get_mode_decision()["hard_reasons"] == []


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


def test_schema_aware_prompt_includes_non_null_required_scalar_guidance():
    class StrictModel(BaseModel):
        title: str
        score: int
        note: str | None = None

    parser = ToonParser(model=StrictModel, cfg=ParserConfig(instructions_mode="adaptive"))
    instructions = parser.get_format_instructions()
    assert "Required non-null scalar fields:" in instructions
    assert "- title: must not be null" in instructions
    assert "- score: must not be null" in instructions


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

    parser = ToonParser(
        model=Root,
        cfg=ParserConfig(instructions_mode="adaptive", expand_paths="safe"),
    )
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

    parser = ToonParser(
        model=Root,
        cfg=ParserConfig(instructions_mode="adaptive", expand_paths="safe"),
    )
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


def test_list_object_can_be_coerced_from_dash_scalar_items():
    class Trait(BaseModel):
        name: str
        level: str = ""

    class Member(BaseModel):
        traits: list[Trait] = Field(default_factory=list)

    parser = ToonParser(model=Member, cfg=ParserConfig(instructions_mode="adaptive"))
    out = parser.parse("traits:\n  - detail-oriented\n  - methodical")
    assert len(out.traits) == 2
    assert out.traits[0].name == "detail-oriented"
    assert out.traits[1].name == "methodical"


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


def test_official_tabular_header_is_supported():
    class Item(BaseModel):
        name: str
        price: int

    class Inventory(BaseModel):
        items: list[Item] = Field(default_factory=list)

    parser = ToonParser(model=Inventory, cfg=ParserConfig(instructions_mode="adaptive"))
    out = parser.parse(
        "items[2]{name,price}:\n"
        "  iPhone,1200000\n"
        "  Galaxy,1100000\n"
    )
    assert [item.name for item in out.items] == ["iPhone", "Galaxy"]
    assert out.items[1].price == 1100000


def test_pipe_and_tab_delimited_tabular_headers_are_supported():
    class Item(BaseModel):
        sku: str
        name: str
        qty: int

    class Inventory(BaseModel):
        items: list[Item] = Field(default_factory=list)

    pipe_parser = ToonParser(model=Inventory, cfg=ParserConfig(instructions_mode="adaptive"))
    pipe_out = pipe_parser.parse("items[1|]{sku|name|qty}:\n  A1|Widget, large|2\n")
    assert pipe_out.items[0].name == "Widget, large"

    tab_parser = ToonParser(model=Inventory, cfg=ParserConfig(instructions_mode="adaptive"))
    tab_out = tab_parser.parse("items[1\t]{sku\tname\tqty}:\n  B2\tGadget\t3\n")
    assert tab_out.items[0].qty == 3


def test_legacy_tabular_header_can_be_disabled():
    class Item(BaseModel):
        name: str

    class Inventory(BaseModel):
        items: list[Item] = Field(default_factory=list)

    parser = ToonParser(
        model=Inventory,
        cfg=ParserConfig(instructions_mode="adaptive", compat_legacy_headers=False),
    )
    with pytest.raises(ValueError):
        parser.parse("items[1,]{name}:\n  iPhone\n")


def test_strict_count_detects_truncated_scalar_lists_by_default():
    class Tags(BaseModel):
        tags: list[str] = Field(default_factory=list)

    parser = ToonParser(model=Tags, cfg=ParserConfig(instructions_mode="adaptive"))
    with pytest.raises(ValueError):
        parser.parse("tags[3]: a,b\n")


def test_quoted_string_escapes_are_unescaped_and_invalid_escapes_fail():
    class Text(BaseModel):
        note: str

    parser = ToonParser(model=Text, cfg=ParserConfig(instructions_mode="adaptive"))
    out = parser.parse('note: "first\\nsecond\\tline"')
    assert out.note == "first\nsecond\tline"

    with pytest.raises(ValueError):
        parser.parse('note: "bad\\xescape"')


def test_smart_string_safety_accepts_datetime_url_and_time_scalars():
    class Text(BaseModel):
        extraction_date: str
        date_iso: str = ""
        time_only: str = ""
        url: str = ""

    parser = ToonParser(model=Text, cfg=ParserConfig(instructions_mode="adaptive"))
    out = parser.parse(
        "extraction_date: 2026-04-24 12:40:15\n"
        "date_iso: 2026-04-24\n"
        "time_only: 12:40:15\n"
        "url: https://example.com/a:b\n"
    )

    assert out.extraction_date == "2026-04-24 12:40:15"
    assert out.date_iso == "2026-04-24"
    assert out.time_only == "12:40:15"
    assert out.url == "https://example.com/a:b"


def test_general_unquoted_string_with_colon_still_requires_quotes_in_smart_mode():
    class Text(BaseModel):
        note: str

    parser = ToonParser(model=Text, cfg=ParserConfig(instructions_mode="adaptive"))
    with pytest.raises(ValueError):
        parser.parse("note: key: value")


def test_strict_string_safety_rejects_unquoted_url_and_datetime():
    class Text(BaseModel):
        extraction_date: str
        url: str

    parser = ToonParser(
        model=Text,
        cfg=ParserConfig(instructions_mode="adaptive", string_safety="strict"),
    )
    with pytest.raises(ValueError):
        parser.parse("extraction_date: 2026-04-24 12:40:15\nurl: https://example.com/a:b")


def test_unquoted_json_like_string_still_fails():
    class Text(BaseModel):
        json_text: str

    parser = ToonParser(model=Text, cfg=ParserConfig(instructions_mode="adaptive"))
    with pytest.raises(ValueError):
        parser.parse('json_text: {"a":1}')


def test_insurance_example_raw_output_with_unquoted_datetime_parses():
    class InsuranceProduct(BaseModel):
        product_name: str
        insurer: str = ""
        monthly_premium: float = 0.0
        coverage_amount: float = 0.0

    class InsuranceDocumentExtraction(BaseModel):
        document_id: str
        extraction_date: str
        products: list[InsuranceProduct] = Field(default_factory=list)
        total_premium_calculated: float = 0.0
        average_coverage_amount: float = 0.0
        summary: str = ""

    parser = ToonParser(
        model=InsuranceDocumentExtraction,
        cfg=ParserConfig(instructions_mode="adaptive"),
    )
    out = parser.parse(
        "document_id: INS-2026-0001\n"
        "extraction_date: 2026-04-24 12:40:15\n"
        "products[2]{product_name,insurer,monthly_premium,coverage_amount}:\n"
        "  Product A,Alpha Insurance,43000,50000000\n"
        "  Product B,Beta Insurance,61000,70000000\n"
        "total_premium_calculated: 104000\n"
        "average_coverage_amount: 60000000\n"
        'summary: ""\n'
    )

    assert out.extraction_date == "2026-04-24 12:40:15"
    assert len(out.products) == 2
    assert out.products[1].insurer == "Beta Insurance"


def test_datetime_formats_fixture_parses_unquoted_datetime_values():
    class DateTimeData(BaseModel):
        date_iso: str
        datetime_iso: str
        date_simple: str
        time_only: str
        timestamp: int
        timezone: str
        relative_time: str

    parser = ToonParser(model=DateTimeData, cfg=ParserConfig(instructions_mode="adaptive"))
    out = parser.parse(
        "date_iso: 2024-01-15\n"
        "datetime_iso: 2024-01-15 14:30:00\n"
        "date_simple: 2024/01/15\n"
        "time_only: 14:30:00\n"
        "timestamp: 1705312200\n"
        "timezone: +09:00\n"
        'relative_time: "3일 후"\n'
    )

    assert out.datetime_iso == "2024-01-15 14:30:00"
    assert out.timezone == "+09:00"


def test_dotted_key_is_literal_by_default_and_safe_expansion_is_opt_in():
    class Nested(BaseModel):
        value: str

    class Root(BaseModel):
        details: Nested

    default_parser = ToonParser(model=Root, cfg=ParserConfig(instructions_mode="adaptive"))
    with pytest.raises(ValueError):
        default_parser.parse("details.value: ok")

    expanding_parser = ToonParser(
        model=Root,
        cfg=ParserConfig(instructions_mode="adaptive", expand_paths="safe"),
    )
    assert expanding_parser.parse("details.value: ok").details.value == "ok"


def test_safe_path_expansion_rejects_collisions():
    class Inner(BaseModel):
        value: str = ""

    class Root(BaseModel):
        details: Inner | str

    parser = ToonParser(
        model=Root,
        cfg=ParserConfig(instructions_mode="adaptive", expand_paths="safe"),
    )
    with pytest.raises(ValueError):
        parser.decode("details: literal\ndetails.value: ok")


def test_prompt_contains_official_header_template_and_strict_count_guidance():
    parser = ToonParser(model=TeamModel, cfg=ParserConfig(instructions_mode="adaptive"))
    instructions = parser.get_format_instructions()
    assert "field[N]{col1,col2}" in instructions
    assert "[N] must match" in instructions
    assert "members[2]{name,role}:" in instructions


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
