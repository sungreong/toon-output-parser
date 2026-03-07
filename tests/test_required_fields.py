from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from toon_langchain_parser import ToonOutputParser


class RequiredFieldsModel(BaseModel):
    required_field: str = Field(...)
    required_number: int = Field(...)
    optional_field: str = Field(default="")
    optional_number: int | None = Field(None)
    optional_list: list[str] = Field(default_factory=list)


def test_required_and_optional_fields_all_present():
    parser = ToonOutputParser(model=RequiredFieldsModel)
    out = parser.parse(
        "required_field: test\n"
        "required_number: 42\n"
        "optional_field: optional\n"
        "optional_number: 10\n"
        "optional_list[2]: a,b\n"
    )
    assert out.required_field == "test"
    assert out.required_number == 42
    assert out.optional_field == "optional"
    assert out.optional_number == 10
    assert out.optional_list == ["a", "b"]


def test_required_fields_only_uses_defaults_for_optional_fields():
    parser = ToonOutputParser(model=RequiredFieldsModel)
    out = parser.parse("required_field: test\nrequired_number: 42\n")
    assert out.required_field == "test"
    assert out.required_number == 42
    assert out.optional_field == ""
    assert out.optional_number is None
    assert out.optional_list == []


def test_missing_required_fields_raise_error():
    parser = ToonOutputParser(model=RequiredFieldsModel)
    with pytest.raises(ValueError):
        parser.parse("optional_field: test\noptional_number: 10\n")
