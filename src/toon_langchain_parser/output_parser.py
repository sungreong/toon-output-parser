# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Generic, Optional, Type, TypeVar

from pydantic import BaseModel, Field

from .toon_parser_ultimate import ParserConfig, ToonParser

T = TypeVar("T")

try:
    from langchain_core.exceptions import OutputParserException
    from langchain_core.output_parsers import BaseOutputParser
except Exception:  # pragma: no cover - optional dependency path
    class OutputParserException(ValueError):
        """Fallback exception when langchain-core is not installed."""

    class BaseOutputParser(Generic[T]):
        """Minimal fallback compatible with direct parser usage."""

        def __init__(self, *args, **kwargs) -> None:
            pass


DEFAULT_PARSER_CONFIG = ParserConfig(
    indent_step=2,
    complexity_threshold=3,
    protect_string_ids=True,
    strict_schema=True,
    strict_count=False,
    allow_tabular_for_flat_objects=True,
)


class ToonOutputParser(BaseOutputParser[BaseModel]):
    """TOON output parser usable with or without LangChain installed."""

    pydantic_model: Type[BaseModel] = Field(default=None)
    cfg: ParserConfig = Field(default_factory=lambda: DEFAULT_PARSER_CONFIG)
    _parser: Any = None

    def __init__(self, model: Type[BaseModel], cfg: Optional[ParserConfig] = None, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "pydantic_model", model)
        object.__setattr__(self, "cfg", cfg or DEFAULT_PARSER_CONFIG)
        object.__setattr__(self, "_parser", ToonParser(model=model, cfg=self.cfg))

    def get_format_instructions(self) -> str:
        return self._parser.get_format_instructions()

    def get_toon_format_instructions(self) -> str:
        return self._parser.get_toon_format_instructions()

    def get_effective_mode(self) -> str:
        return self._parser.get_effective_mode()

    def get_mode_reason(self) -> str:
        return self._parser.get_mode_reason()

    def build_repair_prompt(self, raw_output: str, error: Exception | str) -> str:
        return self._parser.build_repair_prompt(raw_output, error)

    def build_json_retry_prompt(self, error: Exception | str) -> str:
        return self._parser.build_json_retry_prompt(error)

    def parse(self, text: str) -> BaseModel:
        try:
            return self._parser.parse(text)
        except Exception as e:
            raise OutputParserException(str(e)) from e

    def parse_with_recovery(
        self,
        text: str,
        repair_callback=None,
        json_callback=None,
    ) -> BaseModel:
        try:
            return self._parser.parse_with_recovery(
                text,
                repair_callback=repair_callback,
                json_callback=json_callback,
            )
        except Exception as e:
            raise OutputParserException(str(e)) from e

    def decode(self, text: str) -> dict:
        return self._parser.decode(text)

    @property
    def _type(self) -> str:
        return "toon"
