# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Optional, Type

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser
from pydantic import BaseModel, Field

from .toon_parser_ultimate import ParserConfig, ToonParser


DEFAULT_PARSER_CONFIG = ParserConfig(
    indent_step=2,
    complexity_threshold=3,
    protect_string_ids=True,
    strict_schema=True,
    strict_count=False,
    allow_tabular_for_flat_objects=True,
)


class ToonOutputParser(BaseOutputParser[BaseModel]):
    """LangChain용 TOON 출력 파서(ultimate 기반)."""

    pydantic_model: Type[BaseModel] = Field(default=None)
    cfg: ParserConfig = Field(default_factory=lambda: DEFAULT_PARSER_CONFIG)
    _parser: Any = None

    def __init__(self, model: Type[BaseModel], cfg: Optional[ParserConfig] = None, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'pydantic_model', model)
        object.__setattr__(self, 'cfg', cfg or DEFAULT_PARSER_CONFIG)
        object.__setattr__(self, '_parser', ToonParser(model=model, cfg=self.cfg))

    def get_format_instructions(self) -> str:
        return self._parser.get_format_instructions()

    def parse(self, text: str) -> BaseModel:
        try:
            return self._parser.parse(text)
        except Exception as e:
            raise OutputParserException(str(e)) from e

    def decode(self, text: str) -> dict:
        """TOON 텍스트를 딕셔너리로 디코딩 (Pydantic 검증 없이).
        
        Args:
            text: TOON 형식의 텍스트
            
        Returns:
            dict: 디코딩된 딕셔너리 (Pydantic 검증 전)
        """
        return self._parser.decode(text)

    @property
    def _type(self) -> str:
        return "toon"
