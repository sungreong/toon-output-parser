from __future__ import annotations

from pydantic import BaseModel, Field

from toon_langchain_parser import ParserConfig, ToonOutputParser


class FlatModel(BaseModel):
    name: str
    age: int


class RecursiveNode(BaseModel):
    name: str
    children: list["RecursiveNode"] = Field(default_factory=list)


RecursiveNode.model_rebuild()


def main() -> None:
    flat_parser = ToonOutputParser(model=FlatModel, cfg=ParserConfig(instructions_mode="adaptive"))
    flat_result = flat_parser.parse("name: Alice\nage: 30")
    print("flat_parse_ok", flat_result.model_dump())
    print("flat_mode", flat_parser.get_effective_mode(), flat_parser.get_mode_reason())

    recursive_parser = ToonOutputParser(
        model=RecursiveNode,
        cfg=ParserConfig(instructions_mode="adaptive"),
    )
    print(
        "recursive_mode",
        recursive_parser.get_effective_mode(),
        recursive_parser.get_mode_reason(),
    )


if __name__ == "__main__":
    main()
