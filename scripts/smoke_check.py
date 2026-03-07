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
    print("[SMOKE] 1) Flat 파싱 성공")
    print(f"[SMOKE]    파싱 결과: {flat_result.model_dump()}")
    print(f"[SMOKE]    적용 모드: {flat_parser.get_effective_mode()}")
    print(f"[SMOKE]    모드 사유: {flat_parser.get_mode_reason()}")

    recursive_parser = ToonOutputParser(
        model=RecursiveNode,
        cfg=ParserConfig(instructions_mode="adaptive"),
    )
    print("[SMOKE] 2) Recursive 모델 모드 판정 성공")
    print(f"[SMOKE]    적용 모드: {recursive_parser.get_effective_mode()}")
    print(f"[SMOKE]    모드 사유: {recursive_parser.get_mode_reason()}")
    print("[SMOKE] 최종 결과: 모든 스모크 체크를 통과했습니다.")


if __name__ == "__main__":
    main()
