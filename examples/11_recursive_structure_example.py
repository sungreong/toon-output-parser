from __future__ import annotations

import json
import os
import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    try:
        from langchain_community.chat_models import ChatOpenAI
    except ImportError:
        raise ImportError(
            "langchain-openai 또는 langchain-community가 필요합니다. 설치: pip install langchain-openai"
        ) from None

from toon_langchain_parser import ToonOutputParser
from toon_langchain_parser.toon_parser_ultimate import ParserConfig

class TreeNode(BaseModel):
    """재귀적 트리 구조 노드."""

    name: str = Field(..., description="노드 이름")
    value: str | None = Field(None, description="노드 값")
    children: list["TreeNode"] = Field(default_factory=list, description="자식 노드 리스트 (재귀적 구조)")


# Forward reference 해결
TreeNode.model_rebuild()


def extract_tree_structure(document: str) -> TreeNode:
    """문서에서 트리 구조를 추출합니다.

    Args:
        document: 트리 구조가 설명된 문서

    Returns:
        TreeNode: 추출된 트리 구조 (루트 노드)
    """
    cfg = ParserConfig(instructions_mode="minimal")
    parser = ToonOutputParser(model=TreeNode, cfg=cfg)

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
    )

    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 문서에서 트리 구조를 정확하게 추출하는 전문가입니다. "
                "재귀적 구조(자기 자신을 참조하는 구조)를 올바르게 처리해야 합니다.",
            ),
            (
                "human",
                """다음 문서에서 트리 구조를 추출해주세요.

문서:
{document}

추출해야 할 정보:
- name: 노드 이름 (필수)
- value: 노드 값 (선택, 없으면 생략 가능)
- children: 자식 노드 리스트 (재귀적 구조)
  - 각 자식 노드도 같은 구조를 가짐 (name, value, children)
  - 자식이 없으면 빈 리스트 []

주의사항:
- 모든 필수 필드(name)는 반드시 포함해야 합니다
- 재귀 구조를 올바르게 처리하여 모든 노드를 포함해야 합니다
- 깊이 제한 없이 모든 중첩된 노드를 추출해야 합니다

{format_instructions}""",
            ),
        ]
    )

    llm_chain = prompt | llm | StrOutputParser()

    raw_output = llm_chain.invoke({"document": document, "format_instructions": format_instructions})

    try:
        result = parser.parse(raw_output)
    except Exception as e:
        return raw_output, None, str(e)

    return raw_output, result, None


def print_tree(node: TreeNode, indent: int = 0) -> None:
    """트리 구조를 시각적으로 출력합니다."""
    prefix = "  " * indent
    print(f"{prefix}├─ {node.name}", end="")
    if node.value:
        print(f" ({node.value})")
    else:
        print()
    for child in node.children:
        print_tree(child, indent + 1)


def main() -> None:
    """테스트용 메인 함수 (재귀 구조 예시)."""
    test_document = """
    조직도 구조:
    
    CEO (이름: John, 값: CEO)
      ├─ CTO (이름: Alice, 값: CTO)
      │   ├─ 개발팀장 (이름: Bob, 값: Team Lead)
      │   │   ├─ 개발자1 (이름: Charlie)
      │   │   └─ 개발자2 (이름: David)
      │   └─ QA팀장 (이름: Eve, 값: QA Lead)
      │       └─ QA1 (이름: Frank)
      └─ CFO (이름: Grace, 값: CFO)
          └─ 회계팀장 (이름: Henry, 값: Accountant)
    """

    print("=" * 80)
    print("재귀 구조 처리 예시")
    print("=" * 80)
    print(f"\n입력 문서:\n{test_document}\n")

    try:
        raw_output, result, parse_error = extract_tree_structure(test_document)

        print("=" * 80)
        print("1. LLM이 생성한 원본 TOON 출력:")
        print("=" * 80)
        print("```toon")
        print(raw_output)
        print("```")
        print()

        if parse_error:
            print("=" * 80)
            print("⚠️ 파싱 에러 발생:")
            print("=" * 80)
            print(parse_error)
            print()
            return

        if result:
            print("=" * 80)
            print("2. 최종 추출 결과 (JSON):")
            print("=" * 80)
            print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
            print()

            print("=" * 80)
            print("3. 트리 구조 시각화:")
            print("=" * 80)
            print_tree(result)
            print()

            print("=" * 80)
            print("4. 구조 통계:")
            print("=" * 80)

            def count_nodes(node: TreeNode) -> tuple[int, int]:
                """노드 수와 최대 깊이 계산"""
                if not node.children:
                    return 1, 1
                max_depth = 0
                total = 1
                for child in node.children:
                    child_count, child_depth = count_nodes(child)
                    total += child_count
                    max_depth = max(max_depth, child_depth)
                return total, max_depth + 1

            total_nodes, max_depth = count_nodes(result)
            print(f"총 노드 수: {total_nodes}")
            print(f"최대 깊이: {max_depth}")

            def count_children(node: TreeNode) -> int:
                """직접 자식 수 계산"""
                return len(node.children)

            print(f"\n루트 노드: {result.name}")
            print(f"루트의 자식 수: {count_children(result)}")
            if result.children:
                print(f"첫 번째 자식: {result.children[0].name}")
                print(f"첫 번째 자식의 자식 수: {count_children(result.children[0])}")

    except Exception as e:
        print("=" * 80)
        print("❌ 치명적 오류 발생!")
        print("=" * 80)
        print(f"오류 타입: {type(e).__name__}")
        print(f"오류 메시지: {e}")
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        raise

    print("\n" + "=" * 80)
    print("테스트 완료")
    print("=" * 80)


if __name__ == "__main__":
    main()
