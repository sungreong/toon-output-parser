from __future__ import annotations

import json
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from eval_metrics import print_evaluation

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain_community.chat_models import ChatOpenAI

from toon_langchain_parser import ToonOutputParser
from toon_langchain_parser.toon_parser_ultimate import ParserConfig


class TreeNode(BaseModel):
    name: str
    value: str | None = None
    children: list["TreeNode"] = Field(default_factory=list)


TreeNode.model_rebuild()


def extract_tree_structure(document: str) -> tuple[str, TreeNode | None, str | None, str, str]:
    cfg = ParserConfig(instructions_mode="adaptive")
    parser = ToonOutputParser(model=TreeNode, cfg=cfg)

    effective_mode = parser._parser.get_effective_mode()
    mode_reason = parser._parser.get_mode_reason()

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract hierarchy into a recursive tree schema.",
            ),
            (
                "human",
                "Document:\n{document}\n\n{format_instructions}",
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    raw_output = chain.invoke(
        {
            "document": document,
            "format_instructions": parser.get_format_instructions(),
        }
    )

    try:
        result = parser.parse(raw_output)
        return raw_output, result, None, effective_mode, mode_reason
    except Exception as e:
        return raw_output, None, str(e), effective_mode, mode_reason


def print_tree(node: TreeNode, indent: int = 0) -> None:
    prefix = "  " * indent
    suffix = f" ({node.value})" if node.value else ""
    print(f"{prefix}- {node.name}{suffix}")
    for child in node.children:
        print_tree(child, indent + 1)


def main() -> None:
    document = (
        "CEO: John. CTO: Alice under CEO. QA Lead: Eve under CTO. "
        "QA Engineer: Frank under QA Lead. CFO: Grace under CEO."
    )

    raw_output, result, error, mode, reason = extract_tree_structure(document)

    print("=== MODE ===")
    print(f"effective_mode={mode}")
    print(f"reason={reason}")

    print("\n=== RAW ===")
    print(raw_output)

    if error:
        print("\n=== ERROR ===")
        print(error)
        return

    assert result is not None
    print("\n=== TREE ===")
    print_tree(result)
    print("\n=== JSON ===")
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    expected = {
        "name": "John",
    }
    print_evaluation("QUALITY", result.model_dump(), expected)


if __name__ == "__main__":
    main()
