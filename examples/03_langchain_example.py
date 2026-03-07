from __future__ import annotations

import json
import os
import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from eval_metrics import print_evaluation

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain_community.chat_models import ChatOpenAI

from toon_langchain_parser import CostAnalyzer, ToonOutputParser
from toon_langchain_parser.toon_parser_ultimate import ParserConfig


class CharacterFeatures(BaseModel):
    name: str = Field(..., description="Character name")
    personality: str = Field(default="", description="Personality summary")
    appearance: str = Field(default="", description="Appearance summary")


def _print_raw_block(raw_output: str) -> None:
    print("=== RAW BEGIN ===")
    print(raw_output)
    print("=== RAW END ===")


def _dump_raw_if_configured(raw_output: str) -> str | None:
    dump_path = os.getenv("TOON_RAW_DUMP_PATH", "").strip()
    if not dump_path:
        return None
    with open(dump_path, "w", encoding="utf-8") as f:
        f.write(raw_output)
    return dump_path


def extract_character_info(document: str) -> tuple[str, ToonOutputParser]:
    cfg = ParserConfig(instructions_mode="adaptive")
    parser = ToonOutputParser(model=CharacterFeatures, cfg=cfg)

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract structured character information. Return only schema-conformant output.",
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
    return raw_output, parser


def main() -> None:
    document = (
        "Aria is calm and analytical, but speaks directly in conflict. "
        "She has short black hair and always carries a silver notebook."
    )

    raw_output, parser = extract_character_info(document)
    _print_raw_block(raw_output)

    try:
        result = parser.parse(raw_output)
    except Exception as e:
        print("\n=== PARSE ERROR ===")
        print(str(e))
        print("\n=== RAW REPLAY ===")
        _print_raw_block(raw_output)
        dumped_path = _dump_raw_if_configured(raw_output)
        if dumped_path:
            print(f"\nRAW output dumped to: {dumped_path}")
        sys.exit(1)

    print("\n=== PARSED ===")
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    expected = {
        "name": "Aria",
        "personality": "calm and analytical",
        "appearance": "short black hair",
    }
    print_evaluation("QUALITY", result.model_dump(), expected)

    analysis = CostAnalyzer.analyze_actual_usage(
        model=CharacterFeatures,
        toon_raw_output=raw_output,
        parsed_result=result,
        cfg=ParserConfig(instructions_mode="adaptive"),
    )
    print("\n=== COST ===")
    CostAnalyzer.print_actual_usage_analysis(analysis)


if __name__ == "__main__":
    main()
