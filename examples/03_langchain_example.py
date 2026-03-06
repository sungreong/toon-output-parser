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

from toon_langchain_parser import CostAnalyzer, ToonOutputParser
from toon_langchain_parser.toon_parser_ultimate import ParserConfig


class CharacterFeatures(BaseModel):
    name: str = Field(..., description="Character name")
    personality: str = Field(default="", description="Personality summary")
    appearance: str = Field(default="", description="Appearance summary")


def extract_character_info(document: str) -> tuple[str, CharacterFeatures]:
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
    return raw_output, parser.parse(raw_output)


def main() -> None:
    document = (
        "Aria is calm and analytical, but speaks directly in conflict. "
        "She has short black hair and always carries a silver notebook."
    )

    raw_output, result = extract_character_info(document)

    print("=== RAW ===")
    print(raw_output)
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
