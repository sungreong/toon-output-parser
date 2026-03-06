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


class PersonalityTraits(BaseModel):
    traits: list[str] = Field(default_factory=list)
    summary: str = ""


class AppearanceDetails(BaseModel):
    features: list[str] = Field(default_factory=list)
    summary: str = ""


class BackgroundInfo(BaseModel):
    origin: str = ""
    history: str = ""


class CharacterDetails(BaseModel):
    personality: PersonalityTraits = Field(default_factory=PersonalityTraits)
    appearance: AppearanceDetails = Field(default_factory=AppearanceDetails)
    background: BackgroundInfo = Field(default_factory=BackgroundInfo)


class CharacterFeatures(BaseModel):
    name: str
    age: int | None = None
    details: CharacterDetails = Field(default_factory=CharacterDetails)


def extract_character_info(document: str) -> tuple[str, CharacterFeatures | None, str | None, dict | None]:
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
                "Extract character data into nested schema. Return only schema-conformant output.",
            ),
            (
                "human",
                "Document:\n{document}\n\n"
                "Required data:\n"
                "- name, age\n"
                "- details.personality(traits, summary)\n"
                "- details.appearance(features, summary)\n"
                "- details.background(origin, history)\n\n"
                "{format_instructions}",
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
        cost_analysis = CostAnalyzer.analyze_actual_usage(
            model=CharacterFeatures,
            toon_raw_output=raw_output,
            parsed_result=result,
            cfg=cfg,
        )
        return raw_output, result, None, cost_analysis
    except Exception as e:
        return raw_output, None, str(e), None


def main() -> None:
    test_document = (
        "해리 제임스 포터는 17세의 마법사다. 그는 용감하고 정의감이 강하다. "
        "검은 머리와 초록색 눈, 번개 모양 흉터를 가졌다. "
        "영국 출신이며 어린 시절 부모를 잃고 머글 가정에서 자랐다."
    )

    print("=" * 80)
    print("Depth 3 nested character extraction")
    print("=" * 80)
    print("\nINPUT DOCUMENT:\n")
    print(test_document)

    raw_output, result, parse_error, cost_analysis = extract_character_info(test_document)

    print("\n" + "=" * 80)
    print("RAW MODEL OUTPUT")
    print("=" * 80)
    print(raw_output)

    if parse_error:
        print("\n" + "=" * 80)
        print("PARSING ERROR")
        print("=" * 80)
        print(parse_error)
        return

    assert result is not None
    print("\n" + "=" * 80)
    print("PARSED JSON")
    print("=" * 80)
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    expected = {
        "name": "해리 제임스 포터",
        "age": 17,
        "details": {
            "background": {"origin": "영국"},
        },
    }
    print_evaluation("QUALITY", result.model_dump(), expected)

    if cost_analysis:
        print("\n" + "=" * 80)
        print("COST ANALYSIS")
        print("=" * 80)
        CostAnalyzer.print_actual_usage_analysis(cost_analysis)


if __name__ == "__main__":
    main()
