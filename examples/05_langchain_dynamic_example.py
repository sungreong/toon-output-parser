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


class PersonTrait(BaseModel):
    name: str
    level: str = ""
    description: str = ""


class Person(BaseModel):
    name: str
    age: int | None = None
    role: str = ""
    traits: list[PersonTrait] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)


class TeamAnalysis(BaseModel):
    team_name: str
    total_members: int
    members: list[Person] = Field(default_factory=list)
    team_strengths: list[str] = Field(default_factory=list)
    team_weaknesses: list[str] = Field(default_factory=list)
    overall_assessment: str = ""


def extract_team_analysis(document: str) -> tuple[str, TeamAnalysis | None, str | None]:
    cfg = ParserConfig(instructions_mode="adaptive")
    parser = ToonOutputParser(model=TeamAnalysis, cfg=cfg)

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract team structure and member traits. Keep output strictly in target format.",
            ),
            (
                "human",
                "Document:\n{document}\n\nRules:\n- include all members found\n- include traits and skills per member\n\n{format_instructions}",
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
        return raw_output, parser.parse(raw_output), None
    except Exception as e:
        return raw_output, None, str(e)


def main() -> None:
    document = (
        "Team Atlas has 3 members. Minji leads the backend and is detail-oriented. "
        "Sora leads design and is fast at prototyping. Junho handles QA and is methodical."
    )

    raw_output, result, error = extract_team_analysis(document)

    print("=== RAW ===")
    print(raw_output)

    if error:
        print("\n=== ERROR ===")
        print(error)
        return

    assert result is not None
    print("\n=== PARSED ===")
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    expected = {
        "team_name": "Team Atlas",
        "total_members": 3,
    }
    print_evaluation("QUALITY", result.model_dump(), expected)

    analysis = CostAnalyzer.analyze_actual_usage(
        model=TeamAnalysis,
        toon_raw_output=raw_output,
        parsed_result=result,
        cfg=ParserConfig(instructions_mode="adaptive"),
    )
    print("\n=== COST ===")
    CostAnalyzer.print_actual_usage_analysis(analysis)


if __name__ == "__main__":
    main()
