from __future__ import annotations

import json
import os
from datetime import datetime

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


class InsuranceProduct(BaseModel):
    product_name: str
    insurer: str = ""
    monthly_premium: float = 0.0
    coverage_amount: float = 0.0


class InsuranceDocumentExtraction(BaseModel):
    document_id: str
    extraction_date: str
    products: list[InsuranceProduct] = Field(default_factory=list)
    total_premium_calculated: float = 0.0
    average_coverage_amount: float = 0.0
    summary: str = ""


def extract_insurance_document(
    document_id: str,
    chunk_texts: list[str],
) -> tuple[str, InsuranceDocumentExtraction | None, str | None]:
    cfg = ParserConfig(instructions_mode="adaptive")
    parser = ToonOutputParser(model=InsuranceDocumentExtraction, cfg=cfg)

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
    )

    joined_chunks = "\n\n".join(f"[Chunk {idx + 1}]\n{text}" for idx, text in enumerate(chunk_texts))

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Aggregate insurance chunks into one structured output. Use schema exactly.",
            ),
            (
                "human",
                "Document ID: {document_id}\nExtraction Date: {extraction_date}\n\nChunks:\n{chunks}\n\n{format_instructions}",
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    raw_output = chain.invoke(
        {
            "document_id": document_id,
            "extraction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "chunks": joined_chunks,
            "format_instructions": parser.get_format_instructions(),
        }
    )

    try:
        result = parser.parse(raw_output)
    except Exception as e:
        return raw_output, None, str(e)

    return raw_output, result, None


def main() -> None:
    chunk_texts = [
        "Product A by Alpha Insurance. Monthly premium 43000 KRW. Coverage amount 50000000 KRW.",
        "Product B by Beta Insurance. Monthly premium 61000 KRW. Coverage amount 70000000 KRW.",
    ]

    raw_output, result, error = extract_insurance_document(
        document_id="INS-2026-0001",
        chunk_texts=chunk_texts,
    )

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
        "document_id": "INS-2026-0001",
    }
    print_evaluation("QUALITY", result.model_dump(), expected)

    analysis = CostAnalyzer.analyze_actual_usage(
        model=InsuranceDocumentExtraction,
        toon_raw_output=raw_output,
        parsed_result=result,
        cfg=ParserConfig(instructions_mode="adaptive"),
    )
    print("\n=== COST ===")
    CostAnalyzer.print_actual_usage_analysis(analysis)


if __name__ == "__main__":
    main()
