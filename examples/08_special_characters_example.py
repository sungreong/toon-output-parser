from __future__ import annotations

import json
import os

from eval_metrics import print_evaluation
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain_community.chat_models import ChatOpenAI

from toon_langchain_parser import ToonOutputParser
from toon_langchain_parser.toon_parser_ultimate import ParserConfig


class SpecialTextData(BaseModel):
    description: str = Field(default="", description="Text containing colon characters.")
    note: str = Field(default="", description="Plain string note. Keep as a single string.")
    url: str = Field(default="", description="URL value.")
    email: str = Field(default="", description="Email value.")
    code_snippet: str = Field(default="", description="Code snippet as text.")
    json_example: str = Field(default="", description="JSON example serialized as string.")


def run_extraction(document: str) -> tuple[str, SpecialTextData | None, str | None]:
    cfg = ParserConfig(instructions_mode="adaptive")
    parser = ToonOutputParser(model=SpecialTextData, cfg=cfg)
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract exactly the schema fields. Keep special characters unchanged.",
            ),
            (
                "human",
                "Document:\n{document}\n\n"
                "Rules:\n"
                "- `note` is a string field, not a list.\n"
                "- Preserve `:`, `-`, URL, email, code, and JSON text.\n"
                "- Return only schema-conformant output.\n\n"
                "{format_instructions}",
            ),
        ]
    )

    raw_output = (prompt | llm | StrOutputParser()).invoke(
        {"document": document, "format_instructions": parser.get_format_instructions()}
    )

    try:
        result = parser.parse(raw_output)
        return raw_output, result, None
    except Exception as e:
        return raw_output, None, str(e)


def main() -> None:
    document = """
제품 설명서:

설명: 이 제품은 키: 값 형태로 데이터를 저장합니다.
노트: - 이것은 중요한 노트입니다
- 이것도 노트입니다

연락처:
웹사이트: https://example.com/products/item-123
이메일: support@example.com

코드 예시:
def process(data: dict) -> None:
    print(f"Key: {data['key']}")

JSON 예시:
{"name": "제품", "price": 10000, "tags": ["new", "sale"]}
""".strip()

    print("=" * 80)
    print("Special character handling")
    print("=" * 80)
    print("\nINPUT DOCUMENT:\n")
    print(document)

    raw_output, result, parse_error = run_extraction(document)

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
    parsed = result.model_dump()
    print("\n" + "=" * 80)
    print("PARSED JSON")
    print("=" * 80)
    print(json.dumps(parsed, ensure_ascii=False, indent=2))

    expected = {
        "description": "이 제품은 키: 값 형태로 데이터를 저장합니다.",
        "url": "https://example.com/products/item-123",
        "email": "support@example.com",
    }
    print_evaluation("QUALITY", parsed, expected)


if __name__ == "__main__":
    main()
