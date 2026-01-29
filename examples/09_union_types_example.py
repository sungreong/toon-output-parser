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

from toon_langchain_parser import ToonOutputParser, CostAnalyzer
from toon_langchain_parser.toon_parser_ultimate import ParserConfig


class FlexibleData(BaseModel):
    """Union 타입을 사용하는 유연한 데이터 모델."""

    value: str | int | float = Field(..., description="문자열, 정수, 실수 중 하나 (어떤 타입이든 가능)")
    metadata: dict | list | None = Field(None, description="딕셔너리 또는 리스트 또는 None")
    count: int | str = Field(..., description="숫자 또는 '무제한' 같은 문자열")
    status: str | int | bool = Field(..., description="상태 (문자열, 숫자, 불린 중 하나)")


def extract_flexible_data(document: str) -> FlexibleData:
    """문서에서 유연한 타입의 데이터를 추출합니다.

    Args:
        document: 다양한 타입의 데이터가 포함된 문서

    Returns:
        FlexibleData: 추출된 유연한 데이터
    """
    parser = ToonOutputParser(model=FlexibleData)

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
    )

    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 문서에서 유연한 타입의 데이터를 추출하는 전문가입니다. "
                "Union 타입 필드는 여러 타입 중 적절한 하나를 선택하여 출력해야 합니다.",
            ),
            (
                "human",
                """다음 문서에서 데이터를 추출해주세요. 각 필드는 여러 타입 중 하나를 가질 수 있습니다.

문서:
{document}

추출해야 할 정보:
1. value: 문자열, 정수, 실수 중 가장 적절한 타입으로 출력
   - 숫자면 숫자로 (예: 100, 3.14)
   - 텍스트면 문자열로 (예: "text")
   
2. metadata: 딕셔너리, 리스트, 또는 None
   - 객체면 딕셔너리로
   - 배열이면 리스트로
   - 없으면 null
   
3. count: 숫자 또는 문자열
   - 숫자면 정수로 (예: 42)
   - "무제한", "N/A" 같은 텍스트면 문자열로
   
4. status: 문자열, 숫자, 불린 중 하나
   - "active", "pending" 같은 텍스트면 문자열
   - 1, 2, 3 같은 코드면 숫자
   - true/false면 불린

⚠️ CRITICAL: 타입 선택 규칙
- 문서의 내용을 보고 가장 적절한 타입을 선택하세요
- 숫자로 표현 가능하면 숫자로, 텍스트가 필요하면 문자열로
- 불린 값은 true/false로 출력
- null은 null로 출력

주의사항:
- 모든 필수 필드는 반드시 포함해야 합니다
- 타입은 문서 내용에 맞게 자유롭게 선택하세요
- TOON 형식의 들여쓰기를 정확하게 지켜주세요

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


def main() -> None:
    """테스트용 메인 함수 (Union 타입 예시)."""
    test_cases = [
        {
            "name": "숫자 중심 데이터",
            "document": """
            제품 정보:
            가격: 10000원
            재고: 50개
            상태: 활성화됨 (코드: 1)
            메타데이터: {{"category": "electronics", "brand": "Samsung"}}
            """,
        },
        {
            "name": "텍스트 중심 데이터",
            "document": """
            사용자 정보:
            이름: John Doe
            나이: 30세
            상태: active
            메타데이터: ["tag1", "tag2", "tag3"]
            """,
        },
        {
            "name": "혼합 타입 데이터",
            "document": """
            설정 정보:
            값: 3.14159
            개수: 무제한
            상태: true
            메타데이터: 없음
            """,
        },
    ]

    print("=" * 80)
    print("Union 타입 처리 예시")
    print("=" * 80)

    for idx, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"테스트 케이스 {idx}: {test_case['name']}")
        print("=" * 80)
        print(f"\n입력 문서:\n{test_case['document']}\n")

        try:
            raw_output, result, parse_error = extract_flexible_data(test_case["document"])

            print("=" * 80)
            print(f"케이스 {idx} - LLM 원본 출력:")
            print("=" * 80)
            print("```toon")
            print(raw_output)
            print("```")
            print()

            if parse_error:
                print(f"⚠️ 파싱 에러: {parse_error}\n")
                continue

            if result:
                print("=" * 80)
                print(f"케이스 {idx} - 추출 결과:")
                print("=" * 80)
                print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
                print()

                print("=" * 80)
                print(f"케이스 {idx} - 타입 확인:")
                print("=" * 80)
                print(f"value: {result.value} (타입: {type(result.value).__name__})")
                print(f"metadata: {result.metadata} (타입: {type(result.metadata).__name__})")
                print(f"count: {result.count} (타입: {type(result.count).__name__})")
                print(f"status: {result.status} (타입: {type(result.status).__name__})")
                print()
                
                # ========================================================================
                # 🔥 실제 사용 비용 분석
                # ========================================================================
                print("\n")
                cfg = ParserConfig(instructions_mode="minimal")
                analysis = CostAnalyzer.analyze_actual_usage(
                    model=FlexibleData,
                    toon_raw_output=raw_output,
                    parsed_result=result,
                    cfg=cfg,
                )
                
                CostAnalyzer.print_actual_usage_analysis(analysis)
                print()

        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            import traceback

            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("모든 테스트 완료")
    print("=" * 80)


if __name__ == "__main__":
    main()
