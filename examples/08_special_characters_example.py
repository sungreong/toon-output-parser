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


class SpecialTextData(BaseModel):
    """특수 문자가 포함된 텍스트 데이터."""

    description: str = Field(..., description="콜론(:)을 포함할 수 있는 설명")
    note: str = Field(..., description="대시(-)로 시작할 수 있는 노트")
    url: str = Field(..., description="URL 주소 (http:// 또는 https:// 포함)")
    email: str = Field(..., description="이메일 주소 (@ 포함)")
    code_snippet: str = Field(..., description="코드 조각 (특수 문자 포함 가능)")
    json_example: str = Field(..., description="JSON 형식 문자열 (중괄호, 콜론 포함)")


def extract_special_text(document: str) -> SpecialTextData:
    """문서에서 특수 문자가 포함된 텍스트를 추출합니다.

    Args:
        document: 특수 문자가 포함된 텍스트가 있는 문서

    Returns:
        SpecialTextData: 추출된 특수 텍스트 데이터
    """
    parser = ToonOutputParser(model=SpecialTextData)

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
    )

    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 문서에서 특수 문자가 포함된 텍스트를 정확하게 추출하는 전문가입니다. "
                "콜론(:), 대시(-), URL, 이메일 등 특수 문자를 올바르게 처리해야 합니다.",
            ),
            (
                "human",
                """다음 문서에서 특수 문자가 포함된 텍스트를 추출해주세요.

문서:
{document}

추출해야 할 정보:
1. description: 콜론(:)이 포함될 수 있는 설명 텍스트
2. note: 대시(-)로 시작할 수 있는 노트
3. url: URL 주소 (http:// 또는 https:// 포함)
4. email: 이메일 주소 (@ 포함)
5. code_snippet: 코드 조각 (특수 문자 포함)
6. json_example: JSON 형식 문자열

⚠️ CRITICAL: 특수 문자 처리 규칙
- 콜론(:)이 포함된 텍스트는 따옴표로 감싸주세요: "키: 값 형태"
- 대시(-)로 시작하는 텍스트도 따옴표로 감싸주세요: "- 이것은 노트"
- URL과 이메일은 그대로 출력해도 됩니다 (파서가 자동 처리)
- JSON 문자열은 따옴표로 감싸주세요: "{{"key": "value"}}"

주의사항:
- 특수 문자가 TOON 파싱을 방해하지 않도록 따옴표를 적절히 사용하세요
- 모든 필드는 반드시 포함해야 합니다
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
    """테스트용 메인 함수 (특수 문자 처리 예시)."""
    test_document = """
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
    {{"name": "제품", "price": 10000, "tags": ["new", "sale"]}}
    """

    print("=" * 80)
    print("특수 문자 처리 예시")
    print("=" * 80)
    print(f"\n입력 문서:\n{test_document}\n")

    try:
        raw_output, result, parse_error = extract_special_text(test_document)

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
            print("2. 최종 추출 결과:")
            print("=" * 80)
            print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
            print()

            print("=" * 80)
            print("3. 특수 문자 처리 확인:")
            print("=" * 80)
            print(f"Description (콜론 포함): {result.description}")
            print(f"Note (대시 포함): {result.note}")
            print(f"URL: {result.url}")
            print(f"Email: {result.email}")
            print(f"Code snippet: {result.code_snippet[:50]}...")
            print(f"JSON example: {result.json_example[:50]}...")
            print()

            # 특수 문자 검증
            print("=" * 80)
            print("4. 특수 문자 검증:")
            print("=" * 80)
            checks = {
                "콜론 포함": ":" in result.description,
                "대시 포함": "-" in result.note or result.note.startswith("-"),
                "URL 형식": result.url.startswith(("http://", "https://")),
                "이메일 형식": "@" in result.email and "." in result.email,
                "JSON 형식": "{" in result.json_example and "}" in result.json_example,
            }
            for check_name, passed in checks.items():
                status = "✅" if passed else "❌"
                print(f"{status} {check_name}: {passed}")
            
            # ========================================================================
            # 🔥 실제 사용 비용 분석
            # ========================================================================
            print("\n")
            cfg = ParserConfig(instructions_mode="minimal")
            analysis = CostAnalyzer.analyze_actual_usage(
                model=SpecialTextData,
                toon_raw_output=raw_output,
                parsed_result=result,
                cfg=cfg,
            )
            
            CostAnalyzer.print_actual_usage_analysis(analysis)
            print()

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
