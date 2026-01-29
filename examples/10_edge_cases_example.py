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


class EdgeCases(BaseModel):
    """엣지 케이스 테스트 모델."""

    empty_string: str = Field(default="", description="빈 문자열 (기본값)")
    very_large_number: int = Field(..., description="매우 큰 숫자 (예: 9999999999999)")
    very_small_float: float = Field(..., description="매우 작은 실수 (예: 0.00000001)")
    negative_number: int = Field(..., description="음수 (예: -1000)")
    zero: int = Field(..., description="0")
    null_value: str | None = Field(None, description="null 값 (없으면 null)")
    empty_list: list[str] = Field(default_factory=list, description="빈 리스트")
    empty_dict: dict[str, str] = Field(default_factory=dict, description="빈 딕셔너리")


def extract_edge_cases(document: str) -> EdgeCases:
    """문서에서 엣지 케이스 데이터를 추출합니다.

    Args:
        document: 엣지 케이스가 포함된 문서

    Returns:
        EdgeCases: 추출된 엣지 케이스 데이터
    """
    parser = ToonOutputParser(model=EdgeCases)

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
    )

    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 문서에서 엣지 케이스 데이터를 정확하게 추출하는 전문가입니다. "
                "빈 값, 매우 큰/작은 숫자, 음수, null 등을 올바르게 처리해야 합니다.",
            ),
            (
                "human",
                """다음 문서에서 엣지 케이스 데이터를 추출해주세요.

문서:
{document}

추출해야 할 정보:
1. empty_string: 빈 문자열 (값이 없으면 "" 또는 생략 가능)
2. very_large_number: 매우 큰 숫자 (천 단위 구분자 없이)
3. very_small_float: 매우 작은 실수
4. negative_number: 음수
5. zero: 0
6. null_value: null 값 (없으면 null)
7. empty_list: 빈 리스트 (없으면 [])
8. empty_dict: 빈 딕셔너리 (없으면 {{}})

⚠️ CRITICAL: 엣지 케이스 처리 규칙
- 숫자는 천 단위 구분자(쉼표)를 사용하지 마세요: 9999999999999 (NOT 9,999,999,999,999)
- 음수는 - 기호를 사용: -1000
- 0은 그대로 0으로 출력
- null은 null로 출력
- 빈 값은 "" 또는 [] 또는 {{}}로 출력

주의사항:
- 모든 필수 필드는 반드시 포함해야 합니다
- 숫자 포맷팅 규칙을 정확히 지켜주세요
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
    """테스트용 메인 함수 (엣지 케이스 예시)."""
    test_document = """
    수치 데이터:
    
    빈 문자열: (값 없음)
    매우 큰 숫자: 9,999,999,999,999 (9조 9999억...)
    매우 작은 실수: 0.00000001
    음수: -1000
    영: 0
    null 값: 없음
    빈 리스트: 없음
    빈 딕셔너리: 없음
    """

    print("=" * 80)
    print("엣지 케이스 처리 예시")
    print("=" * 80)
    print(f"\n입력 문서:\n{test_document}\n")

    try:
        raw_output, result, parse_error = extract_edge_cases(test_document)

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
            print("3. 엣지 케이스 검증:")
            print("=" * 80)
            checks = {
                "빈 문자열": result.empty_string == "",
                "매우 큰 숫자": result.very_large_number > 1000000000,
                "매우 작은 실수": 0 < result.very_small_float < 0.001,
                "음수": result.negative_number < 0,
                "영": result.zero == 0,
                "null 값": result.null_value is None,
                "빈 리스트": result.empty_list == [],
                "빈 딕셔너리": result.empty_dict == {},
            }
            for check_name, passed in checks.items():
                status = "✅" if passed else "❌"
                print(f"{status} {check_name}: {passed}")
            print()

            print("=" * 80)
            print("4. 값 확인:")
            print("=" * 80)
            print(f"empty_string: '{result.empty_string}' (길이: {len(result.empty_string)})")
            print(f"very_large_number: {result.very_large_number:,}")
            print(f"very_small_float: {result.very_small_float}")
            print(f"negative_number: {result.negative_number}")
            print(f"zero: {result.zero}")
            print(f"null_value: {result.null_value}")
            print(f"empty_list: {result.empty_list} (길이: {len(result.empty_list)})")
            print(f"empty_dict: {result.empty_dict} (키 수: {len(result.empty_dict)})")

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
