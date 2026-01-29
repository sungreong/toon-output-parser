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


class DateTimeData(BaseModel):
    """날짜/시간 형식 데이터."""

    date_iso: str = Field(..., description="ISO 날짜 형식 (YYYY-MM-DD)")
    datetime_iso: str = Field(..., description="ISO 날짜시간 형식 (YYYY-MM-DD HH:MM:SS)")
    date_simple: str = Field(..., description="간단한 날짜 형식 (YYYY/MM/DD)")
    time_only: str = Field(..., description="시간만 (HH:MM:SS)")
    timestamp: int = Field(..., description="Unix 타임스탬프 (초 단위)")
    timezone: str = Field(..., description="타임존 정보 (예: UTC, KST, +09:00)")
    relative_time: str = Field(..., description="상대 시간 (예: '2시간 전', '3일 후')")


def extract_datetime_data(document: str) -> tuple[str, DateTimeData | None, str | None, dict | None]:
    """문서에서 날짜/시간 데이터를 추출합니다.

    Args:
        document: 날짜/시간 정보가 포함된 문서

    Returns:
        tuple[str, DateTimeData | None, str | None, dict | None]: 
        (raw_output, result, parse_error, cost_analysis)
    """
    cfg = ParserConfig(instructions_mode="minimal")
    parser = ToonOutputParser(model=DateTimeData, cfg=cfg)

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
    )

    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 문서에서 날짜/시간 정보를 정확하게 추출하는 전문가입니다. "
                "다양한 날짜/시간 형식을 올바르게 처리해야 합니다.",
            ),
            (
                "human",
                """다음 문서에서 날짜/시간 정보를 추출해주세요.

문서:
{document}

추출해야 할 정보:
1. date_iso: ISO 날짜 형식 (YYYY-MM-DD)
   예: 2024-01-15
   
2. datetime_iso: ISO 날짜시간 형식 (YYYY-MM-DD HH:MM:SS)
   예: 2024-01-15 14:30:00
   
3. date_simple: 간단한 날짜 형식 (YYYY/MM/DD)
   예: 2024/01/15
   
4. time_only: 시간만 (HH:MM:SS)
   예: 14:30:00
   
5. timestamp: Unix 타임스탬프 (초 단위 정수)
   예: 1705312200
   
6. timezone: 타임존 정보
   예: UTC, KST, +09:00, -05:00
   
7. relative_time: 상대 시간 표현
   예: "2시간 전", "3일 후", "1주 전"

⚠️ CRITICAL: 날짜/시간 형식 규칙
- 날짜는 YYYY-MM-DD 형식으로 통일하세요
- 날짜시간은 YYYY-MM-DD HH:MM:SS 형식으로 통일하세요
- 타임스탬프는 정수로 출력하세요 (소수점 없이)
- 타임존은 문자열로 출력하세요
- 상대 시간은 그대로 문자열로 출력하세요

주의사항:
- 모든 필수 필드를 포함하세요
- 날짜 형식을 정확히 지켜주세요
- TOON 형식의 들여쓰기를 정확하게 지켜주세요

{format_instructions}""",
            ),
        ]
    )

    llm_chain = prompt | llm | StrOutputParser()

    raw_output = llm_chain.invoke({"document": document, "format_instructions": format_instructions})

    try:
        result = parser.parse(raw_output)
        # 비용 분석
        cost_analysis = CostAnalyzer.analyze_actual_usage(
            model=DateTimeData,
            toon_raw_output=raw_output,
            parsed_result=result,
            cfg=cfg,
        )
        return raw_output, result, None, cost_analysis
    except Exception as e:
        return raw_output, None, str(e), None


def main() -> None:
    """테스트용 메인 함수 (날짜/시간 형식 예시)."""
    test_document = """
    이벤트 일정:
    
    시작 날짜: 2024년 1월 15일
    시작 시간: 오후 2시 30분
    종료 날짜: 2024/01/20
    종료 시간: 18:00:00
    
    타임스탬프: 1705312200 (2024-01-15 14:30:00 UTC)
    타임존: 한국 표준시 (KST, UTC+9)
    
    상대 시간: 이벤트는 3일 후에 시작됩니다.
    """

    print("=" * 80)
    print("날짜/시간 형식 처리 예시")
    print("=" * 80)
    print(f"\n입력 문서:\n{test_document}\n")

    try:
        raw_output, result, parse_error, cost_analysis = extract_datetime_data(test_document)

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
            print("3. 날짜/시간 형식 확인:")
            print("=" * 80)
            print(f"ISO 날짜: {result.date_iso}")
            print(f"ISO 날짜시간: {result.datetime_iso}")
            print(f"간단한 날짜: {result.date_simple}")
            print(f"시간만: {result.time_only}")
            print(f"타임스탬프: {result.timestamp}")
            print(f"타임존: {result.timezone}")
            print(f"상대 시간: {result.relative_time}")
            print()

            print("=" * 80)
            print("4. 형식 검증:")
            print("=" * 80)
            
            import re
            
            checks = {
                "ISO 날짜 형식 (YYYY-MM-DD)": bool(re.match(r"^\d{4}-\d{2}-\d{2}$", result.date_iso)),
                "ISO 날짜시간 형식": bool(re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", result.datetime_iso)),
                "간단한 날짜 형식 (YYYY/MM/DD)": bool(re.match(r"^\d{4}/\d{2}/\d{2}$", result.date_simple)),
                "시간 형식 (HH:MM:SS)": bool(re.match(r"^\d{2}:\d{2}:\d{2}$", result.time_only)),
                "타임스탬프는 정수": isinstance(result.timestamp, int),
                "타임존은 문자열": isinstance(result.timezone, str),
                "상대 시간은 문자열": isinstance(result.relative_time, str),
            }
            
            for check_name, passed in checks.items():
                status = "✅" if passed else "❌"
                print(f"{status} {check_name}: {passed}")
            
            # 비용 분석 출력
            if cost_analysis:
                print("\n" + "=" * 80)
                print("5. 비용 분석:")
                print("=" * 80)
                CostAnalyzer.print_actual_usage_analysis(cost_analysis)

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
