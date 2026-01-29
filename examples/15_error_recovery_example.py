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


class UserProfile(BaseModel):
    """사용자 프로필."""

    name: str = Field(..., description="사용자 이름")
    age: int = Field(..., description="나이")
    email: str = Field(..., description="이메일 주소")
    bio: str = Field(default="", description="자기소개")


def extract_user_profile_with_retry(document: str, max_retries: int = 3) -> tuple[str, UserProfile | None, str | None, dict | None]:
    """에러 복구 전략을 사용하여 사용자 프로필을 추출합니다.

    Args:
        document: 사용자 정보가 포함된 문서
        max_retries: 최대 재시도 횟수

    Returns:
        tuple[str, UserProfile | None, str | None, dict | None]: 
        (raw_output, result, final_error, cost_analysis)
    """
    cfg = ParserConfig(instructions_mode="minimal")
    parser = ToonOutputParser(model=UserProfile, cfg=cfg)

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
    )

    format_instructions = parser.get_format_instructions()

    # 1차 시도: 기본 프롬프트
    prompt_v1 = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 문서에서 사용자 프로필을 정확하게 추출하는 전문가입니다.",
            ),
            (
                "human",
                """다음 문서에서 사용자 프로필을 추출해주세요.

문서:
{document}

추출해야 할 정보:
- name: 사용자 이름
- age: 나이 (정수)
- email: 이메일 주소
- bio: 자기소개 (선택사항)

{format_instructions}""",
            ),
        ]
    )

    llm_chain = prompt_v1 | llm | StrOutputParser()
    raw_output = llm_chain.invoke({"document": document, "format_instructions": format_instructions})

    # 1차 파싱 시도
    try:
        result = parser.parse(raw_output)
        # 비용 분석
        cost_analysis = CostAnalyzer.analyze_actual_usage(
            model=UserProfile,
            toon_raw_output=raw_output,
            parsed_result=result,
            cfg=cfg,
        )
        return raw_output, result, None, cost_analysis
    except Exception as e:
        error_msg = str(e)
        print(f"[1차 시도 실패] 에러: {error_msg}")

    # 2차 시도: 에러 메시지와 함께 재요청
    if max_retries >= 2:
        prompt_v2 = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 문서에서 사용자 프로필을 정확하게 추출하는 전문가입니다. "
                    "이전 시도에서 파싱 에러가 발생했습니다. 에러를 참고하여 올바른 형식으로 출력하세요.",
                ),
                (
                    "human",
                    """다음 문서에서 사용자 프로필을 추출해주세요.

문서:
{document}

⚠️ 이전 시도에서 발생한 에러:
{error_message}

이 에러를 해결하여 올바른 TOON 형식으로 출력하세요.

추출해야 할 정보:
- name: 사용자 이름 (필수)
- age: 나이 (정수, 필수)
- email: 이메일 주소 (필수)
- bio: 자기소개 (선택사항, 없으면 빈 문자열)

{format_instructions}""",
                ),
            ]
        )

        llm_chain_v2 = prompt_v2 | llm | StrOutputParser()
        raw_output_v2 = llm_chain_v2.invoke({
            "document": document,
            "error_message": error_msg,
            "format_instructions": format_instructions,
        })

        try:
            result = parser.parse(raw_output_v2)
            print(f"[2차 시도 성공]")
            # 비용 분석
            cost_analysis = CostAnalyzer.analyze_actual_usage(
                model=UserProfile,
                toon_raw_output=raw_output_v2,
                parsed_result=result,
                cfg=cfg,
            )
            return raw_output_v2, result, None, cost_analysis
        except Exception as e:
            error_msg_v2 = str(e)
            print(f"[2차 시도 실패] 에러: {error_msg_v2}")

    # 3차 시도: Fallback 전략 - 더 명확한 가이드
    if max_retries >= 3:
        prompt_v3 = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 문서에서 사용자 프로필을 정확하게 추출하는 전문가입니다. "
                    "반드시 올바른 TOON 형식으로 출력해야 합니다.",
                ),
                (
                    "human",
                    """다음 문서에서 사용자 프로필을 추출해주세요.

문서:
{document}

⚠️ 이전 시도들에서 발생한 에러들:
1차: {error_message_1}
2차: {error_message_2}

이 에러들을 모두 해결하여 반드시 올바른 TOON 형식으로 출력하세요.

⚠️ CRITICAL: 필수 필드 확인
- name: 반드시 포함
- age: 반드시 포함 (정수)
- email: 반드시 포함
- bio: 없으면 빈 문자열 ""

{format_instructions}""",
                ),
            ]
        )

        llm_chain_v3 = prompt_v3 | llm | StrOutputParser()
        raw_output_v3 = llm_chain_v3.invoke({
            "document": document,
            "error_message_1": error_msg,
            "error_message_2": error_msg_v2 if max_retries >= 2 else "",
            "format_instructions": format_instructions,
        })

        try:
            result = parser.parse(raw_output_v3)
            print(f"[3차 시도 성공]")
            # 비용 분석
            cost_analysis = CostAnalyzer.analyze_actual_usage(
                model=UserProfile,
                toon_raw_output=raw_output_v3,
                parsed_result=result,
                cfg=cfg,
            )
            return raw_output_v3, result, None, cost_analysis
        except Exception as e:
            final_error = str(e)
            print(f"[3차 시도 실패] 최종 에러: {final_error}")
            return raw_output_v3, None, final_error, None

    return raw_output, None, error_msg, None


def main() -> None:
    """테스트용 메인 함수 (에러 복구 예시)."""
    test_document = """
    사용자 정보:
    
    이름: 홍길동
    나이: 25세
    이메일: hong@example.com
    자기소개: 안녕하세요. 개발자입니다.
    """

    print("=" * 80)
    print("에러 복구 전략 예시")
    print("=" * 80)
    print(f"\n입력 문서:\n{test_document}\n")

    print("=" * 80)
    print("에러 복구 프로세스 시작...")
    print("=" * 80)
    print()

    try:
        raw_output, result, final_error, cost_analysis = extract_user_profile_with_retry(test_document, max_retries=3)

        print("\n" + "=" * 80)
        print("최종 결과:")
        print("=" * 80)

        if final_error:
            print(f"❌ 최종 파싱 실패")
            print(f"에러: {final_error}")
            print(f"\n마지막 시도의 원본 출력:")
            print("```toon")
            print(raw_output)
            print("```")
        else:
            print("✅ 파싱 성공!")
            print(f"\n최종 원본 출력:")
            print("```toon")
            print(raw_output)
            print("```")
            print(f"\n추출된 프로필:")
            print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
            print(f"\n프로필 정보:")
            print(f"  이름: {result.name}")
            print(f"  나이: {result.age}세")
            print(f"  이메일: {result.email}")
            print(f"  자기소개: {result.bio}")
            
            # 비용 분석 출력
            if cost_analysis:
                print("\n" + "=" * 80)
                print("비용 분석:")
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
    print("\n에러 복구 전략 요약:")
    print("1. 1차 시도: 기본 프롬프트로 시도")
    print("2. 2차 시도: 에러 메시지와 함께 재요청")
    print("3. 3차 시도: 더 명확한 가이드와 예시 제공")
    print("4. 실패 시: 원본 출력 반환하여 수동 처리 가능")


if __name__ == "__main__":
    main()
