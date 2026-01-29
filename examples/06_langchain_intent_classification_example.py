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


class IntentClassification(BaseModel):
    """의도 분류 결과."""

    reason: str = Field(..., description="라우팅 결정 이유 (왜 이렇게 분류했는지)")
    goal_type: str = Field(
        ..., description="목표 타입: 'general_chat' (일반챗) 또는 'marketing_message' (마케팅 메시지 생성)"
    )
    query: str = Field(..., description="검색하기 용이하게 rewrite된 쿼리")
    is_new_question: bool = Field(..., description="새로운 질문인지 (true) 이어서 하는 질문인지 (false)")


def classify_intent(user_query: str, conversation_history: list[str] | None = None) -> IntentClassification:
    """사용자 쿼리의 의도를 분류합니다.

    Args:
        user_query: 사용자의 질문 또는 요청
        conversation_history: 이전 대화 기록 (선택사항)

    Returns:
        IntentClassification: 의도 분류 결과
    """
    # ToonOutputParser 초기화
    cfg = ParserConfig(instructions_mode="minimal")
    parser = ToonOutputParser(model=IntentClassification, cfg=cfg)

    # ChatOpenAI 모델 초기화
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
    )

    # 프롬프트 템플릿 생성
    format_instructions = parser.get_format_instructions()
    
    # 디버깅: minimal 모드가 제대로 반영되었는지 확인
    if os.getenv("DEBUG_FORMAT_INSTRUCTIONS", "false").lower() == "true":
        print(f"[DEBUG] instructions_mode: {cfg.instructions_mode}")
        print(f"[DEBUG] format_instructions 길이: {len(format_instructions)} chars")
        print(f"[DEBUG] format_instructions 내용:\n{format_instructions}\n")

    # 대화 기록 포맷팅
    history_text = ""
    if conversation_history:
        history_text = "\n이전 대화 기록:\n" + "\n".join(f"- {msg}" for msg in conversation_history[-3:])  # 최근 3개만

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 사용자의 의도를 정확하게 분류하는 전문가입니다. 질문의 목적과 맥락을 분석하여 적절한 라우팅을 결정해야 합니다.",
            ),
            (
                "human",
                """다음 사용자 쿼리의 의도를 분류해주세요.

사용자 쿼리:
{user_query}
{history_text}

분류해야 할 항목:
1. reason: 라우팅 결정 이유 (왜 이렇게 분류했는지 명확하게 설명)
2. goal_type: 목표 타입
   - "general_chat": 일반적인 대화, 질문, 정보 요청
   - "marketing_message": 마케팅 메시지, 홍보 문구, 광고 문구 생성 요청
3. query: 검색하기 용이하게 rewrite된 쿼리 (검색 엔진에 최적화된 형태)
4. is_new_question: 새로운 질문인지 (true) 이전 대화를 이어서 하는 질문인지 (false)

주의사항:
- reason은 구체적이고 명확하게 작성하세요.
- goal_type은 반드시 "general_chat" 또는 "marketing_message" 중 하나여야 합니다.
- query는 검색 엔진에서 찾기 쉬운 키워드 중심으로 rewrite하세요.
- is_new_question은 이전 대화 맥락과의 연관성을 고려하여 판단하세요.
- TOON 형식의 들여쓰기를 정확하게 지켜주세요.

{format_instructions}""",
            ),
        ]
    )

    # 체인 구성: 프롬프트 -> LLM -> 문자열 출력
    llm_chain = prompt | llm | StrOutputParser()

    # 프롬프트 변수 준비
    prompt_vars = {
        "user_query": user_query,
        "history_text": history_text,
        "format_instructions": format_instructions,
    }

    # LLM 출력 확인 (디버깅용)
    raw_output = llm_chain.invoke(prompt_vars)

    # TOON 파싱 (에러가 발생해도 raw_output은 반환)
    try:
        result = parser.parse(raw_output)
    except Exception as e:
        # 파싱 에러가 발생해도 raw_output은 반환
        return raw_output, None, str(e)

    return raw_output, result, None


def main() -> None:
    """테스트용 메인 함수 (의도 분류 예시)."""
    # 테스트 케이스들
    test_cases = [
        {
            "query": "Nori 사용자 사전은 어떻게 설정하나요?",
            "history": None,
            "description": "일반 질문 (새 질문)",
        },
        {
            "query": "그럼 이전에 설정한 사전을 삭제하려면?",
            "history": ["Nori 사용자 사전은 어떻게 설정하나요?", "사전 설정은 설정 메뉴에서 할 수 있습니다."],
            "description": "이어서 하는 질문",
        },
        {
            "query": "신제품 출시를 알리는 이메일 마케팅 문구를 작성해주세요",
            "history": None,
            "description": "마케팅 메시지 생성 요청",
        },
        {
            "query": "SNS에 올릴 제품 홍보 문구 만들어줘",
            "history": None,
            "description": "마케팅 메시지 생성 요청 (SNS)",
        },
        {
            "query": "파이썬에서 리스트를 정렬하는 방법이 뭐야?",
            "history": None,
            "description": "일반 질문 (프로그래밍)",
        },
    ]

    print("=" * 80)
    print("의도 분류 예시")
    print("=" * 80)

    for idx, test_case in enumerate(test_cases, 1):
        print("\n" + "=" * 80)
        print(f"테스트 케이스 {idx}: {test_case['description']}")
        print("=" * 80)
        print(f"\n사용자 쿼리: {test_case['query']}")
        if test_case["history"]:
            print(f"이전 대화: {test_case['history']}")
        print()

        try:
            raw_output, result, parse_error = classify_intent(test_case["query"], test_case["history"])

            print("=" * 80)
            print("1. LLM이 생성한 원본 TOON 출력:")
            print("=" * 80)
            print("```toon")
            print(raw_output)
            print("```")
            print()
            print(f"원본 출력 길이: {len(raw_output)} 문자")
            print(f"원본 출력 줄 수: {len(raw_output.splitlines())} 줄")
            print()

            # 파싱 에러가 발생한 경우
            if parse_error:
                print("=" * 80)
                print("⚠️ 파싱 에러 발생:")
                print("=" * 80)
                print(parse_error)
                print()
                print("원본 출력은 위에 표시되었습니다.")
                print()
                continue

            # 파서가 TOON을 파싱한 후의 중간 결과도 보여주기
            cfg = ParserConfig(instructions_mode="minimal")
            parser = ToonOutputParser(model=IntentClassification, cfg=cfg)
            # TOON 본문 추출 (코드펜스 제거 등)
            import re

            toon_fence_re = re.compile(r"```(?:toon|text)?\s*(?P<body>.*?)```", re.DOTALL | re.IGNORECASE)
            s = raw_output.strip()
            m = toon_fence_re.search(s)
            if m:
                extracted_toon = m.group("body").strip()
            else:
                # 코드펜스가 없으면 첫 key: 부터 시작
                lines = [ln.rstrip() for ln in s.splitlines() if ln.strip() != ""]
                for idx_line, ln in enumerate(lines):
                    if ":" in ln and not ln.lstrip().startswith(("{", "[", '"')):
                        extracted_toon = "\n".join(lines[idx_line:]).strip()
                        break
                else:
                    extracted_toon = s

            print("=" * 80)
            print("1-1. 파서가 추출한 TOON 본문:")
            print("=" * 80)
            sys.stdout.flush()
            print("```toon")
            sys.stdout.flush()
            # 긴 TOON 텍스트를 청크로 나눠서 출력
            chunk_size = 2000
            for i in range(0, len(extracted_toon), chunk_size):
                chunk = extracted_toon[i : i + chunk_size]
                print(chunk, end="", flush=True)
            print()  # 마지막 줄바꿈
            print("```")
            sys.stdout.flush()
            print()
            print(f"추출된 TOON 길이: {len(extracted_toon)} 문자")
            print(f"추출된 TOON 줄 수: {len(extracted_toon.splitlines())} 줄")
            print()
            sys.stdout.flush()

            # result가 None이면 파싱 실패
            if result is None:
                print("=" * 80)
                print("⚠️ 파싱 실패 - 결과를 생성할 수 없습니다.")
                print("=" * 80)
                if parse_error:
                    print(f"파싱 에러 메시지: {parse_error}")
                    print()
                print("위의 원본 TOON 출력을 확인하여 문제를 파악하세요.")
                print()
                sys.stdout.flush()
                continue

            print("=" * 80)
            print("2. 최종 JSON으로 변환된 결과 (Pydantic 검증 후):")
            print("=" * 80)
            sys.stdout.flush()

            print("[DEBUG] JSON 변환 시작...")
            sys.stdout.flush()

            json_result = result.model_dump()
            json_output = ""  # 초기화

            print("[DEBUG] model_dump() 완료, JSON 직렬화 시도...")
            sys.stdout.flush()

            try:
                json_output = json.dumps(json_result, ensure_ascii=False, indent=2)
                print(f"[DEBUG] JSON 직렬화 성공, 길이: {len(json_output)} 문자")
                sys.stdout.flush()

                # 긴 JSON을 청크로 나눠서 출력
                chunk_size = 2000
                print("[DEBUG] JSON 출력 시작 (청크 단위)...")
                sys.stdout.flush()

                for i in range(0, len(json_output), chunk_size):
                    chunk = json_output[i : i + chunk_size]
                    print(chunk, end="", flush=True)
                print()  # 마지막 줄바꿈
                sys.stdout.flush()

                print("[DEBUG] JSON 출력 완료")
                sys.stdout.flush()

            except Exception as json_err:
                print(f"⚠️ JSON 직렬화 중 오류: {json_err}")
                import traceback

                traceback.print_exc()
                sys.stdout.flush()

                print("대신 Pydantic 객체를 직접 출력합니다:")
                sys.stdout.flush()

                try:
                    fallback_output = str(result)[:500] + "..." if len(str(result)) > 500 else str(result)
                    print(fallback_output)
                    json_output = fallback_output  # 비교를 위해 할당
                    print(f"[DEBUG] Fallback 출력 완료, 길이: {len(json_output)} 문자")
                except Exception as e:
                    print(f"Pydantic 객체 출력도 실패했습니다: {e}")
                    import traceback

                    traceback.print_exc()
                    json_output = ""  # 빈 문자열로 설정
                sys.stdout.flush()

            print()
            sys.stdout.flush()

            print("[DEBUG] JSON 출력 단계 완료, 다음 단계로 진행...")
            sys.stdout.flush()

            # TOON vs JSON 글자수 비교
            print("[DEBUG] 글자수 비교 단계 시작...")
            sys.stdout.flush()

            print("=" * 80)
            print("📊 TOON vs JSON 글자수 비교:")
            print("=" * 80)
            sys.stdout.flush()

            toon_length = len(extracted_toon)
            json_length = len(json_output)

            print(f"[DEBUG] TOON 길이: {toon_length}, JSON 길이: {json_length}")
            sys.stdout.flush()

            # 줄 수 비교
            toon_lines = len(extracted_toon.splitlines())
            json_lines = len(json_output.splitlines())

            print(f"TOON 형식:")
            print(f"  - 글자수: {toon_length:,} 자")
            print(f"  - 줄 수: {toon_lines:,} 줄")
            print()
            print(f"JSON 형식:")
            print(f"  - 글자수: {json_length:,} 자")
            print(f"  - 줄 수: {json_lines:,} 줄")
            print()

            if toon_length < json_length:
                diff = json_length - toon_length
                savings = (diff / json_length) * 100
                print(f"✅ TOON이 JSON보다 {diff:,} 자 ({savings:.1f}%) 더 짧습니다!")
                print(f"   절약된 글자수: {diff:,} 자")
                print(f"   줄 수 차이: {json_lines - toon_lines} 줄")
            elif json_length < toon_length:
                diff = toon_length - json_length
                overhead = (diff / json_length) * 100
                print(f"⚠️ JSON이 TOON보다 {diff:,} 자 ({overhead:.1f}%) 더 짧습니다.")
                print(f"   추가된 글자수: {diff:,} 자")
                print(f"   줄 수 차이: {toon_lines - json_lines} 줄")
            else:
                print("동일한 글자수입니다.")
            print()

            # 압축률 계산
            compression_ratio = (1 - toon_length / json_length) * 100 if json_length > 0 else 0
            print(f"압축률: {compression_ratio:.1f}%")

            # 효율성 비교
            if json_length > 0:
                efficiency = (toon_length / json_length) * 100
                print(f"TOON 효율성: {efficiency:.1f}% (JSON 대비)")
            print()

            # 상세 비교
            print("=" * 80)
            print("📈 상세 비교:")
            print("=" * 80)
            print(
                f"글자수 비율: TOON / JSON = {toon_length / json_length:.3f}" if json_length > 0 else "비율 계산 불가"
            )
            print(f"줄 수 비율: TOON / JSON = {toon_lines / json_lines:.3f}" if json_lines > 0 else "비율 계산 불가")
            if json_length > 0:
                bytes_saved = json_length - toon_length
                print(f"절약된 바이트: {bytes_saved:,} bytes")
            print()

            # 분류 결과 요약
            print("=" * 80)
            print("3. 분류 결과 요약:")
            print("=" * 80)
            print(f"라우팅 이유: {result.reason}")
            print()
            print(f"목표 타입: {result.goal_type}")
            goal_type_kr = "일반챗" if result.goal_type == "general_chat" else "마케팅 메시지 생성"
            print(f"  → {goal_type_kr}")
            print()
            print(f"Rewrite된 쿼리: {result.query}")
            print()
            is_new_kr = "새로운 질문" if result.is_new_question else "이어서 하는 질문"
            print(f"질문 유형: {result.is_new_question} → {is_new_kr}")
            print()
            
            # ========================================================================
            # 🔥 실제 사용 비용 분석
            # ========================================================================
            print("\n")
            cfg = ParserConfig(instructions_mode="minimal")
            analysis = CostAnalyzer.analyze_actual_usage(
                model=IntentClassification,
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

            print("\n전체 스택 트레이스:")
            traceback.print_exc()
            sys.stdout.flush()
            continue

    print("\n" + "=" * 80)
    print("모든 테스트 케이스 완료")
    print("=" * 80)


if __name__ == "__main__":
    main()
