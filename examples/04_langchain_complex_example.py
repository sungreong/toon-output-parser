from __future__ import annotations

import json
import os

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


# Level 3: 가장 깊은 레벨의 모델들
class PersonalityTraits(BaseModel):
    """성격 특성 상세 정보 (Depth 3)."""

    traits: list[str] = Field(default_factory=list, description="성격 특성 리스트")
    summary: str = Field(default="", description="성격 요약")


class AppearanceDetails(BaseModel):
    """외형 상세 정보 (Depth 3)."""

    features: list[str] = Field(default_factory=list, description="외형 특징 리스트")
    summary: str = Field(default="", description="외형 요약")


class BackgroundInfo(BaseModel):
    """배경 정보 (Depth 3)."""

    origin: str = Field(default="", description="출신지 또는 기원")
    history: str = Field(default="", description="과거 이력")


# Level 2: 중간 레벨 모델
class CharacterDetails(BaseModel):
    """캐릭터 상세 정보 (Depth 2)."""

    personality: PersonalityTraits = Field(default_factory=PersonalityTraits, description="성격 상세 정보")
    appearance: AppearanceDetails = Field(default_factory=AppearanceDetails, description="외형 상세 정보")
    background: BackgroundInfo = Field(default_factory=BackgroundInfo, description="배경 정보")


# Level 1: 최상위 레벨 모델
class CharacterFeatures(BaseModel):
    """캐릭터의 이름과 상세 특징을 추출하는 모델 (Depth 1)."""

    name: str = Field(..., description="캐릭터의 이름")
    age: int | None = Field(None, description="나이")
    details: CharacterDetails = Field(default_factory=CharacterDetails, description="상세 정보")


def extract_character_info(document: str) -> tuple[str, CharacterFeatures | None, str | None, dict | None]:
    """문서에서 캐릭터 정보를 추출합니다 (Depth 3 중첩 구조).

    Args:
        document: 캐릭터에 대한 설명이 포함된 문서

    Returns:
        tuple[str, CharacterFeatures | None, str | None, dict | None]: 
        (raw_output, result, parse_error, cost_analysis)
    """
    # ToonOutputParser 초기화
    cfg = ParserConfig(instructions_mode="minimal")
    parser = ToonOutputParser(model=CharacterFeatures, cfg=cfg)

    # ChatOpenAI 모델 초기화
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
    )

    # 프롬프트 템플릿 생성
    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 문서에서 캐릭터 정보를 추출하는 전문가입니다. 중첩된 구조의 정보를 정확하게 추출해야 합니다.",
            ),
            (
                "human",
                """다음 문서에서 캐릭터의 이름과 상세 특징을 추출해주세요.

문서:
{document}

추출해야 할 정보:
1. name: 캐릭터 이름
2. age: 나이 (있으면)
3. details.personality.traits: 성격 특성 리스트
4. details.personality.summary: 성격 요약
5. details.appearance.features: 외형 특징 리스트
6. details.appearance.summary: 외형 요약
7. details.background.origin: 출신지/기원
8. details.background.history: 과거 이력

주의사항:
- 중첩된 구조를 정확하게 따라야 합니다.
- 리스트 필드는 배열로 출력하세요.
- 정보가 없으면 빈 문자열("") 또는 빈 배열([])로 설정하세요.
- TOON 형식의 들여쓰기를 정확하게 지켜주세요.

{format_instructions}""",
            ),
        ]
    )

    # 체인 구성: 프롬프트 -> LLM -> 문자열 출력
    llm_chain = prompt | llm | StrOutputParser()

    # LLM 출력 확인 (디버깅용)
    raw_output = llm_chain.invoke({"document": document, "format_instructions": format_instructions})

    # TOON 파싱 (에러가 발생해도 raw_output은 반환)
    try:
        result = parser.parse(raw_output)
        # 비용 분석
        cost_analysis = CostAnalyzer.analyze_actual_usage(
            model=CharacterFeatures,
            toon_raw_output=raw_output,
            parsed_result=result,
            cfg=cfg,
        )
        return raw_output, result, None, cost_analysis
    except Exception as e:
        # 파싱 에러가 발생해도 raw_output은 반환
        return raw_output, None, str(e), None


def main() -> None:
    """테스트용 메인 함수 (Depth 3 중첩 구조 예시)."""
    # 테스트용 문서 예시 (더 상세한 정보 포함)
    test_document = """
    해리 제임스 포터는 17세의 마법사이다. 그는 영국에서 태어났고, 어린 시절부터 어둠의 마법사 볼드모트와의 연결고리를 가지고 있었다.
    
    성격 면에서 해리는 용감함, 정의로움, 충성심, 자상함, 인내심 등의 특성을 가지고 있다. 
    그는 위험을 무릅쓰고 친구들을 구하는 것을 주저하지 않으며, 불의에 맞서 싸우는 강한 의지를 가지고 있다.
    
    외형적으로 해리는 검은 머리, 초록색 눈, 둥근 안경, 번개 모양의 이마 흉터, 마른 체형 등의 특징을 가지고 있다.
    그는 항상 검은 로브를 입고 다니며, 마법 지팡이를 소지하고 있다.
    
    해리는 볼드모트에 의해 부모를 잃고 머글 가정에서 자랐으며, 11세에 호그와트 마법학교에 입학했다.
    그는 그리핀도르 기숙사에 배정되었고, 퀴디치 팀의 수색꾼으로 활약했다.
    """

    print("=" * 80)
    print("Depth 3 중첩 구조 캐릭터 정보 추출 예시")
    print("=" * 80)
    print(f"\n입력 문서:\n{test_document}\n")

    try:
        raw_output, result, parse_error, cost_analysis = extract_character_info(test_document)

        print("=" * 80)
        print("1. LLM이 생성한 원본 TOON 출력 (AI가 실제로 생성한 값):")
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
            print("원본 출력은 위에 표시되었습니다. 파싱을 계속 시도합니다...")
            print()

        # 파서가 TOON을 파싱한 후의 중간 결과도 보여주기
        cfg = ParserConfig(instructions_mode="minimal")
        parser = ToonOutputParser(model=CharacterFeatures, cfg=cfg)
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
            for idx, ln in enumerate(lines):
                if ":" in ln and not ln.lstrip().startswith(("{", "[", '"')):
                    extracted_toon = "\n".join(lines[idx:]).strip()
                    break
            else:
                extracted_toon = s

        print("=" * 80)
        print("1-1. 파서가 추출한 TOON 본문:")
        print("=" * 80)
        print("```toon")
        print(extracted_toon)
        print("```")
        print()

        # TOON을 디코딩한 결과 (파싱 전)
        try:
            decoded_before_validation = parser.decode(extracted_toon)
            print("=" * 80)
            print("1-2. TOON 디코딩 결과 (Pydantic 검증 전):")
            print("=" * 80)
            print(json.dumps(decoded_before_validation, ensure_ascii=False, indent=2))
            print()
        except Exception as e:
            print(f"디코딩 중 오류: {e}")
            print()

        # result가 None이면 파싱 실패
        if result is None:
            print("=" * 80)
            print("⚠️ 파싱 실패 - 결과를 생성할 수 없습니다.")
            print("=" * 80)
            print("위의 원본 출력과 디코딩 결과를 확인하여 문제를 파악하세요.")
            return

        print("=" * 80)
        print("2. 최종 JSON으로 변환된 결과 (Pydantic 검증 후):")
        print("=" * 80)
        json_result = result.model_dump()
        json_output = json.dumps(json_result, ensure_ascii=False, indent=2)
        print(json_output)
        print()

        # TOON vs JSON 글자수 비교
        print("=" * 80)
        print("📊 TOON vs JSON 글자수 비교:")
        print("=" * 80)
        toon_length = len(extracted_toon)
        json_length = len(json_output)

        print(f"TOON 형식 글자수: {toon_length:,} 자")
        print(f"JSON 형식 글자수: {json_length:,} 자")
        print()

        if toon_length < json_length:
            diff = json_length - toon_length
            savings = (diff / json_length) * 100
            print(f"✅ TOON이 JSON보다 {diff:,} 자 ({savings:.1f}%) 더 짧습니다!")
            print(f"   절약된 글자수: {diff:,} 자")
        elif json_length < toon_length:
            diff = toon_length - json_length
            overhead = (diff / json_length) * 100
            print(f"⚠️ JSON이 TOON보다 {diff:,} 자 ({overhead:.1f}%) 더 짧습니다.")
            print(f"   추가된 글자수: {diff:,} 자")
        else:
            print("동일한 글자수입니다.")
        print()

        # 압축률 계산
        compression_ratio = (1 - toon_length / json_length) * 100 if json_length > 0 else 0
        print(f"압축률: {compression_ratio:.1f}%")
        print()
        
        # ========================================================================
        # 🔥 실제 사용 비용 분석
        # ========================================================================
        if cost_analysis:
            print("\n")
            print("=" * 80)
            print("💰 TOON 파서 비용 분석:")
            print("=" * 80)
            CostAnalyzer.print_actual_usage_analysis(cost_analysis)
            print()

        print("=" * 80)
        print("3. 추출된 필드별 결과 (Depth별):")
        print("=" * 80)
        print(f"이름 (Depth 1): {result.name}")
        print(f"나이 (Depth 1): {result.age}")
        print()
        print("성격 정보 (Depth 2 -> 3):")
        print(f"  - 특성 리스트: {result.details.personality.traits}")
        print(f"  - 요약: {result.details.personality.summary}")
        print()
        print("외형 정보 (Depth 2 -> 3):")
        print(f"  - 특징 리스트: {result.details.appearance.features}")
        print(f"  - 요약: {result.details.appearance.summary}")
        print()
        print("배경 정보 (Depth 2 -> 3):")
        print(f"  - 출신지: {result.details.background.origin}")
        print(f"  - 이력: {result.details.background.history}")
        print()

        print("=" * 80)
        print("4. 구조 깊이 확인:")
        print("=" * 80)
        print("최상위 레벨 (Depth 1): CharacterFeatures")
        print("  └─ details (Depth 2): CharacterDetails")
        print("      ├─ personality (Depth 3): PersonalityTraits")
        print("      ├─ appearance (Depth 3): AppearanceDetails")
        print("      └─ background (Depth 3): BackgroundInfo")
        print()

        print("=" * 80)
        print("5. 전체 Pydantic 객체:")
        print("=" * 80)
        print(result)
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
