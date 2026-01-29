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


# Level 3: 가장 깊은 레벨 - 개인의 특성
class PersonTrait(BaseModel):
    """개인의 특성 정보 (Depth 3)."""

    name: str = Field(..., description="특성 이름")
    level: str = Field(default="", description="특성 수준 (높음/보통/낮음)")
    description: str = Field(default="", description="특성 설명")


# Level 2: 개인 정보
class Person(BaseModel):
    """개인 정보 (Depth 2)."""

    name: str = Field(..., description="이름")
    age: int | None = Field(None, description="나이")
    role: str = Field(default="", description="역할 또는 직업")
    traits: list[PersonTrait] = Field(default_factory=list, description="개인 특성 리스트")
    skills: list[str] = Field(default_factory=list, description="보유 기술 리스트")
    background: str = Field(default="", description="배경 정보")


# Level 1: 최상위 - 팀 또는 그룹
class TeamAnalysis(BaseModel):
    """팀 분석 결과 (Depth 1)."""

    team_name: str = Field(..., description="팀 이름")
    total_members: int = Field(..., description="총 인원 수")
    members: list[Person] = Field(default_factory=list, description="팀원 리스트")
    team_strengths: list[str] = Field(default_factory=list, description="팀 강점 리스트")
    team_weaknesses: list[str] = Field(default_factory=list, description="팀 약점 리스트")
    overall_assessment: str = Field(default="", description="전체 평가")


def extract_team_analysis(document: str) -> TeamAnalysis:
    """문서에서 팀 분석 정보를 추출합니다 (동적 리스트 구조).

    Args:
        document: 팀에 대한 설명이 포함된 문서

    Returns:
        TeamAnalysis: 추출된 팀 분석 정보
    """
    # ToonOutputParser 초기화
    cfg = ParserConfig(instructions_mode="minimal")
    parser = ToonOutputParser(model=TeamAnalysis, cfg=cfg)

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
                "당신은 문서에서 팀 구성원과 특성을 분석하는 전문가입니다. 동적으로 생성되는 리스트 구조를 정확하게 추출해야 합니다.",
            ),
            (
                "human",
                """다음 문서에서 팀 정보와 각 구성원의 특성을 추출해주세요.

문서:
{document}

추출해야 할 정보:
1. team_name: 팀 이름
2. total_members: 총 인원 수
3. members: 팀원 리스트 (각 팀원마다):
   - name: 이름
   - age: 나이 (있으면)
   - role: 역할/직업
   - traits: 특성 리스트 (각 특성마다):
     - name: 특성 이름
     - level: 특성 수준
     - description: 특성 설명
   - skills: 보유 기술 리스트
   - background: 배경 정보
4. team_strengths: 팀 강점 리스트
5. team_weaknesses: 팀 약점 리스트
6. overall_assessment: 전체 평가

주의사항:
- members는 동적으로 생성되는 리스트입니다. 문서에 나온 모든 사람을 포함해야 합니다.
- 각 사람의 traits도 동적으로 생성되는 리스트입니다.
- 리스트 항목은 `-`로 시작하는 형식을 사용하세요.
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
    except Exception as e:
        # 파싱 에러가 발생해도 raw_output은 반환
        return raw_output, None, str(e)

    return raw_output, result, None


def main() -> None:
    """테스트용 메인 함수 (동적 리스트 구조 예시)."""
    # 테스트용 문서 예시 - 여러 사람이 각각 다른 특성을 가진 경우
    test_document = """
    호그와트 마법학교의 그리핀도르 퀴디치 팀은 총 7명의 선수로 구성되어 있습니다.
    
    팀의 주장은 올리버 우드입니다. 그는 17세의 골키퍼로, 퀴디치에 대한 열정과 리더십이 뛰어납니다. 
    그의 특성으로는 책임감이 높고, 팀원들을 잘 이끄는 리더십, 그리고 골키퍼로서의 뛰어난 반사신경을 가지고 있습니다.
    그는 퀴디치 경기 전략을 세우는 데 능숙하며, 팀원들의 사기를 북돋우는 능력이 있습니다.
    
    수색꾼은 해리 포터입니다. 그는 14세로, 최연소 퀴디치 선수입니다. 
    해리의 특성은 용감함이 매우 높고, 자연스러운 비행 재능, 그리고 스니치를 찾는 예리한 직감력을 가지고 있습니다.
    그는 골든 스니치를 찾는 데 탁월한 능력을 보여주며, 위험한 상황에서도 침착함을 유지합니다.
    
    또 다른 수색꾼은 케이티 벨입니다. 그녀는 15세로 경험이 풍부한 선수입니다.
    케이티의 특성으로는 인내심이 높고, 팀워크가 뛰어나며, 정확한 비행 기술을 가지고 있습니다.
    그녀는 오랜 기간 퀴디치를 해온 경험으로 팀의 안정성을 제공합니다.
    
    추격꾼은 프레드 위즐리입니다. 그는 16세로, 쌍둥이 중 한 명입니다.
    프레드의 특성은 유머 감각이 높고, 창의적인 플레이, 그리고 팀원들과의 협력 능력이 뛰어납니다.
    그는 경기를 즐기는 마인드로 팀 분위기를 밝게 만듭니다.
    
    또 다른 추격꾼은 조지 위즐리입니다. 그는 16세로, 프레드의 쌍둥이 형제입니다.
    조지의 특성은 전략적 사고가 뛰어나고, 프레드와의 완벽한 호흡, 그리고 빠른 판단력을 가지고 있습니다.
    쌍둥이들은 함께 플레이할 때 시너지 효과를 발휘합니다.
    
    세 번째 추격꾼은 앤젤리나 존슨입니다. 그녀는 16세로, 공격적인 플레이 스타일을 가지고 있습니다.
    앤젤리나의 특성으로는 공격성이 높고, 결정력이 뛰어나며, 골을 넣는 데 탁월한 능력을 가지고 있습니다.
    그녀는 팀의 주요 득점원 역할을 합니다.
    
    마지막으로 수비수는 알리시아 스핀넷입니다. 그녀는 17세로, 팀의 베테랑 선수입니다.
    알리시아의 특성은 방어 능력이 높고, 경험이 풍부하며, 팀을 보호하는 데 헌신적입니다.
    그녀는 팀의 마지막 방어선을 책임지고 있습니다.
    
    팀의 강점은 다양한 플레이 스타일의 조화, 강한 팀워크, 그리고 각 선수들의 개별 역량이 뛰어나다는 점입니다.
    약점으로는 일부 선수들의 경험 부족과, 때때로 감정적으로 플레이하는 경향이 있다는 점입니다.
    
    전체적으로 이 팀은 젊지만 재능이 넘치며, 미래가 밝은 팀입니다.
    """

    print("=" * 80)
    print("동적 리스트 구조 팀 분석 예시")
    print("=" * 80)
    print(f"\n입력 문서:\n{test_document}\n")

    try:
        raw_output, result, parse_error = extract_team_analysis(test_document)

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
        parser = ToonOutputParser(model=TeamAnalysis, cfg=cfg)
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
            print("프로그램을 종료합니다.")
            sys.stdout.flush()
            return

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
        print(f"글자수 비율: TOON / JSON = {toon_length / json_length:.3f}" if json_length > 0 else "비율 계산 불가")
        print(f"줄 수 비율: TOON / JSON = {toon_lines / json_lines:.3f}" if json_lines > 0 else "비율 계산 불가")
        if json_length > 0:
            bytes_saved = json_length - toon_length
            print(f"절약된 바이트: {bytes_saved:,} bytes")
        print()

        print("=" * 80)
        print("3. 추출된 팀 정보:")
        print("=" * 80)
        print(f"팀 이름: {result.team_name}")
        print(f"총 인원: {result.total_members}명")
        print(f"팀원 수: {len(result.members)}명")
        print()

        print("=" * 80)
        print("4. 각 팀원별 상세 정보:")
        print("=" * 80)
        for idx, member in enumerate(result.members, 1):
            print(f"\n[{idx}] {member.name} ({member.role})")
            if member.age:
                print(f"    나이: {member.age}세")
            print(f"    특성 수: {len(member.traits)}개")
            for trait in member.traits:
                level_info = f" ({trait.level})" if trait.level else ""
                print(f"      - {trait.name}{level_info}: {trait.description}")
            if member.skills:
                print(f"    기술: {', '.join(member.skills)}")
            if member.background:
                print(f"    배경: {member.background}")
        print()

        print("=" * 80)
        print("5. 팀 분석:")
        print("=" * 80)
        print(f"강점 ({len(result.team_strengths)}개):")
        for strength in result.team_strengths:
            print(f"  - {strength}")
        print()
        print(f"약점 ({len(result.team_weaknesses)}개):")
        for weakness in result.team_weaknesses:
            print(f"  - {weakness}")
        print()
        if result.overall_assessment:
            print(f"전체 평가: {result.overall_assessment}")
        print()

        print("=" * 80)
        print("6. 구조 깊이 확인:")
        print("=" * 80)
        print("최상위 레벨 (Depth 1): TeamAnalysis")
        print("  └─ members (Depth 2): list[Person]")
        print("      └─ Person (Depth 2):")
        print("          ├─ traits (Depth 3): list[PersonTrait]")
        print("          └─ skills (Depth 3): list[str]")
        print()
        print(f"동적 리스트 구조:")
        print(f"  - members: {len(result.members)}명 (동적)")
        for member in result.members:
            print(f"    - {member.name}: traits {len(member.traits)}개, skills {len(member.skills)}개")
        print()
        
        # ========================================================================
        # 🔥 실제 사용 비용 분석
        # ========================================================================
        print("\n")
        cfg = ParserConfig(instructions_mode="minimal")
        analysis = CostAnalyzer.analyze_actual_usage(
            model=TeamAnalysis,
            toon_raw_output=raw_output,
            parsed_result=result,
            cfg=cfg,
        )
        
        CostAnalyzer.print_actual_usage_analysis(analysis)
        print()

        print("=" * 80)
        print("7. 전체 Pydantic 객체:")
        print("=" * 80)
        print(result)
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
        raise


if __name__ == "__main__":
    main()
