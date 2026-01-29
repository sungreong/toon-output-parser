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
            "langchain-openai 또는 langchain-community가 필요합니다. "
            "설치: pip install langchain-openai"
        ) from None

from toon_langchain_parser import ToonOutputParser, CostAnalyzer
from toon_langchain_parser.toon_parser_ultimate import ParserConfig


class CharacterFeatures(BaseModel):
    """캐릭터의 이름과 상세 특징을 추출하는 모델."""
    name: str = Field(..., description="캐릭터의 이름")
    personality: str = Field(default="", description="성격 특징")
    appearance: str = Field(default="", description="외형 특징")


def extract_character_info(document: str) -> CharacterFeatures:
    """문서에서 캐릭터 정보를 추출합니다.
    
    Args:
        document: 캐릭터에 대한 설명이 포함된 문서
        
    Returns:
        CharacterFeatures: 추출된 캐릭터 정보
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
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 문서에서 캐릭터 정보를 추출하는 전문가입니다."),
        ("human", """다음 문서에서 캐릭터의 이름과 상세 특징(성격, 외형)을 추출해주세요.

문서:
{document}

주의사항:
- 모든 필드는 반드시 문자열(string) 타입으로 출력해야 합니다.
- personality와 appearance는 딕셔너리가 아닌 문자열로 출력하세요.
- 정보가 없으면 빈 문자열("")로 설정하세요.
- 딕셔너리나 객체를 사용하지 마세요.

{format_instructions}"""),
    ])
    
    # 체인 구성: 프롬프트 -> LLM -> 문자열 출력
    llm_chain = prompt | llm | StrOutputParser()
    
    # LLM 출력 확인 (디버깅용)
    raw_output = llm_chain.invoke({"document": document, "format_instructions": format_instructions})
    
    # TOON 파싱
    result = parser.parse(raw_output)
    
    return raw_output, result


def main() -> None:
    """테스트용 메인 함수."""
    # 테스트용 문서 예시
    test_document = """
    해리 포터는 검은 머리에 초록색 눈을 가진 소년이다. 
    그는 용감하고 정의로운 성격을 가지고 있으며, 친구들을 위해 자신을 희생할 수 있는 자상한 면이 있다.
    얼굴에는 번개 모양의 흉터가 있고, 항상 둥근 안경을 쓰고 다닌다.
    """
    
    print("=" * 80)
    print("📝 문서에서 캐릭터 정보 추출")
    print("=" * 80)
    print(f"\n입력 문서:\n{test_document}\n")
    
    try:
        # LLM 호출하여 정보 추출
        raw_output, result = extract_character_info(test_document)
        
        print("=" * 80)
        print("1. LLM이 출력한 원본 결과 (TOON 형식):")
        print("=" * 80)
        print(raw_output)
        print()
        
        print("=" * 80)
        print("2. JSON으로 변환된 결과:")
        print("=" * 80)
        json_result = result.model_dump()
        print(json.dumps(json_result, ensure_ascii=False, indent=2))
        print()
        
        print("=" * 80)
        print("3. 추출된 필드별 결과:")
        print("=" * 80)
        print(f"이름: {result.name}")
        print(f"성격: {result.personality}")
        print(f"외형: {result.appearance}")
        print()
        
        # ========================================================================
        # 🔥 실제 사용 비용 분석 - 이게 핵심!
        # ========================================================================
        print("\n")
        cfg = ParserConfig(instructions_mode="minimal")
        analysis = CostAnalyzer.analyze_actual_usage(
            model=CharacterFeatures,
            toon_raw_output=raw_output,
            parsed_result=result,
            cfg=cfg,
        )
        
        CostAnalyzer.print_actual_usage_analysis(analysis)
        
        print()
        print("💡 팁: 위 분석은 실제 LLM 호출 결과를 기반으로 합니다.")
        print("   - toon_raw_output: LLM이 출력한 원본 TOON 문자열")
        print("   - parsed_result: 파싱된 Pydantic 객체")
        print("   - JSON 출력은 동일한 데이터를 JSON으로 변환한 것과 비교합니다.")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()
