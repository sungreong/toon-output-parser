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


class DocumentContent(BaseModel):
    """다중 라인 텍스트가 포함된 문서."""

    title: str = Field(..., description="문서 제목")
    short_description: str = Field(..., description="짧은 설명 (한 줄)")
    long_description: str = Field(..., description="긴 설명 (여러 줄 가능)")
    code_example: str = Field(..., description="코드 예시 (여러 줄)")
    notes: list[str] = Field(default_factory=list, description="노트 리스트 (각각 여러 줄 가능)")


def extract_multiline_text(document: str) -> tuple[str, DocumentContent | None, str | None, dict | None]:
    """문서에서 다중 라인 텍스트를 추출합니다.

    Args:
        document: 다중 라인 텍스트가 포함된 문서

    Returns:
        tuple[str, DocumentContent | None, str | None, dict | None]: 
        (raw_output, result, parse_error, cost_analysis)
    """
    cfg = ParserConfig(instructions_mode="minimal")
    parser = ToonOutputParser(model=DocumentContent, cfg=cfg)

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
    )

    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 문서에서 다중 라인 텍스트를 정확하게 추출하는 전문가입니다. "
                "긴 설명, 코드 블록, 여러 줄 노트 등을 올바르게 처리해야 합니다.",
            ),
            (
                "human",
                """다음 문서에서 다중 라인 텍스트를 추출해주세요.

문서:
{document}

추출해야 할 정보:
1. title: 문서 제목 (한 줄)
2. short_description: 짧은 설명 (한 줄)
3. long_description: 긴 설명 (여러 줄 가능)
4. code_example: 코드 예시 (여러 줄)
5. notes: 노트 리스트 (각 노트는 여러 줄 가능)

⚠️ CRITICAL: 다중 라인 텍스트 처리 규칙
- 여러 줄 텍스트는 한 줄로 연결하거나 특수 문자로 구분할 수 있습니다
- 줄바꿈은 \\n으로 표현하거나 공백으로 대체할 수 있습니다
- 코드 블록은 그대로 유지하되, 줄바꿈을 \\n으로 표현하세요
- 따옴표로 감싸서 특수 문자를 보호하세요

예시:
```toon
long_description: "첫 번째 줄\\n두 번째 줄\\n세 번째 줄"
code_example: "def hello():\\n    print('Hello')\\n    return True"
```

주의사항:
- 모든 필수 필드를 포함하세요
- 여러 줄 텍스트는 \\n으로 줄바꿈을 표현하세요
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
            model=DocumentContent,
            toon_raw_output=raw_output,
            parsed_result=result,
            cfg=cfg,
        )
        return raw_output, result, None, cost_analysis
    except Exception as e:
        return raw_output, None, str(e), None


def main() -> None:
    """테스트용 메인 함수 (다중 라인 텍스트 예시)."""
    test_document = """
    API 문서:
    
    제목: 사용자 인증 API
    
    짧은 설명: 사용자 로그인 및 인증을 처리하는 REST API입니다.
    
    긴 설명:
    이 API는 사용자의 로그인 요청을 받아서 인증을 처리합니다.
    성공 시 JWT 토큰을 발급하며, 실패 시 에러 메시지를 반환합니다.
    보안을 위해 비밀번호는 해시화되어 저장되며, 세션 관리는 Redis를 사용합니다.
    
    코드 예시:
    ```python
    def authenticate_user(username: str, password: str) -> dict:
        user = get_user_by_username(username)
        if not user:
            return {"error": "User not found"}
        
        if verify_password(password, user.hashed_password):
            token = generate_jwt_token(user.id)
            return {"token": token, "user_id": user.id}
        else:
            return {"error": "Invalid password"}
    ```
    
    노트:
    1. 이 API는 rate limiting이 적용됩니다.
       초당 10회 요청을 초과하면 429 에러가 반환됩니다.
    
    2. 토큰은 24시간 후 만료됩니다.
       만료 전에 refresh token을 사용하여 갱신할 수 있습니다.
    
    3. 보안을 위해 HTTPS를 사용해야 합니다.
       HTTP로 요청하면 403 에러가 반환됩니다.
    """

    print("=" * 80)
    print("다중 라인 텍스트 처리 예시")
    print("=" * 80)
    print(f"\n입력 문서:\n{test_document}\n")

    try:
        raw_output, result, parse_error, cost_analysis = extract_multiline_text(test_document)

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
            print("3. 다중 라인 텍스트 확인:")
            print("=" * 80)
            print(f"Title: {result.title}")
            print(f"\nShort Description: {result.short_description}")
            print(f"\nLong Description:")
            print(result.long_description.replace("\\n", "\n"))
            print(f"\nCode Example:")
            print(result.code_example.replace("\\n", "\n"))
            print(f"\nNotes ({len(result.notes)}개):")
            for i, note in enumerate(result.notes, 1):
                print(f"\n  [{i}]")
                print(note.replace("\\n", "\n"))

            print("\n" + "=" * 80)
            print("4. 텍스트 길이 통계:")
            print("=" * 80)
            print(f"Title 길이: {len(result.title)} 문자")
            print(f"Short Description 길이: {len(result.short_description)} 문자")
            print(f"Long Description 길이: {len(result.long_description)} 문자")
            print(f"Code Example 길이: {len(result.code_example)} 문자")
            print(f"Total Notes 길이: {sum(len(note) for note in result.notes)} 문자")
            
            # 줄바꿈 확인
            newline_count_long = result.long_description.count("\\n")
            newline_count_code = result.code_example.count("\\n")
            print(f"\n줄바꿈 수:")
            print(f"  Long Description: {newline_count_long}개")
            print(f"  Code Example: {newline_count_code}개")
            
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
