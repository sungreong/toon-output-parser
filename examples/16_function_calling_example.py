from __future__ import annotations

import json
import os
import sys
from typing import Literal

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


# ============================================================================
# Function Calling을 위한 파라미터 모델들
# ============================================================================


class WeatherParams(BaseModel):
    """날씨 API 호출 파라미터."""

    location: str = Field(..., description="지역명 (예: 서울, 부산, New York)")
    unit: Literal["celsius", "fahrenheit"] = Field(
        default="celsius", description="온도 단위"
    )
    days: int = Field(default=1, description="예보 일수 (1-7)")


class SearchParams(BaseModel):
    """검색 API 호출 파라미터."""

    query: str = Field(..., description="검색어")
    max_results: int = Field(default=10, description="최대 결과 수")
    language: str = Field(default="ko", description="언어 코드")


class CalculatorParams(BaseModel):
    """계산기 함수 호출 파라미터."""

    expression: str = Field(..., description="계산할 수식 (예: 2+3*4)")
    precision: int = Field(default=2, description="소수점 자릿수")


class FunctionCallRequest(BaseModel):
    """Function calling 요청 (어떤 함수를 호출할지 결정)."""

    function_name: Literal[
        "get_weather", 
        "search", 
        "calculate",
        "python_sandbox",
        "send_email",
        "get_stock_price",
        "translate",
        "image_generate"
    ] = Field(..., description="호출할 함수 이름")
    
    # get_weather 파라미터
    location: str | None = Field(None, description="지역명 (get_weather용)")
    unit: Literal["celsius", "fahrenheit"] | None = Field(None, description="온도 단위 (get_weather용)")
    days: int | None = Field(None, description="예보 일수 (get_weather용)")
    
    # search 파라미터
    query: str | None = Field(None, description="검색어 (search용)")
    max_results: int | None = Field(None, description="최대 결과 수 (search용)")
    language: str | None = Field(None, description="언어 코드 (search용)")
    
    # calculate 파라미터
    expression: str | None = Field(None, description="계산할 수식 (calculate용)")
    precision: int | None = Field(None, description="소수점 자릿수 (calculate용)")
    
    # python_sandbox 파라미터
    code: str | None = Field(None, description="실행할 파이썬 코드 (python_sandbox용)")
    timeout: int | None = Field(None, description="실행 타임아웃(초) (python_sandbox용, 기본값: 5)")
    
    # send_email 파라미터
    to: str | None = Field(None, description="받는 사람 이메일 (send_email용)")
    subject: str | None = Field(None, description="이메일 제목 (send_email용)")
    body: str | None = Field(None, description="이메일 본문 (send_email용)")
    
    # get_stock_price 파라미터
    symbol: str | None = Field(None, description="주식 심볼 (get_stock_price용, 예: AAPL, TSLA)")
    
    # translate 파라미터
    text: str | None = Field(None, description="번역할 텍스트 (translate용)")
    target_language: str | None = Field(None, description="목표 언어 코드 (translate용, 예: en, ko, ja)")
    source_language: str | None = Field(None, description="원본 언어 코드 (translate용, 선택)")
    
    # image_generate 파라미터
    prompt: str | None = Field(None, description="이미지 생성 프롬프트 (image_generate용)")
    size: Literal["256x256", "512x512", "1024x1024"] | None = Field(None, description="이미지 크기 (image_generate용, 기본값: 512x512)")
    
    def get_parameters(self) -> dict:
        """함수 이름에 맞는 파라미터 딕셔너리를 반환합니다."""
        if self.function_name == "get_weather":
            return {
                "location": self.location or "",
                "unit": self.unit or "celsius",
                "days": self.days or 1,
            }
        elif self.function_name == "search":
            return {
                "query": self.query or "",
                "max_results": self.max_results or 10,
                "language": self.language or "ko",
            }
        elif self.function_name == "calculate":
            return {
                "expression": self.expression or "",
                "precision": self.precision or 2,
            }
        elif self.function_name == "python_sandbox":
            return {
                "code": self.code or "",
                "timeout": self.timeout or 5,
            }
        elif self.function_name == "send_email":
            return {
                "to": self.to or "",
                "subject": self.subject or "",
                "body": self.body or "",
            }
        elif self.function_name == "get_stock_price":
            return {
                "symbol": self.symbol or "",
            }
        elif self.function_name == "translate":
            return {
                "text": self.text or "",
                "target_language": self.target_language or "",
                "source_language": self.source_language,
            }
        elif self.function_name == "image_generate":
            return {
                "prompt": self.prompt or "",
                "size": self.size or "512x512",
            }
        return {}


# ============================================================================
# TOON 파서를 사용한 Function Calling 파라미터 추출
# ============================================================================


def extract_function_params_with_toon(
    user_query: str,
) -> tuple[str, FunctionCallRequest | None, str | None, dict | None]:
    """TOON 파서를 사용하여 사용자 쿼리에서 함수 호출 파라미터를 추출합니다.

    Args:
        user_query: 사용자 쿼리 (예: "서울 날씨 알려줘")

    Returns:
        tuple[str, FunctionCallRequest | None, str | None, dict | None]: 
        (raw_output, result, parse_error, cost_analysis)
    """
    cfg = ParserConfig(instructions_mode="minimal")
    parser = ToonOutputParser(model=FunctionCallRequest, cfg=cfg)

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
    )

    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 사용자 쿼리를 분석하여 적절한 함수를 호출하기 위한 파라미터를 추출하는 전문가입니다. "
                "사용자의 의도를 파악하여 올바른 함수와 파라미터를 결정해야 합니다.",
            ),
            (
                "human",
                """다음 사용자 쿼리를 분석하여 함수 호출 파라미터를 추출해주세요.

사용자 쿼리: {user_query}

사용 가능한 함수들:
1. get_weather: 날씨 정보 조회
   - location: 지역명 (필수)
   - unit: 온도 단위 ("celsius" 또는 "fahrenheit", 기본값: "celsius")
   - days: 예보 일수 (1-7, 기본값: 1)

2. search: 웹 검색
   - query: 검색어 (필수)
   - max_results: 최대 결과 수 (기본값: 10)
   - language: 언어 코드 (기본값: "ko")

3. calculate: 수식 계산
   - expression: 계산할 수식 (필수)
   - precision: 소수점 자릿수 (기본값: 2)

4. python_sandbox: 파이썬 코드 실행 (안전한 샌드박스 환경)
   - code: 실행할 파이썬 코드 (필수)
   - timeout: 실행 타임아웃(초) (기본값: 5)

5. send_email: 이메일 전송
   - to: 받는 사람 이메일 주소 (필수)
   - subject: 이메일 제목 (필수)
   - body: 이메일 본문 (필수)

6. get_stock_price: 주식 가격 조회
   - symbol: 주식 심볼 (필수, 예: AAPL, TSLA, 005930)

7. translate: 텍스트 번역
   - text: 번역할 텍스트 (필수)
   - target_language: 목표 언어 코드 (필수, 예: en, ko, ja)
   - source_language: 원본 언어 코드 (선택, 자동 감지 가능)

8. image_generate: AI 이미지 생성
   - prompt: 이미지 생성 프롬프트 (필수)
   - size: 이미지 크기 ("256x256", "512x512", "1024x1024", 기본값: "512x512")

지시사항:
- 사용자 쿼리의 의도를 파악하여 적절한 function_name을 선택하세요
- 선택한 함수에 맞는 필드만 채우세요
- 사용하지 않는 필드는 생략하세요

{format_instructions}""",
            ),
        ]
    )

    llm_chain = prompt | llm | StrOutputParser()
    raw_output = llm_chain.invoke({"user_query": user_query, "format_instructions": format_instructions})

    try:
        result = parser.parse(raw_output)
        # 비용 분석
        cost_analysis = CostAnalyzer.analyze_actual_usage(
            model=FunctionCallRequest,
            toon_raw_output=raw_output,
            parsed_result=result,
            cfg=cfg,
        )
        return raw_output, result, None, cost_analysis
    except Exception as e:
        return raw_output, None, str(e), None


# ============================================================================
# 실제 함수 호출 시뮬레이션
# ============================================================================


def call_function(function_name: str, parameters: dict) -> dict:
    """실제 함수 호출을 시뮬레이션합니다.

    Args:
        function_name: 함수 이름
        parameters: 함수 파라미터

    Returns:
        dict: 함수 실행 결과
    """
    if function_name == "get_weather":
        location = parameters.get("location", "Unknown")
        unit = parameters.get("unit", "celsius")
        days = parameters.get("days", 1)
        return {
            "function": "get_weather",
            "result": f"{location}의 {days}일 날씨 예보 (단위: {unit})",
            "data": {
                "location": location,
                "temperature": 22 if unit == "celsius" else 72,
                "condition": "맑음",
                "humidity": 65,
            },
        }
    elif function_name == "search":
        query = parameters.get("query", "")
        max_results = parameters.get("max_results", 10)
        return {
            "function": "search",
            "result": f"'{query}' 검색 결과 ({max_results}개)",
            "data": {
                "query": query,
                "results": [f"결과 {i+1}" for i in range(min(max_results, 3))],
            },
        }
    elif function_name == "calculate":
        expression = parameters.get("expression", "")
        precision = parameters.get("precision", 2)
        try:
            result = eval(expression)  # 실제로는 안전한 계산 라이브러리 사용
            return {
                "function": "calculate",
                "result": f"{expression} = {result:.{precision}f}",
                "data": {"expression": expression, "result": result},
            }
        except Exception as e:
            return {
                "function": "calculate",
                "error": str(e),
            }
    elif function_name == "python_sandbox":
        code = parameters.get("code", "")
        timeout = parameters.get("timeout", 5)
        # 실제로는 안전한 샌드박스 환경에서 실행
        try:
            # 시뮬레이션: 간단한 코드만 실행
            if "import" in code or "__" in code:
                return {
                    "function": "python_sandbox",
                    "error": "안전하지 않은 코드입니다. import나 특수 메서드는 사용할 수 없습니다.",
                }
            result = eval(code) if code.strip() else None
            return {
                "function": "python_sandbox",
                "result": f"코드 실행 완료 (타임아웃: {timeout}초)",
                "data": {"code": code, "output": str(result)},
            }
        except Exception as e:
            return {
                "function": "python_sandbox",
                "error": str(e),
            }
    elif function_name == "send_email":
        to = parameters.get("to", "")
        subject = parameters.get("subject", "")
        body = parameters.get("body", "")
        return {
            "function": "send_email",
            "result": f"이메일 전송 완료: {to}",
            "data": {
                "to": to,
                "subject": subject,
                "body": body[:50] + "..." if len(body) > 50 else body,
            },
        }
    elif function_name == "get_stock_price":
        symbol = parameters.get("symbol", "")
        # 시뮬레이션: 가짜 주식 가격
        prices = {"AAPL": 175.50, "TSLA": 245.30, "005930": 75000}
        price = prices.get(symbol.upper(), 100.0)
        return {
            "function": "get_stock_price",
            "result": f"{symbol} 현재가: ${price:.2f}",
            "data": {"symbol": symbol, "price": price, "currency": "USD"},
        }
    elif function_name == "translate":
        text = parameters.get("text", "")
        target_language = parameters.get("target_language", "")
        source_language = parameters.get("source_language")
        # 시뮬레이션: 간단한 번역
        translations = {
            ("안녕하세요", "en"): "Hello",
            ("Hello", "ko"): "안녕하세요",
            ("こんにちは", "en"): "Hello",
        }
        translated = translations.get((text, target_language), f"[번역: {text} -> {target_language}]")
        return {
            "function": "translate",
            "result": translated,
            "data": {
                "text": text,
                "source_language": source_language or "auto",
                "target_language": target_language,
                "translated": translated,
            },
        }
    elif function_name == "image_generate":
        prompt = parameters.get("prompt", "")
        size = parameters.get("size", "512x512")
        return {
            "function": "image_generate",
            "result": f"이미지 생성 완료: {prompt[:30]}...",
            "data": {
                "prompt": prompt,
                "size": size,
                "image_url": f"https://example.com/generated/{hash(prompt) % 10000}.png",
            },
        }
    else:
        return {"error": f"Unknown function: {function_name}"}


# ============================================================================
# 비교 및 테스트
# ============================================================================


def test_function_calling(user_query: str) -> None:
    """Function calling 파라미터 추출을 테스트합니다."""
    print("=" * 80)
    print("🔧 Function Calling 파라미터 추출 테스트")
    print("=" * 80)
    print(f"\n사용자 쿼리: {user_query}\n")

    # TOON 파서 사용
    print("=" * 80)
    print("TOON 파서를 사용한 파라미터 추출")
    print("=" * 80)
    raw_output, result, error, cost_analysis = extract_function_params_with_toon(user_query)

    if error:
        print(f"❌ 파싱 에러: {error}")
        print(f"\n원본 출력:\n{raw_output}")
    else:
        print("✅ 파싱 성공!")
        print(f"\n원본 TOON 출력:\n{raw_output}")
        print(f"\n추출된 파라미터:")
        print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))

        # 함수 호출 시뮬레이션
        params = result.parameters if hasattr(result, 'parameters') else result.get_parameters()
        function_result = call_function(result.function_name, params)
        print(f"\n함수 호출 결과:")
        print(json.dumps(function_result, ensure_ascii=False, indent=2))

        if cost_analysis:
            print("\n" + "=" * 80)
            print("💰 TOON 파서 비용 분석:")
            print("=" * 80)
            CostAnalyzer.print_actual_usage_analysis(cost_analysis)


def main() -> None:
    """테스트용 메인 함수."""
    test_queries = [
        "서울 날씨 알려줘",
        "파이썬으로 리스트 합계 구하는 코드 실행해줘: sum([1,2,3,4,5])",
        "john@example.com에게 '회의 안내' 제목으로 이메일 보내줘",
        "애플 주식 가격 알려줘",
        "안녕하세요를 영어로 번역해줘",
        "고양이가 하늘을 나는 이미지 생성해줘",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'='*80}")
        print(f"테스트 케이스 {i}/{len(test_queries)}")
        print(f"{'='*80}\n")
        try:
            test_function_calling(query)
        except Exception as e:
            print("=" * 80)
            print("❌ 치명적 오류 발생!")
            print("=" * 80)
            print(f"오류 타입: {type(e).__name__}")
            print(f"오류 메시지: {e}")
            import traceback

            traceback.print_exc()
            sys.stdout.flush()
            if i < len(test_queries):
                print("\n다음 테스트 케이스로 계속...")
                continue
            else:
                raise

    print("\n" + "=" * 80)
    print("테스트 완료")
    print("=" * 80)


if __name__ == "__main__":
    main()
