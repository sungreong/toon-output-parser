from __future__ import annotations

import json
import os
from typing import Annotated, Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, RootModel
from eval_metrics import print_evaluation

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain_community.chat_models import ChatOpenAI

from toon_langchain_parser import CostAnalyzer, ToonOutputParser
from toon_langchain_parser.toon_parser_ultimate import ParserConfig


class WeatherParams(BaseModel):
    location: str
    unit: Literal["celsius", "fahrenheit"] = "celsius"
    days: int = Field(default=1, ge=1, le=7)


class SearchParams(BaseModel):
    query: str
    max_results: int = Field(default=10, ge=1, le=50)
    language: str = "ko"


class CalculatorParams(BaseModel):
    expression: str
    precision: int = Field(default=2, ge=0, le=8)


class PythonSandboxParams(BaseModel):
    code: str
    timeout: int = Field(default=5, ge=1, le=60)


class SendEmailParams(BaseModel):
    to: str
    subject: str
    body: str


class StockPriceParams(BaseModel):
    symbol: str


class TranslateParams(BaseModel):
    text: str
    target_language: str
    source_language: str | None = None


class ImageGenerateParams(BaseModel):
    prompt: str
    size: Literal["256x256", "512x512", "1024x1024"] = "512x512"


class GetWeatherCall(BaseModel):
    function_name: Literal["get_weather"]
    parameters: WeatherParams


class SearchCall(BaseModel):
    function_name: Literal["search"]
    parameters: SearchParams


class CalculateCall(BaseModel):
    function_name: Literal["calculate"]
    parameters: CalculatorParams


class PythonSandboxCall(BaseModel):
    function_name: Literal["python_sandbox"]
    parameters: PythonSandboxParams


class SendEmailCall(BaseModel):
    function_name: Literal["send_email"]
    parameters: SendEmailParams


class StockPriceCall(BaseModel):
    function_name: Literal["get_stock_price"]
    parameters: StockPriceParams


class TranslateCall(BaseModel):
    function_name: Literal["translate"]
    parameters: TranslateParams


class ImageGenerateCall(BaseModel):
    function_name: Literal["image_generate"]
    parameters: ImageGenerateParams


FunctionCallUnion = Annotated[
    (
        GetWeatherCall
        | SearchCall
        | CalculateCall
        | PythonSandboxCall
        | SendEmailCall
        | StockPriceCall
        | TranslateCall
        | ImageGenerateCall
    ),
    Field(discriminator="function_name"),
]


class FunctionCallRequest(RootModel[FunctionCallUnion]):
    @property
    def function_name(self) -> str:
        return self.root.function_name

    @property
    def parameters(self) -> dict:
        return self.root.parameters.model_dump(exclude_none=True)


def extract_function_params_with_toon(
    user_query: str,
) -> tuple[str, FunctionCallRequest | None, str | None, dict | None]:
    cfg = ParserConfig(instructions_mode="adaptive")
    parser = ToonOutputParser(model=FunctionCallRequest, cfg=cfg)

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Map user query to one function call. Return only schema-conformant output.",
            ),
            (
                "human",
                "User query: {user_query}\n\n"
                "Choose exactly one function_name and fill parameters.\n"
                "{format_instructions}",
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    raw_output = chain.invoke(
        {
            "user_query": user_query,
            "format_instructions": parser.get_format_instructions(),
        }
    )

    try:
        result = parser.parse(raw_output)
        cost_analysis = CostAnalyzer.analyze_actual_usage(
            model=FunctionCallRequest,
            toon_raw_output=raw_output,
            parsed_result=result,
            cfg=cfg,
        )
        return raw_output, result, None, cost_analysis
    except Exception as e:
        return raw_output, None, str(e), None


def call_function(function_name: str, parameters: dict) -> dict:
    if function_name == "get_weather":
        return {
            "location": parameters.get("location", ""),
            "unit": parameters.get("unit", "celsius"),
            "forecast": [
                {"day": 1, "temp": 24, "condition": "sunny"},
                {"day": 2, "temp": 22, "condition": "cloudy"},
            ][: parameters.get("days", 1)],
        }
    if function_name == "search":
        query = parameters.get("query", "")
        max_results = parameters.get("max_results", 10)
        return {
            "query": query,
            "results": [
                {"title": f"Result {i + 1}", "url": f"https://example.com/{i + 1}"}
                for i in range(min(max_results, 3))
            ],
        }
    if function_name == "calculate":
        expression = parameters.get("expression", "")
        try:
            value = eval(expression, {"__builtins__": {}}, {})
            return {"expression": expression, "result": value}
        except Exception:
            return {"error": "Invalid expression", "expression": expression}
    if function_name == "python_sandbox":
        return {"status": "simulated", "timeout": parameters.get("timeout", 5)}
    if function_name == "send_email":
        return {"status": "sent", "to": parameters.get("to", "")}
    if function_name == "get_stock_price":
        return {"symbol": parameters.get("symbol", ""), "price": 123.45}
    if function_name == "translate":
        return {
            "text": parameters.get("text", ""),
            "target_language": parameters.get("target_language", ""),
            "translated": "(simulated translation)",
        }
    if function_name == "image_generate":
        return {
            "prompt": parameters.get("prompt", ""),
            "size": parameters.get("size", "512x512"),
            "image_url": "https://example.com/generated-image.png",
        }
    return {"error": f"Unknown function: {function_name}"}


def main() -> None:
    test_queries = [
        "서울 3일 날씨 알려줘",
        "TSLA 주가 보여줘",
        "2+3*4 계산해줘",
        "hello를 일본어로 번역해줘",
    ]

    for query in test_queries:
        print("=" * 80)
        print(f"QUERY: {query}")
        raw_output, result, parse_error, cost_analysis = extract_function_params_with_toon(query)

        print("\nRAW:")
        print(raw_output)

        if parse_error:
            print("\nPARSE ERROR:")
            print(parse_error)
            continue

        assert result is not None
        print("\nPARSED:")
        print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
        expected = {"function_name": result.function_name}
        print_evaluation("QUALITY", result.model_dump(), expected)

        function_result = call_function(result.function_name, result.parameters)
        print("\nFUNCTION RESULT:")
        print(json.dumps(function_result, ensure_ascii=False, indent=2))

        if cost_analysis:
            print("\nCOST:")
            CostAnalyzer.print_actual_usage_analysis(cost_analysis)


if __name__ == "__main__":
    main()
