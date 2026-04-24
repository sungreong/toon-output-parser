# TOON Output Parser for LangChain (한국어)

TOON 형식의 압축된 출력 텍스트를 Pydantic 모델로 검증/복원하는 구조화 출력 파서입니다.

[English README](README.md)

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-Integration-green.svg)](https://github.com/langchain-ai/langchain)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-red.svg)](https://docs.pydantic.dev/)

## TOON을 쓰는 이유

TOON은 LLM 추출 워크플로우를 위한 간결한 들여쓰기 기반 포맷입니다. 파서는 TOON 텍스트를 dict/object로 복원하고 원래의 Pydantic 스키마로 검증합니다.

## 문법 예시

```toon
name: John Doe
age: 30
hobbies[2]: soccer,coding
address:
  city: Seoul
```

배열 표현:

```toon
items:
  - name: Item 1
  - name: Item 2

products[2]{name,price}:
  iPhone 15,1200000
  Galaxy S24,1100000
```

## 기능 매트릭스

| 기능 | 지원 | 비고 |
| :--- | :---: | :--- |
| 스칼라 | Yes | `str`, `int`, `float`, `bool`, `None` |
| 중첩 객체 | Yes | 2칸 들여쓰기 |
| 인라인 스칼라 리스트 | Yes | `tags[3]: red,green,blue` |
| 테이블형 객체 배열 | Yes | `items[N]{f1,f2}:` |
| 구분자 | Yes | comma 기본, tabular header의 pipe/tab 지원 |
| 안전한 경로 확장 | Opt-in | `ParserConfig(expand_paths="safe")` |
| 스마트 문자열 안전성 | Yes | 날짜/시간, URL, 이메일, ID 문자열은 quote 없이 허용 |
| 재귀 스키마 처리 | Auto fallback | `adaptive`에서 JSON 모드로 자동 전환 |

## 설치

```bash
# 소스 설치 (PyPI 미배포)
git clone https://github.com/sungreong/toon-output-parser.git
cd toon-output-parser
python -m pip install -e .

# 선택 extras
python -m pip install -e ".[langchain]"
python -m pip install -e ".[openai]"
python -m pip install -e ".[community]"
python -m pip install -e ".[dev]"
```

## 빠른 시작

```python
from pydantic import BaseModel, Field
from toon_langchain_parser import ToonOutputParser

class UserInfo(BaseModel):
    name: str = Field(..., description="User name")
    age: int = Field(..., description="User age")
    hobbies: list[str] = Field(default_factory=list)

parser = ToonOutputParser(model=UserInfo)
result = parser.parse("name: John\nage: 25\nhobbies[2]: soccer,coding")
```

스마트 문자열 처리:

```toon
document_id: INS-2026-0001
extraction_date: 2026-04-24 12:40:15
url: https://example.com/a:b
note: "key: value"
```

## 테스트할 때 실행할 스크립트

```bash
# 1) 코어 파서 스모크 테스트
python scripts/smoke_check.py

# 2) pytest 전체 테스트
python -m pytest -q

# 3) LangChain LCEL 진단(수동)
python tests/diagnostics/verify_lcel.py
```

Docker:

```bash
docker compose build
docker compose run --rm toon-dev python scripts/smoke_check.py
docker compose run --rm toon-dev python -m pytest -q
```

## LangChain LCEL

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Describe {input}\\n\\n{format_instructions}"),
])

chain = prompt | llm | parser
result = chain.invoke(
    {
        "input": "John, 25 years old, likes soccer and coding.",
        "format_instructions": parser.get_format_instructions(),
    }
)
```

## 모드 및 제약

- 기본 모드는 `adaptive`입니다.
- `adaptive`에서는 hard-risk 스키마를 JSON 모드로 전환합니다. depth/width 같은 soft-risk는 기본적으로 TOON에 남깁니다.
- `get_effective_mode()`는 실제 런타임 모드인 `toon`, `minimal`, `json` 중 하나를 반환합니다.
- `ParserConfig(fallback_policy="toon_first" | "balanced" | "safe")`로 soft-risk를 얼마나 적극적으로 JSON fallback할지 조절합니다.
- `get_mode_decision()`은 hard reason, soft reason, risk score, schema metrics를 반환합니다.
- 기본 문자열 정책은 `ParserConfig(string_safety="smart")`입니다. 날짜/시간, URL, 이메일, 타임존, ID 같은 안전한 scalar 문자열은 quote 없이 허용하고, `note: "key: value"`처럼 모호한 일반 텍스트는 quote를 요구합니다.
- `minimal` 모드는 더 엄격한 복잡도 검증을 적용합니다.
- TOON은 들여쓰기 민감 포맷이므로, 들여쓰기 오류, 개수 불일치, 스키마 불일치 구조는 검증에 실패합니다.
- 이 패키지는 LLM 구조화 추출에 필요한 공식 TOON 핵심 호환성을 목표로 하며, TOON으로 생성하기 위험하거나 비효율적인 스키마는 JSON으로 폴백합니다.

## Experimental 상태

> [!WARNING]
> 이 프로젝트는 Beta/Experimental 상태입니다. 추출 워크플로우의 토큰 효율을 목표로 하며, 모든 native JSON mode 시나리오를 완전히 대체하지는 않습니다.

## 라이선스

MIT License. 자세한 내용은 [LICENSE](LICENSE)를 참고하세요.
