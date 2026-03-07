# TOON Output Parser for LangChain

A structured output parser for Pydantic models that accepts compact TOON text and validates it into typed objects.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-Integration-green.svg)](https://github.com/langchain-ai/langchain)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-red.svg)](https://docs.pydantic.dev/)

## Why TOON

TOON is a compact, indentation-based format intended for LLM extraction workflows. The parser restores TOON text into dict/object form and validates it with your original Pydantic schema.

## Syntax Snapshot

```toon
name: John Doe
age: 30
hobbies[2]: soccer,coding
address:
  city: Seoul
```

Array options:

```toon
items:
  - name: Item 1
  - name: Item 2

products[2,]{name,price}:
  iPhone 15,1200000
  Galaxy S24,1100000
```

## Feature Matrix

| Feature | Support | Notes |
| :--- | :---: | :--- |
| Scalars | Yes | `str`, `int`, `float`, `bool`, `None` |
| Nested objects | Yes | 2-space indentation |
| Inline scalar list | Yes | `tags[3]: red,green,blue` |
| Tabular object array | Yes | `items[N,]{f1,f2}:` |
| Dot notation | Yes (default) | Example: `details.summary: concise` |
| Dot notation disabled mode | Yes | Set `ParserConfig(allow_dotted_paths=False)` |
| Recursive schema handling | Auto fallback | `adaptive` mode routes recursive models to JSON mode |

## Installation

```bash
pip install toon-output-parser

# Optional extras
pip install "toon-output-parser[langchain]"
pip install "toon-output-parser[openai]"
pip install "toon-output-parser[community]"
```

## Quick Start

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

## Modes and Constraints

- Default mode is `adaptive`.
- In `adaptive`, recursive/high-complexity schemas can be switched to JSON mode for stability.
- `minimal` mode applies stricter complexity validation.
- TOON remains indentation-sensitive; malformed indentation and schema-incompatible shapes will fail validation.

## Experimental Status

> [!WARNING]
> This project is in Beta/Experimental status. It is optimized for token efficiency in extraction workflows, not a full replacement for every native JSON mode scenario.

## Pre-release Local Gates

Run before publishing:

```bash
python -m compileall src
python -m pytest -q
python -m ruff check .
```

Optional packaging smoke:

```bash
python -m build
pip install --force-reinstall dist/toon_output_parser-*.whl
python -c "from toon_langchain_parser import ToonOutputParser, ToonParser"
```

## License

MIT License. See [LICENSE](LICENSE).
