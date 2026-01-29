from __future__ import annotations

from pydantic import BaseModel, Field
from toon_langchain_parser import ToonOutputParser

class Simple(BaseModel):
    name: str
    age: int = Field(..., ge=0, le=150)

parser = ToonOutputParser(model=Simple)
print(parser.get_format_instructions())
