from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field
from toon_langchain_parser import ToonOutputParser

class ExtractedEntities(BaseModel):
    tenant_id: Optional[str] = Field(None, description="테넌트 ID")
    system: Optional[str] = Field(None, description="시스템 이름")
    keyword: Optional[str] = Field(None, description="키워드")

class RoutingDecision(BaseModel):
    route: str = Field(..., description="search|faq|handoff")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str
    entities: Optional[ExtractedEntities] = None

parser = ToonOutputParser(model=RoutingDecision)

toon_output = '''
route: search
confidence: 0.82
reason: 키워드 "NORI 사용자 사전"이 설정 가이드를 요구함
entities:
  system: elasticsearch
  keyword: nori 사용자 사전
'''.strip()

print(parser.parse(toon_output))
