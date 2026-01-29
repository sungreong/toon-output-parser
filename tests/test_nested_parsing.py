"""중첩 배열 파싱 테스트"""
from pydantic import BaseModel, Field
from src.toon_langchain_parser import ToonOutputParser

# 간단한 중첩 구조 테스트
class Trait(BaseModel):
    name: str
    level: str

class Person(BaseModel):
    name: str
    traits: list[Trait]

test_toon = """
name: John
traits:
  - name: 책임감
    level: 높음
  - name: 반사신경
    level: 뛰어남
"""

parser = ToonOutputParser(model=Person)

print("=" * 80)
print("중첩 배열 파싱 테스트")
print("=" * 80)
print("\n입력 TOON:")
print(test_toon)
print("\n파싱 시도 중...")

try:
    result = parser.parse(test_toon)
    print("\n✅ 파싱 성공!")
    print(result.model_dump())
except Exception as e:
    print(f"\n❌ 파싱 실패: {e}")
    import traceback
    traceback.print_exc()
