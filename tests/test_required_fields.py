"""필수 vs 선택 필드 테스트"""
from pydantic import BaseModel, Field, ValidationError
from src.toon_langchain_parser import ToonOutputParser

class TestModel(BaseModel):
    # 필수 필드
    required_field: str = Field(..., description="필수 필드")
    required_number: int = Field(..., description="필수 숫자")
    
    # 선택 필드
    optional_field: str = Field(default="", description="선택 필드")
    optional_number: int | None = Field(None, description="선택 숫자")
    optional_list: list[str] = Field(default_factory=list, description="선택 리스트")

parser = ToonOutputParser(model=TestModel)

print("=" * 80)
print("필수 vs 선택 필드 테스트")
print("=" * 80)

# 테스트 1: 모든 필드 포함 (성공해야 함)
test1 = """
required_field: test
required_number: 42
optional_field: optional
optional_number: 10
optional_list[2]: a,b
"""

print("\n[테스트 1] 모든 필드 포함:")
try:
    result = parser.parse(test1)
    print("✅ 성공!")
    print(f"  required_field: {result.required_field}")
    print(f"  optional_field: {result.optional_field}")
except Exception as e:
    print(f"❌ 실패: {e}")

# 테스트 2: 필수 필드만 포함 (성공해야 함)
test2 = """
required_field: test
required_number: 42
"""

print("\n[테스트 2] 필수 필드만 포함:")
try:
    result = parser.parse(test2)
    print("✅ 성공!")
    print(f"  required_field: {result.required_field}")
    print(f"  optional_field: '{result.optional_field}' (기본값)")
    print(f"  optional_number: {result.optional_number} (기본값)")
    print(f"  optional_list: {result.optional_list} (기본값)")
except Exception as e:
    print(f"❌ 실패: {e}")

# 테스트 3: 필수 필드 누락 (실패해야 함)
test3 = """
optional_field: test
optional_number: 10
"""

print("\n[테스트 3] 필수 필드 누락 (required_field, required_number 없음):")
try:
    result = parser.parse(test3)
    print("❌ 이상함: 에러가 나야 하는데 성공했습니다!")
except ValidationError as e:
    print("✅ 예상대로 ValidationError 발생:")
    for err in e.errors():
        print(f"  - {err['loc']}: {err['msg']}")
except Exception as e:
    print(f"✅ 예상대로 에러 발생: {e}")

print("\n" + "=" * 80)
print("결론:")
print("=" * 80)
print("- Field(...) = 필수 → AI가 반드시 출력해야 함")
print("- Field(default=값) = 선택 → AI가 안 써도 기본값 사용")
print("- Field(default_factory=함수) = 선택 → AI가 안 써도 기본값 생성")
print("- 타입 | None + Field(None) = 선택 → AI가 안 써도 None")
