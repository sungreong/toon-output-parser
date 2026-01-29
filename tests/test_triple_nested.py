"""3단계 중첩 배열 파싱 테스트 - 실제 에러 재현"""
from pydantic import BaseModel, Field
from src.toon_langchain_parser import ToonOutputParser

class PersonTrait(BaseModel):
    name: str = Field(..., description="특성 이름")
    level: str = Field(default="", description="특성 수준")
    description: str = Field(default="", description="특성 설명")

class Person(BaseModel):
    name: str = Field(..., description="이름")
    age: int | None = Field(None, description="나이")
    role: str = Field(default="", description="역할")
    traits: list[PersonTrait] = Field(default_factory=list, description="특성 리스트")
    skills: list[str] = Field(default_factory=list, description="기술 리스트")
    background: str = Field(default="", description="배경")

class Team(BaseModel):
    team_name: str = Field(..., description="팀 이름")
    total_members: int = Field(..., description="총 인원")
    members: list[Person] = Field(default_factory=list, description="팀원 리스트")

# 실제 에러가 발생한 TOON 구조 (간소화)
test_toon = """
team_name: 그리핀도르 퀴디치 팀
total_members: 2
members:
  - name: 올리버 우드
    age: 17
    role: 골키퍼
    traits:
      - name: 책임감
        level: 높음
        description: 팀원들을 잘 이끄는 리더십
      - name: 반사신경
        level: 뛰어남
        description: 골키퍼로서의 뛰어난 반사신경
    skills: []
    background: 퀴디치 전략 수립
  - name: 해리 포터
    age: 14
    role: 수색꾼
    traits:
      - name: 용감함
        level: 매우 높음
        description: 침착함 유지
    skills: []
    background: 스니치 찾기 탁월
"""

parser = ToonOutputParser(model=Team)

print("=" * 80)
print("3단계 중첩 배열 파싱 테스트 (members[] -> traits[])")
print("=" * 80)
print("\n입력 TOON:")
print(test_toon)
print("\n구조: Team -> members[Person] -> traits[PersonTrait]")
print("\n파싱 시도 중...")

try:
    result = parser.parse(test_toon)
    print("\n✅ 파싱 성공!")
    print(f"팀명: {result.team_name}")
    print(f"인원: {result.total_members}")
    print(f"실제 파싱된 멤버 수: {len(result.members)}")
    for idx, member in enumerate(result.members):
        print(f"\n[{idx}] {member.name} ({member.role})")
        print(f"    Traits: {len(member.traits)}개")
        for trait in member.traits:
            print(f"      - {trait.name}: {trait.level}")
except Exception as e:
    print(f"\n❌ 파싱 실패!")
    print(f"에러: {e}")
    import traceback
    print("\n상세 스택:")
    traceback.print_exc()
