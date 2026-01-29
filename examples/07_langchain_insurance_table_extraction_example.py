from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from decimal import Decimal
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator

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


# 보험 상품 기본 정보
class InsuranceProduct(BaseModel):
    """보험 상품 기본 정보."""

    product_name: str = Field(..., description="보험 상품명")
    product_code: str = Field(..., description="상품 코드")
    product_type: str = Field(..., description="상품 유형: '생명보험', '손해보험', '건강보험', '연금보험'")
    is_active: bool = Field(..., description="현재 판매 중인지 여부")
    launch_date: str = Field(..., description="출시일 (YYYY-MM-DD 형식)")


# 보험료 정보
class PremiumInfo(BaseModel):
    """보험료 정보."""

    base_premium: float = Field(..., description="기본 보험료 (원)")
    age_factor: float = Field(..., description="나이 계수 (1.0 기준)")
    gender_factor: float = Field(..., description="성별 계수 (1.0 기준)")
    occupation_factor: float = Field(..., description="직업 계수 (1.0 기준)")
    calculated_premium: float = Field(..., description="계산된 총 보험료 (base_premium * age_factor * gender_factor * occupation_factor)")
    discount_rate: float = Field(default=0.0, description="할인율 (%)")
    final_premium: float = Field(..., description="최종 보험료 (할인 적용 후)")


# 보장 내용
class CoverageDetail(BaseModel):
    """보장 내용 상세."""

    coverage_type: str = Field(..., description="보장 유형: '질병', '사고', '사망', '상해', '입원' 등")
    coverage_amount: float = Field(..., description="보장 금액 (원)")
    coverage_period: int = Field(..., description="보장 기간 (개월)")
    deductible: float = Field(default=0.0, description="자기부담금 (원)")
    coverage_rate: float = Field(..., description="보장률 (%)")


# 가입 조건
class EligibilityCriteria(BaseModel):
    """가입 조건."""

    min_age: int = Field(..., description="최소 가입 나이")
    max_age: int = Field(..., description="최대 가입 나이")
    allowed_genders: list[str] = Field(..., description="가입 가능한 성별 리스트: ['남', '여'] 또는 ['남', '여']")
    restricted_occupations: list[str] = Field(default_factory=list, description="제한 직업 리스트")
    health_check_required: bool = Field(..., description="건강검진 필수 여부")
    max_coverage_amount: float = Field(..., description="최대 가입 한도 (원)")


# 보험금 지급 정보
class ClaimInfo(BaseModel):
    """보험금 지급 정보."""

    claim_type: str = Field(..., description="지급 유형: '일시금', '분할지급', '연금'")
    claim_amount: float = Field(..., description="지급 금액 (원)")
    payment_period: int = Field(default=0, description="지급 기간 (개월, 0이면 일시금)")
    waiting_period: int = Field(default=0, description="대기 기간 (일)")
    claim_conditions: list[str] = Field(..., description="지급 조건 리스트")


# 청크별 추출 결과
class ChunkExtractionResult(BaseModel):
    """청크별 정보 추출 결과."""

    chunk_index: int = Field(..., description="청크 인덱스 (0부터 시작)")
    chunk_type: str = Field(
        ..., description="청크 유형: 'product_info', 'premium_table', 'coverage_table', 'eligibility_table', 'claim_table'"
    )
    extracted_data: dict[str, Any] = Field(..., description="추출된 데이터 (타입에 따라 다름)")
    calculations: dict[str, float] = Field(default_factory=dict, description="수행된 계산 결과 (키: 계산명, 값: 결과값)")
    confidence: float = Field(..., description="추출 신뢰도 (0.0 ~ 1.0)")


# 전체 문서 추출 결과
class InsuranceDocumentExtraction(BaseModel):
    """보험사 테이블 문서 전체 추출 결과."""

    document_id: str = Field(..., description="문서 ID")
    extraction_date: str = Field(..., description="추출 일시 (YYYY-MM-DD HH:MM:SS)")
    products: list[InsuranceProduct] = Field(..., description="추출된 보험 상품 리스트")
    premium_info: list[PremiumInfo] = Field(default_factory=list, description="보험료 정보 리스트")
    coverage_details: list[CoverageDetail] = Field(default_factory=list, description="보장 내용 리스트")
    eligibility_criteria: list[EligibilityCriteria] = Field(default_factory=list, description="가입 조건 리스트")
    claim_info: list[ClaimInfo] = Field(default_factory=list, description="보험금 지급 정보 리스트")
    chunk_results: list[ChunkExtractionResult] = Field(..., description="청크별 추출 결과 리스트")
    total_premium_calculated: float = Field(default=0.0, description="전체 계산된 보험료 합계 (원)")
    average_coverage_amount: float = Field(default=0.0, description="평균 보장 금액 (원)")
    summary: str = Field(..., description="추출 결과 요약")


def extract_insurance_info_from_chunk(chunk_text: str, chunk_index: int) -> ChunkExtractionResult:
    """청크 텍스트에서 보험 정보를 추출합니다.

    Args:
        chunk_text: 추출할 청크 텍스트
        chunk_index: 청크 인덱스

    Returns:
        ChunkExtractionResult: 추출 결과
    """
    cfg = ParserConfig(instructions_mode="minimal")
    parser = ToonOutputParser(model=ChunkExtractionResult, cfg=cfg)

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
    )

    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 보험사 문서에서 테이블 정보를 정확하게 추출하고 필요한 계산을 수행하는 전문가입니다. "
                "테이블에서 숫자, 날짜, 불린 값, 리스트 등 다양한 타입의 데이터를 추출하고, "
                "보험료 계산, 할인율 계산, 보장률 계산 등을 수행해야 합니다.",
            ),
            (
                "human",
                """다음 청크에서 보험 관련 정보를 추출하고 필요한 계산을 수행해주세요.

청크 인덱스: {chunk_index}
청크 텍스트:
{chunk_text}

⚠️ CRITICAL: 다음 필수 필드를 반드시 모두 포함해야 합니다:
- chunk_index: {chunk_index} (이 값을 그대로 사용)
- chunk_type: 청크 유형 문자열
- extracted_data: 추출된 데이터 딕셔너리
- confidence: 신뢰도 숫자 (0.0~1.0)
- calculations: 계산 결과 딕셔너리 (없으면 빈 딕셔너리 {{}})

추출 및 계산 지침:
1. chunk_index: 반드시 {chunk_index} 값을 그대로 출력
2. chunk_type: 청크의 유형을 판단하세요
   - "product_info": 상품 기본 정보 테이블
   - "premium_table": 보험료 테이블 (계산 필요)
   - "coverage_table": 보장 내용 테이블
   - "eligibility_table": 가입 조건 테이블
   - "claim_table": 보험금 지급 테이블

3. extracted_data: 테이블에서 추출한 모든 데이터를 딕셔너리 형태로 저장
   - 숫자: float 또는 int로 변환
   - 날짜: "YYYY-MM-DD" 형식 문자열
   - 불린: true/false
   - 리스트: 배열 형태
   - 문자열: 그대로 저장

4. calculations: 필요한 계산을 수행하고 결과를 저장 (계산이 없으면 빈 딕셔너리 {{}})
   - 보험료 계산: base_premium * age_factor * gender_factor * occupation_factor
   - 할인 적용: calculated_premium * (1 - discount_rate / 100)
   - 보장률 계산: (coverage_amount / max_coverage_amount) * 100
   - 총액 계산: 여러 항목의 합계
   - 평균 계산: 여러 값의 평균
   - 계산 키 예시: "total_premium", "discounted_amount", "coverage_rate", "average_age" 등

5. confidence: 추출 신뢰도 (0.0 ~ 1.0) - 반드시 포함
   - 테이블이 명확하고 완전하면 0.9 이상
   - 일부 정보가 불명확하면 0.7 ~ 0.8
   - 정보가 부족하거나 모호하면 0.5 이하

주의사항:
- 숫자는 반드시 숫자 타입으로 변환하세요 (문자열이 아닌)
- 계산은 정확하게 수행하세요
- extracted_data에는 원본 데이터와 계산에 필요한 모든 정보를 포함하세요
- calculations에는 계산된 결과만 포함하세요
- TOON 형식의 들여쓰기를 정확하게 지켜주세요

출력 예시 (반드시 이 형식을 따르세요):
```toon
chunk_index: {chunk_index}
chunk_type: coverage_table
extracted_data:
  field1: value1
  field2: value2
calculations:
  total: 100
  average: 50
confidence: 0.9
```

{format_instructions}""",
            ),
        ]
    )

    llm_chain = prompt | llm | StrOutputParser()

    prompt_vars = {
        "chunk_index": chunk_index,
        "chunk_text": chunk_text,
        "format_instructions": format_instructions,
    }

    raw_output = llm_chain.invoke(prompt_vars)

    try:
        result = parser.parse(raw_output)
    except Exception as e:
        return raw_output, None, str(e)

    return raw_output, result, None


def extract_full_document(chunks: list[str], document_id: str) -> InsuranceDocumentExtraction:
    """전체 문서 청크들에서 보험 정보를 추출합니다.

    Args:
        chunks: 문서를 나눈 청크 리스트
        document_id: 문서 ID

    Returns:
        InsuranceDocumentExtraction: 전체 추출 결과
    """
    cfg = ParserConfig(instructions_mode="minimal")
    parser = ToonOutputParser(model=InsuranceDocumentExtraction, cfg=cfg)

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
    )

    # 각 청크에서 정보 추출
    chunk_results = []
    all_premiums = []
    all_coverage_amounts = []

    for idx, chunk in enumerate(chunks):
        print(f"\n[청크 {idx} 처리 중...]")
        raw_output, chunk_result, parse_error = extract_insurance_info_from_chunk(chunk, idx)

        if parse_error:
            print(f"⚠️ 청크 {idx} 파싱 에러: {parse_error}")
            continue

        if chunk_result:
            chunk_results.append(chunk_result)

            # 계산 결과에서 보험료 수집
            if "total_premium" in chunk_result.calculations:
                all_premiums.append(chunk_result.calculations["total_premium"])
            elif "final_premium" in chunk_result.calculations:
                all_premiums.append(chunk_result.calculations["final_premium"])

            # 보장 금액 수집
            if "coverage_amount" in chunk_result.extracted_data:
                coverage_amt = chunk_result.extracted_data["coverage_amount"]
                if isinstance(coverage_amt, (int, float)):
                    all_coverage_amounts.append(float(coverage_amt))

    # 전체 문서 추출 결과 생성
    format_instructions = parser.get_format_instructions()

    # 청크 결과 요약 텍스트 생성
    chunk_summary = "\n".join(
        [
            f"청크 {cr.chunk_index} ({cr.chunk_type}): 신뢰도 {cr.confidence:.2f}, 계산 {len(cr.calculations)}개"
            for cr in chunk_results
        ]
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 보험사 문서에서 추출된 정보를 통합하고 요약하는 전문가입니다. "
                "여러 청크에서 추출된 정보를 종합하여 구조화된 데이터로 변환해야 합니다.",
            ),
            (
                "human",
                """다음 청크별 추출 결과를 종합하여 전체 문서 추출 결과를 생성해주세요.

문서 ID: {document_id}
추출 일시: {extraction_date}

청크별 추출 결과:
{chunk_summary}

청크별 상세 데이터:
{chunk_results_json}

계산된 통계:
- 전체 보험료 합계: {total_premium_sum:.2f} 원
- 평균 보장 금액: {average_coverage:.2f} 원

지침:
1. products: 상품 정보를 추출하여 InsuranceProduct 리스트로 구성
2. premium_info: 보험료 정보를 추출하여 PremiumInfo 리스트로 구성 (계산된 보험료 포함)
3. coverage_details: 보장 내용을 추출하여 CoverageDetail 리스트로 구성
4. eligibility_criteria: 가입 조건을 추출하여 EligibilityCriteria 리스트로 구성
5. claim_info: 보험금 지급 정보를 추출하여 ClaimInfo 리스트로 구성
6. chunk_results: 위에서 추출한 청크 결과 리스트를 그대로 포함
7. total_premium_calculated: 모든 보험료의 합계
8. average_coverage_amount: 모든 보장 금액의 평균
9. summary: 추출된 정보의 요약 (2-3문장)

주의사항:
- 각 필드는 적절한 타입으로 변환하세요
- 리스트는 비어있을 수 있습니다
- 계산된 값들은 정확하게 포함하세요
- TOON 형식의 들여쓰기를 정확하게 지켜주세요

{format_instructions}""",
            ),
        ]
    )

    total_premium_sum = sum(all_premiums) if all_premiums else 0.0
    average_coverage = sum(all_coverage_amounts) / len(all_coverage_amounts) if all_coverage_amounts else 0.0

    chunk_results_json = json.dumps([cr.model_dump() for cr in chunk_results], ensure_ascii=False, indent=2)

    llm_chain = prompt | llm | StrOutputParser()

    prompt_vars = {
        "document_id": document_id,
        "extraction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "chunk_summary": chunk_summary,
        "chunk_results_json": chunk_results_json,
        "total_premium_sum": total_premium_sum,
        "average_coverage": average_coverage,
        "format_instructions": format_instructions,
    }

    raw_output = llm_chain.invoke(prompt_vars)

    try:
        result = parser.parse(raw_output)
    except Exception as e:
        return raw_output, None, str(e)

    return raw_output, result, None


def main() -> None:
    """테스트용 메인 함수 (보험사 테이블 문서 추출 예시)."""
    # 샘플 보험사 테이블 문서 청크들
    sample_chunks = [
        """보험 상품 기본 정보 테이블
=====================================
상품명: 실손의료비보험 플러스
상품코드: MED-2024-001
상품유형: 건강보험
판매상태: 판매중
출시일: 2024-01-15
""",
        """보험료 계산 테이블
=====================================
기본 보험료: 50,000원
나이 계수:
  - 20-29세: 0.8
  - 30-39세: 1.0
  - 40-49세: 1.2
  - 50-59세: 1.5
성별 계수:
  - 남성: 1.1
  - 여성: 1.0
직업 계수:
  - 사무직: 1.0
  - 현장직: 1.3
  - 전문직: 0.9
할인율: 10% (온라인 가입 시)

예시 계산 (30세 남성 사무직):
기본 보험료: 50,000원
나이 계수: 1.0
성별 계수: 1.1
직업 계수: 1.0
계산된 보험료: 50,000 * 1.0 * 1.1 * 1.0 = 55,000원
할인 적용: 55,000 * (1 - 0.1) = 49,500원
최종 보험료: 49,500원
""",
        """보장 내용 테이블
=====================================
보장 항목:
1. 질병 입원: 보장금액 1,000만원, 보장기간 12개월, 자기부담금 10만원, 보장률 90%
2. 사고 입원: 보장금액 2,000만원, 보장기간 12개월, 자기부담금 5만원, 보장률 95%
3. 수술비: 보장금액 500만원, 보장기간 12개월, 자기부담금 0원, 보장률 100%
4. 통원치료: 보장금액 300만원, 보장기간 12개월, 자기부담금 3만원, 보장률 90%
""",
        """가입 조건 테이블
=====================================
최소 가입 나이: 20세
최대 가입 나이: 65세
가입 가능 성별: 남, 여
제한 직업: 없음
건강검진 필수: 아니오
최대 가입 한도: 5,000만원
""",
        """보험금 지급 정보 테이블
=====================================
지급 유형: 일시금
지급 금액: 실제 발생한 의료비 (보장 한도 내)
지급 기간: 0개월 (일시금)
대기 기간: 30일
지급 조건:
  - 입원일로부터 30일 경과 후
  - 의료비 영수증 제출 필수
  - 보장 항목에 해당하는 경우
  - 자기부담금 차감 후 지급
""",
    ]

    print("=" * 80)
    print("보험사 테이블 문서 정보 추출 예시")
    print("=" * 80)

    # 각 청크별 추출 테스트
    print("\n" + "=" * 80)
    print("1단계: 청크별 정보 추출")
    print("=" * 80)

    chunk_extraction_results = []
    for idx, chunk in enumerate(sample_chunks):
        print(f"\n{'=' * 80}")
        print(f"청크 {idx} 추출 중...")
        print(f"{'=' * 80}")
        print(f"\n청크 내용:\n{chunk}")

        try:
            raw_output, result, parse_error = extract_insurance_info_from_chunk(chunk, idx)

            print("\n" + "=" * 80)
            print(f"청크 {idx} - LLM 원본 출력:")
            print("=" * 80)
            print("```toon")
            print(raw_output)
            print("```")

            if parse_error:
                print(f"\n⚠️ 파싱 에러: {parse_error}")
                continue

            if result:
                chunk_extraction_results.append(result)
                print("\n" + "=" * 80)
                print(f"청크 {idx} - 추출 결과:")
                print("=" * 80)
                print(f"청크 유형: {result.chunk_type}")
                print(f"신뢰도: {result.confidence:.2f}")
                print(f"\n추출된 데이터:")
                print(json.dumps(result.extracted_data, ensure_ascii=False, indent=2))
                print(f"\n수행된 계산:")
                print(json.dumps(result.calculations, ensure_ascii=False, indent=2))
                
                # 비용 분석
                print("\n" + "=" * 80)
                print(f"청크 {idx} - 비용 분석:")
                print("=" * 80)
                cfg = ParserConfig(instructions_mode="minimal")
                analysis = CostAnalyzer.analyze_actual_usage(
                    model=ChunkExtractionResult,
                    toon_raw_output=raw_output,
                    parsed_result=result,
                    cfg=cfg,
                )
                CostAnalyzer.print_actual_usage_analysis(analysis)

        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            import traceback

            traceback.print_exc()
            continue

    # 전체 문서 통합 추출
    if chunk_extraction_results:
        print("\n\n" + "=" * 80)
        print("2단계: 전체 문서 통합 추출")
        print("=" * 80)

        try:
            raw_output, result, parse_error = extract_full_document(sample_chunks, "DOC-2024-001")

            print("\n" + "=" * 80)
            print("전체 문서 - LLM 원본 출력:")
            print("=" * 80)
            print("```toon")
            # 긴 출력을 청크로 나눠서 출력
            chunk_size = 2000
            for i in range(0, len(raw_output), chunk_size):
                chunk = raw_output[i : i + chunk_size]
                print(chunk, end="", flush=True)
            print()
            print("```")

            if parse_error:
                print(f"\n⚠️ 파싱 에러: {parse_error}")
            elif result:
                print("\n" + "=" * 80)
                print("전체 문서 - 최종 추출 결과:")
                print("=" * 80)
                print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))

                print("\n" + "=" * 80)
                print("요약:")
                print("=" * 80)
                print(f"문서 ID: {result.document_id}")
                print(f"추출 일시: {result.extraction_date}")
                print(f"추출된 상품 수: {len(result.products)}")
                print(f"보험료 정보 수: {len(result.premium_info)}")
                print(f"보장 내용 수: {len(result.coverage_details)}")
                print(f"가입 조건 수: {len(result.eligibility_criteria)}")
                print(f"보험금 지급 정보 수: {len(result.claim_info)}")
                print(f"청크 결과 수: {len(result.chunk_results)}")
                print(f"전체 보험료 합계: {result.total_premium_calculated:,.0f}원")
                print(f"평균 보장 금액: {result.average_coverage_amount:,.0f}원")
                print(f"\n요약: {result.summary}")
                
                # 전체 문서 비용 분석
                print("\n" + "=" * 80)
                print("전체 문서 - 비용 분석:")
                print("=" * 80)
                cfg = ParserConfig(instructions_mode="minimal")
                analysis = CostAnalyzer.analyze_actual_usage(
                    model=InsuranceDocumentExtraction,
                    toon_raw_output=raw_output,
                    parsed_result=result,
                    cfg=cfg,
                )
                CostAnalyzer.print_actual_usage_analysis(analysis)

        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print("모든 테스트 완료")
    print("=" * 80)


if __name__ == "__main__":
    main()
