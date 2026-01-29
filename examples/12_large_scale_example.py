from __future__ import annotations

import json
import os
import sys
import time

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

from toon_langchain_parser import ToonOutputParser


class ProductItem(BaseModel):
    """제품 항목."""

    id: int = Field(..., description="제품 ID")
    name: str = Field(..., description="제품명")
    price: float = Field(..., description="가격")
    category: str = Field(..., description="카테고리")
    in_stock: bool = Field(..., description="재고 여부")


class ProductCatalog(BaseModel):
    """대규모 제품 카탈로그."""

    total_items: int = Field(..., description="총 제품 수")
    products: list[ProductItem] = Field(..., description="제품 리스트 (대량)")


def generate_large_catalog(num_items: int = 100) -> ProductCatalog:
    """대규모 카탈로그를 생성합니다 (실제 LLM 호출 없이 테스트용).

    Args:
        num_items: 생성할 제품 수

    Returns:
        ProductCatalog: 생성된 카탈로그
    """
    products = []
    categories = ["전자제품", "의류", "식품", "도서", "스포츠"]
    
    for i in range(num_items):
        products.append(
            ProductItem(
                id=i + 1,
                name=f"제품 {i + 1}",
                price=float((i + 1) * 1000),
                category=categories[i % len(categories)],
                in_stock=(i % 3 != 0),
            )
        )
    
    return ProductCatalog(total_items=num_items, products=products)


def extract_large_catalog(document: str) -> ProductCatalog:
    """문서에서 대규모 카탈로그를 추출합니다.

    Args:
        document: 제품 목록이 포함된 문서

    Returns:
        ProductCatalog: 추출된 카탈로그
    """
    parser = ToonOutputParser(model=ProductCatalog)

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
    )

    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 문서에서 대규모 제품 카탈로그를 정확하게 추출하는 전문가입니다. "
                "100개 이상의 항목을 효율적으로 처리해야 합니다.",
            ),
            (
                "human",
                """다음 문서에서 제품 카탈로그를 추출해주세요.

문서:
{document}

추출해야 할 정보:
1. total_items: 총 제품 수
2. products: 제품 리스트 (각 제품마다):
   - id: 제품 ID
   - name: 제품명
   - price: 가격
   - category: 카테고리
   - in_stock: 재고 여부 (true/false)

⚠️ CRITICAL: 대량 데이터 처리 규칙
- 모든 제품을 빠짐없이 포함하세요
- 배열은 dash list 형식을 사용하세요
- 숫자는 천 단위 구분자 없이 출력하세요
- 효율적으로 구조화하여 출력하세요

주의사항:
- total_items는 실제 products 배열의 길이와 일치해야 합니다
- 모든 필수 필드를 포함하세요
- TOON 형식의 들여쓰기를 정확하게 지켜주세요

{format_instructions}""",
            ),
        ]
    )

    llm_chain = prompt | llm | StrOutputParser()

    start_time = time.time()
    raw_output = llm_chain.invoke({"document": document, "format_instructions": format_instructions})
    llm_time = time.time() - start_time

    parse_start = time.time()
    try:
        result = parser.parse(raw_output)
        parse_time = time.time() - parse_start
        return raw_output, result, None, llm_time, parse_time
    except Exception as e:
        parse_time = time.time() - parse_start
        return raw_output, None, str(e), llm_time, parse_time


def main() -> None:
    """테스트용 메인 함수 (대규모 데이터 예시)."""
    print("=" * 80)
    print("대규모 데이터 처리 예시")
    print("=" * 80)

    # 테스트 케이스: 다양한 크기
    test_sizes = [10, 50, 100]

    for size in test_sizes:
        print(f"\n{'=' * 80}")
        print(f"테스트: {size}개 항목")
        print("=" * 80)

        # 테스트용 데이터 생성
        catalog = generate_large_catalog(size)
        
        # 문서 생성 (간단한 요약)
        document = f"""
        제품 카탈로그:
        총 {size}개의 제품이 있습니다.
        
        카테고리별 분류:
        - 전자제품: {size // 5}개
        - 의류: {size // 5}개
        - 식품: {size // 5}개
        - 도서: {size // 5}개
        - 스포츠: {size // 5}개
        
        각 제품은 ID, 이름, 가격, 카테고리, 재고 여부 정보를 가집니다.
        """

        print(f"\n입력 문서 (요약): {len(document)} 문자")
        print(f"예상 제품 수: {size}개")

        try:
            raw_output, result, parse_error, llm_time, parse_time = extract_large_catalog(document)

            if parse_error:
                print(f"\n⚠️ 파싱 에러: {parse_error}")
                print(f"LLM 시간: {llm_time:.2f}초")
                print(f"파싱 시간: {parse_time:.2f}초")
                continue

            if result:
                print("\n" + "=" * 80)
                print(f"결과 ({size}개 항목):")
                print("=" * 80)
                print(f"총 제품 수: {result.total_items}")
                print(f"실제 파싱된 제품 수: {len(result.products)}")
                print(f"LLM 생성 시간: {llm_time:.2f}초")
                print(f"파싱 시간: {parse_time:.2f}초")
                print(f"총 처리 시간: {llm_time + parse_time:.2f}초")
                
                if result.products:
                    print(f"\n처음 3개 제품:")
                    for i, product in enumerate(result.products[:3], 1):
                        print(f"  {i}. {product.name} (ID: {product.id}, 가격: {product.price:,.0f}원)")
                    
                    if len(result.products) > 3:
                        print(f"  ... 외 {len(result.products) - 3}개")
                
                # 통계
                print(f"\n통계:")
                categories = {}
                in_stock_count = 0
                for product in result.products:
                    categories[product.category] = categories.get(product.category, 0) + 1
                    if product.in_stock:
                        in_stock_count += 1
                
                print(f"  카테고리별 분포:")
                for cat, count in sorted(categories.items()):
                    print(f"    - {cat}: {count}개")
                print(f"  재고 있음: {in_stock_count}개")
                print(f"  재고 없음: {len(result.products) - in_stock_count}개")
                
                # 성능 평가
                items_per_second = len(result.products) / parse_time if parse_time > 0 else 0
                print(f"\n성능:")
                print(f"  파싱 속도: {items_per_second:.1f} 항목/초")
                print(f"  메모리 효율: TOON 길이 {len(raw_output):,} 문자")

        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("모든 테스트 완료")
    print("=" * 80)
    print("\n참고: 실제 LLM 호출은 비용이 발생할 수 있습니다.")
    print("대규모 데이터는 청크 단위로 나누어 처리하는 것을 권장합니다.")


if __name__ == "__main__":
    main()
