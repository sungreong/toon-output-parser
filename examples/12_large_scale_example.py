from __future__ import annotations

import os
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
            "langchain-openai or langchain-community is required. Install: pip install langchain-openai"
        ) from None

from toon_langchain_parser import ToonOutputParser


class ProductItem(BaseModel):
    id: int = Field(..., description="Product ID")
    name: str = Field(..., description="Product name")
    price: float | None = Field(None, description="Product price")
    category: str = Field(..., description="Product category")
    in_stock: bool | None = Field(None, description="Whether product is in stock")


class ProductCatalog(BaseModel):
    total_items: int = Field(..., description="Total product count")
    products: list[ProductItem] = Field(..., description="Product list")


def extract_large_catalog(document: str):
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
                "Extract a large product catalog from text.\n"
                "Format rules:\n"
                "- Return TOON only.\n"
                "- Use only schema keys (total_items, products, and product fields).\n"
                "- total_items must equal len(products).\n"
                "Quality rules:\n"
                "- Do not guess values.\n"
                "- If unknown, use null.\n"
                "- If evidence exists in the text, fill the value.",
            ),
            (
                "human",
                "Extract product catalog from this document.\n\n"
                "Document:\n{document}\n\n"
                "Output requirements:\n"
                "1) total_items: total number of products\n"
                "2) products: list of products with fields\n"
                "   - id\n"
                "   - name\n"
                "   - price\n"
                "   - category\n"
                "   - in_stock\n\n"
                "Additional constraints:\n"
                "- Keep products in dash-list notation.\n"
                "- For unknown price/in_stock, use null.\n"
                "- Keep total_items consistent with products length.\n\n"
                "{format_instructions}",
            ),
        ]
    )

    llm_chain = prompt | llm | StrOutputParser()

    start_time = time.time()
    raw_output = llm_chain.invoke(
        {
            "document": document,
            "format_instructions": format_instructions,
        }
    )
    llm_time = time.time() - start_time

    parse_start = time.time()
    try:
        result = parser.parse(raw_output)
        parse_time = time.time() - parse_start
        return raw_output, result, None, llm_time, parse_time
    except Exception as e:
        parse_time = time.time() - parse_start
        return raw_output, None, str(e), llm_time, parse_time


def build_document(size: int) -> str:
    return (
        f"Product catalog report with {size} items. "
        "Some products have missing price or stock status in source text. "
        "Extract what is explicit and set unknown fields to null."
    )


def main() -> None:
    print("=" * 80)
    print("Large Scale Example")
    print("=" * 80)

    test_sizes = [10, 50, 100]

    for size in test_sizes:
        print(f"\n{'=' * 80}")
        print(f"Test size: {size} items")
        print("=" * 80)

        document = build_document(size)
        print(f"\nInput length: {len(document)} chars")
        print(f"Expected product count: {size}")

        raw_output, result, parse_error, llm_time, parse_time = extract_large_catalog(document)

        if parse_error:
            print(f"\nParse error: {parse_error}")
            print(f"LLM time: {llm_time:.2f}s")
            print(f"Parse time: {parse_time:.2f}s")
            continue

        assert result is not None
        print("\n" + "=" * 80)
        print(f"Result ({size} items)")
        print("=" * 80)
        print(f"total_items: {result.total_items}")
        print(f"parsed products: {len(result.products)}")
        print(f"LLM time: {llm_time:.2f}s")
        print(f"Parse time: {parse_time:.2f}s")
        print(f"Total time: {llm_time + parse_time:.2f}s")

        if result.products:
            print("\nFirst 3 products:")
            for i, product in enumerate(result.products[:3], 1):
                price_text = f"{product.price:,.0f}" if product.price is not None else "null"
                print(f"  {i}. {product.name} (id={product.id}, price={price_text})")

            if len(result.products) > 3:
                print(f"  ... and {len(result.products) - 3} more")

        categories: dict[str, int] = {}
        in_stock_count = 0
        out_of_stock_count = 0
        unknown_stock_count = 0
        unknown_price_count = 0

        for product in result.products:
            categories[product.category] = categories.get(product.category, 0) + 1

            if product.price is None:
                unknown_price_count += 1

            if product.in_stock is True:
                in_stock_count += 1
            elif product.in_stock is False:
                out_of_stock_count += 1
            else:
                unknown_stock_count += 1

        print("\nStats:")
        print("  Category distribution:")
        for cat, count in sorted(categories.items()):
            print(f"    - {cat}: {count}")
        print(f"  In stock: {in_stock_count}")
        print(f"  Out of stock: {out_of_stock_count}")
        print(f"  Unknown stock: {unknown_stock_count}")
        print(f"  Unknown price: {unknown_price_count}")

        items_per_second = len(result.products) / parse_time if parse_time > 0 else 0
        print("\nPerformance:")
        print(f"  Parse speed: {items_per_second:.1f} items/sec")
        print(f"  Raw TOON length: {len(raw_output):,} chars")

    print("\n" + "=" * 80)
    print("All tests completed")
    print("=" * 80)


if __name__ == "__main__":
    main()
