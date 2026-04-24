# -*- coding: utf-8 -*-
"""
프롬프트 비용 분석 도구

JSON Structured Output vs TOON 포맷의 입력/출력 길이를 비교하여
프롬프트 사용에 따른 비용을 측정합니다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel

from .toon_parser_ultimate import ParserConfig, ToonParser


@dataclass
class PromptCostMetrics:
    """프롬프트 비용 메트릭."""
    
    # 입력 관련
    format_instructions_length: int
    format_instructions_lines: int
    
    # 출력 관련 (실제 데이터로 측정)
    avg_output_length: int = 0
    avg_output_lines: int = 0
    format_instructions_tokens: int | None = None
    avg_output_tokens: int | None = None
    
    # 비율
    output_to_data_ratio: float = 0.0  # 출력 길이 / 실제 데이터 크기
    
    def __str__(self) -> str:
        return (
            f"PromptCostMetrics(\n"
            f"  Input: {self.format_instructions_length} rough chars, "
            f"{self.format_instructions_lines} lines\n"
            f"  Output: {self.avg_output_length} rough chars, "
            f"{self.avg_output_lines} lines\n"
            f"  Ratio: {self.output_to_data_ratio:.2f}x\n"
            f")"
        )


@dataclass
class FormatComparison:
    """포맷 비교 결과."""
    
    json_metrics: PromptCostMetrics
    toon_metrics: PromptCostMetrics
    
    # 비교 지표
    input_reduction_percent: float  # 입력 길이 감소율
    output_reduction_percent: float  # 출력 길이 감소율
    total_reduction_percent: float  # 전체 감소율 (입력+출력)
    
    def print_comparison(self) -> None:
        """비교 결과를 보기 좋게 출력합니다."""
        print("=" * 80)
        print("포맷 비용 비교 분석")
        print("=" * 80)
        print()
        
        print("📊 JSON Structured Output:")
        print(f"  입력 (format instructions): {self.json_metrics.format_instructions_length:,} rough chars "
              f"({self.json_metrics.format_instructions_lines} lines)")
        print(f"  출력 (avg): {self.json_metrics.avg_output_length:,} rough chars "
              f"({self.json_metrics.avg_output_lines} lines)")
        print(f"  출력/데이터 비율: {self.json_metrics.output_to_data_ratio:.2f}x")
        json_total = self.json_metrics.format_instructions_length + self.json_metrics.avg_output_length
        print(f"  총합: {json_total:,} rough chars")
        print()
        
        print("📊 TOON Format:")
        print(f"  입력 (format instructions): {self.toon_metrics.format_instructions_length:,} rough chars "
              f"({self.toon_metrics.format_instructions_lines} lines)")
        print(f"  출력 (avg): {self.toon_metrics.avg_output_length:,} rough chars "
              f"({self.toon_metrics.avg_output_lines} lines)")
        print(f"  출력/데이터 비율: {self.toon_metrics.output_to_data_ratio:.2f}x")
        toon_total = self.toon_metrics.format_instructions_length + self.toon_metrics.avg_output_length
        print(f"  총합: {toon_total:,} rough chars")
        print()
        
        print("💰 비용 절감 분석:")
        print(f"  입력 길이 감소: {self.input_reduction_percent:+.1f}% "
              f"({'절감' if self.input_reduction_percent < 0 else '증가'})")
        print(f"  출력 길이 감소: {self.output_reduction_percent:+.1f}% "
              f"({'절감' if self.output_reduction_percent < 0 else '증가'})")
        print(f"  전체 감소: {self.total_reduction_percent:+.1f}% "
              f"({'절감' if self.total_reduction_percent < 0 else '증가'})")
        print()
        
        # 시각적 표현
        if self.total_reduction_percent < 0:
            saved = abs(json_total - toon_total)
            print(f"✅ TOON 사용 시 요청당 약 {saved:,} rough chars 절감!")
            print(f"   (1000회 호출 시: {saved * 1000:,} rough chars 절감)")
        else:
            print(f"⚠️ TOON 사용 시 요청당 약 {abs(json_total - toon_total):,} rough chars 증가")
        
        print()
        print("=" * 80)


class CostAnalyzer:
    """프롬프트 비용 분석기."""

    @staticmethod
    def _count_tokens(text: str) -> int | None:
        try:
            import tiktoken
        except Exception:
            return None
        try:
            encoding = tiktoken.get_encoding("o200k_base")
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    
    @staticmethod
    def analyze_actual_usage(
        model: Type[BaseModel],
        toon_raw_output: str,
        parsed_result: BaseModel,
        cfg: Optional[ParserConfig] = None,
    ) -> Dict[str, Any]:
        """실제 사용된 입력/출력을 분석합니다.
        
        사용자가 실제로 LLM을 호출한 후, TOON 출력과 파싱 결과를 받아서
        JSON Structured Output을 사용했을 때와 비교합니다.
        
        Args:
            model: 사용한 Pydantic 모델
            toon_raw_output: LLM이 출력한 원본 TOON 문자열
            parsed_result: 파싱된 Pydantic 객체
            cfg: 파서 설정
            
        Returns:
            Dict: 분석 결과
            
        Example:
            >>> raw_output, result = extract_character_info(document)
            >>> analysis = CostAnalyzer.analyze_actual_usage(
            ...     model=CharacterFeatures,
            ...     toon_raw_output=raw_output,
            ...     parsed_result=result
            ... )
            >>> print(f"TOON 사용 시: {analysis['toon_total_chars']} chars")
            >>> print(f"JSON 사용 시: {analysis['json_total_chars']} chars")
            >>> print(f"절감: {analysis['chars_saved']} chars")
        """
        # 1. 입력 프롬프트 길이 측정
        # cfg가 None이면 기본값 사용, 있으면 사용
        actual_cfg = cfg or ParserConfig()
        toon_parser = ToonParser(model=model, cfg=actual_cfg)
        toon_instructions = toon_parser.get_format_instructions()
        toon_input_len = len(toon_instructions)
        
        # effective_mode 확인 (자동 폴백 여부)
        effective_mode = getattr(toon_parser, '_effective_mode', actual_cfg.instructions_mode)
        
        # JSON schema 기반 지시문
        schema = model.model_json_schema()
        json_schema_str = json.dumps(schema, indent=2, ensure_ascii=False)
        json_instructions = f"""Please respond with a JSON object that matches this schema:

{json_schema_str}

Important:
- Output ONLY valid JSON
- Follow the exact schema structure
- Include all required fields
"""
        json_input_len = len(json_instructions)
        toon_input_tokens = CostAnalyzer._count_tokens(toon_instructions)
        json_input_tokens = CostAnalyzer._count_tokens(json_instructions)
        
        # 2. 실제 출력 길이 측정
        toon_output_len = len(toon_raw_output.strip())
        toon_output_lines = len(toon_raw_output.strip().splitlines())
        
        # JSON 출력 시뮬레이션
        data = parsed_result.model_dump()
        json_output = json.dumps(data, indent=2, ensure_ascii=False)
        json_output_len = len(json_output)
        json_output_lines = len(json_output.splitlines())
        toon_output_tokens = CostAnalyzer._count_tokens(toon_raw_output.strip())
        json_output_tokens = CostAnalyzer._count_tokens(json_output)
        
        # 3. 총합 계산
        toon_total = toon_input_len + toon_output_len
        json_total = json_input_len + json_output_len
        chars_saved = json_total - toon_total
        
        # 4. 데이터 크기 (최소화된 JSON)
        data_size = len(json.dumps(data, ensure_ascii=False, separators=(',', ':')))
        
        return {
            # 입력 (format instructions)
            "toon_input_chars": toon_input_len,
            "json_input_chars": json_input_len,
            "toon_input_tokens": toon_input_tokens,
            "json_input_tokens": json_input_tokens,
            "input_diff": json_input_len - toon_input_len,
            "input_diff_percent": ((toon_input_len - json_input_len) / json_input_len * 100),
            "effective_mode": effective_mode,  # 실제 사용된 모드 (minimal/adaptive/json)
            
            # 출력 (actual output)
            "toon_output_chars": toon_output_len,
            "toon_output_lines": toon_output_lines,
            "json_output_chars": json_output_len,
            "json_output_lines": json_output_lines,
            "toon_output_tokens": toon_output_tokens,
            "json_output_tokens": json_output_tokens,
            "output_diff": json_output_len - toon_output_len,
            "output_diff_percent": ((toon_output_len - json_output_len) / json_output_len * 100),
            
            # 총합
            "toon_total_chars": toon_total,
            "json_total_chars": json_total,
            "chars_saved": chars_saved,
            "total_reduction_percent": (chars_saved / json_total * 100),
            
            # 추가 정보
            "data_size": data_size,
            "toon_overhead_ratio": toon_output_len / data_size,
            "json_overhead_ratio": json_output_len / data_size,
            
            # 원본 데이터
            "toon_raw": toon_raw_output.strip(),
            "json_equivalent": json_output,
        }
    
    @staticmethod
    def print_actual_usage_analysis(analysis: Dict[str, Any]) -> None:
        """실제 사용 분석 결과를 보기 좋게 출력합니다.
        
        Args:
            analysis: analyze_actual_usage() 결과
        """
        print("=" * 80)
        print("📊 실제 사용 비용 분석")
        print("=" * 80)
        print()
        
        print("📥 입력 (Format Instructions, rough chars):")
        effective_mode = analysis.get('effective_mode', 'unknown')
        mode_info = f" [{effective_mode}]" if effective_mode != 'unknown' else ""
        print(f"  JSON:  {analysis['json_input_chars']:>6,} rough chars")
        print(f"  TOON:  {analysis['toon_input_chars']:>6,} rough chars{mode_info}")
        if analysis.get("json_input_tokens") is not None and analysis.get("toon_input_tokens") is not None:
            print(
                f"  Tokens: JSON {analysis['json_input_tokens']:,} / "
                f"TOON {analysis['toon_input_tokens']:,}"
            )
        diff = analysis['input_diff']  # json - toon
        percent = analysis['input_diff_percent']  # ((toon - json) / json) * 100
        
        # effective_mode가 json이면 경고
        if effective_mode == 'json':
            print(f"  ⚠️  자동 폴백: 모델이 복잡하여 JSON 모드로 전환됨 (depth 제한 초과)")
        
        if diff > 0:
            # JSON이 더 큼 -> TOON이 더 짧음
            print(f"  차이:  {diff:>+6,} rough chars (TOON이 {abs(percent):.1f}% 더 짧음)")
        else:
            # TOON이 더 큼 -> TOON이 더 김
            print(f"  차이:  {diff:>+6,} rough chars (TOON이 {abs(percent):.1f}% 더 김)")
        print()
        
        print("📤 출력 (Actual Output, rough chars):")
        print(f"  JSON:  {analysis['json_output_chars']:>6,} rough chars ({analysis['json_output_lines']:>3} lines)")
        print(f"  TOON:  {analysis['toon_output_chars']:>6,} rough chars ({analysis['toon_output_lines']:>3} lines)")
        if analysis.get("json_output_tokens") is not None and analysis.get("toon_output_tokens") is not None:
            print(
                f"  Tokens: JSON {analysis['json_output_tokens']:,} / "
                f"TOON {analysis['toon_output_tokens']:,}"
            )
        diff = analysis['output_diff']  # json - toon
        percent = analysis['output_diff_percent']  # ((toon - json) / json) * 100
        if diff > 0:
            # JSON이 더 큼 -> TOON이 더 짧음
            print(f"  차이:  {diff:>+6,} rough chars (TOON이 {abs(percent):.1f}% 더 짧음)")
        else:
            # TOON이 더 큼 -> JSON이 더 짧음
            print(f"  차이:  {diff:>+6,} rough chars (JSON이 {abs(percent):.1f}% 더 짧음)")
        print()
        
        print("💰 총 비용 (입력 + 출력, rough chars):")
        print(f"  JSON:  {analysis['json_total_chars']:>6,} rough chars")
        print(f"  TOON:  {analysis['toon_total_chars']:>6,} rough chars")
        saved = analysis['chars_saved']
        if saved > 0:
            print(f"  절감:  {saved:>+6,} rough chars ({analysis['total_reduction_percent']:.1f}%) ✅")
        else:
            print(f"  추가:  {saved:>+6,} rough chars ({abs(analysis['total_reduction_percent']):.1f}%) ⚠️")
        print()
        
        print("📦 데이터 오버헤드:")
        print(f"  실제 데이터: {analysis['data_size']:>6,} rough chars (최소화된 JSON)")
        print(f"  JSON 오버헤드: {analysis['json_overhead_ratio']:.2f}x")
        print(f"  TOON 오버헤드: {analysis['toon_overhead_ratio']:.2f}x")
        print()
        
        # 비용 추정 (GPT-4 기준 예시)
        if saved > 0:
            print("💵 비용 절감 추정 (GPT-4o 기준: $2.5/1M input, $10/1M output):")
            input_cost_saved = (analysis['input_diff'] / 1_000_000) * 2.5
            output_cost_saved = (analysis['output_diff'] / 1_000_000) * 10.0
            total_cost_saved = input_cost_saved + output_cost_saved
            
            print(f"  요청당 절감: ${abs(total_cost_saved):.6f}")
            print(f"  1,000회: ${abs(total_cost_saved * 1000):.3f}")
            print(f"  10,000회: ${abs(total_cost_saved * 10000):.2f}")
            print(f"  100,000회: ${abs(total_cost_saved * 100000):.2f}")
        
        print()
        print("=" * 80)
    
    @staticmethod
    def measure_json_instructions(model: Type[BaseModel]) -> PromptCostMetrics:
        """JSON Structured Output의 format instructions 길이를 측정합니다.
        
        Args:
            model: Pydantic 모델
            
        Returns:
            PromptCostMetrics: 측정된 메트릭
        """
        # JSON schema를 문자열로 변환 (LLM에게 전달되는 형태)
        schema = model.model_json_schema()
        schema_str = json.dumps(schema, indent=2, ensure_ascii=False)
        
        # 일반적인 JSON structured output 지시문
        instructions = f"""Please respond with a JSON object that matches this schema:

{schema_str}

Important:
- Output ONLY valid JSON
- Follow the exact schema structure
- Include all required fields
"""
        
        return PromptCostMetrics(
            format_instructions_length=len(instructions),
            format_instructions_lines=len(instructions.splitlines()),
            format_instructions_tokens=CostAnalyzer._count_tokens(instructions),
        )
    
    @staticmethod
    def measure_toon_instructions(
        model: Type[BaseModel],
        cfg: Optional[ParserConfig] = None
    ) -> PromptCostMetrics:
        """TOON 포맷의 format instructions 길이를 측정합니다.
        
        Args:
            model: Pydantic 모델
            cfg: 파서 설정
            
        Returns:
            PromptCostMetrics: 측정된 메트릭
        """
        parser = ToonParser(model=model, cfg=cfg or ParserConfig())
        instructions = parser.get_format_instructions()
        
        return PromptCostMetrics(
            format_instructions_length=len(instructions),
            format_instructions_lines=len(instructions.splitlines()),
            format_instructions_tokens=CostAnalyzer._count_tokens(instructions),
        )
    
    @staticmethod
    def measure_output_length(data: Any, format_type: str = "json") -> tuple[int, int]:
        """실제 데이터의 출력 길이를 측정합니다.
        
        Args:
            data: 측정할 데이터 (dict, BaseModel, 또는 str)
            format_type: "json" 또는 "toon"
            
        Returns:
            tuple[길이, 라인수]
        """
        if isinstance(data, BaseModel):
            data = data.model_dump()
        
        if format_type == "json":
            output = json.dumps(data, indent=2, ensure_ascii=False)
        elif format_type == "toon":
            from .simple_toon import SimpleToon
            encoder = SimpleToon()
            output = encoder.encode(data)
        else:
            output = str(data)
        
        return len(output), len(output.splitlines())
    
    @staticmethod
    def compare_formats(
        model: Type[BaseModel],
        sample_data: list[Dict[str, Any]] | None = None,
        cfg: Optional[ParserConfig] = None,
    ) -> FormatComparison:
        """JSON과 TOON 포맷의 비용을 비교합니다.
        
        Args:
            model: Pydantic 모델
            sample_data: 비교를 위한 샘플 데이터 리스트 (없으면 출력 비교 생략)
            cfg: 파서 설정
            
        Returns:
            FormatComparison: 비교 결과
        """
        # 입력 측정
        json_metrics = CostAnalyzer.measure_json_instructions(model)
        toon_metrics = CostAnalyzer.measure_toon_instructions(model, cfg)
        
        # 출력 측정 (샘플 데이터가 있는 경우)
        if sample_data:
            json_lengths = []
            json_lines = []
            toon_lengths = []
            toon_lines = []
            data_sizes = []
            
            for data in sample_data:
                # 실제 데이터 크기 (JSON으로 최소화)
                data_size = len(json.dumps(data, ensure_ascii=False, separators=(',', ':')))
                data_sizes.append(data_size)
                
                # JSON 출력
                json_len, json_line = CostAnalyzer.measure_output_length(data, "json")
                json_lengths.append(json_len)
                json_lines.append(json_line)
                
                # TOON 출력
                toon_len, toon_line = CostAnalyzer.measure_output_length(data, "toon")
                toon_lengths.append(toon_len)
                toon_lines.append(toon_line)
            
            # 평균 계산
            avg_data_size = sum(data_sizes) / len(data_sizes)
            json_metrics.avg_output_length = int(sum(json_lengths) / len(json_lengths))
            json_metrics.avg_output_lines = int(sum(json_lines) / len(json_lines))
            json_metrics.output_to_data_ratio = json_metrics.avg_output_length / avg_data_size
            
            toon_metrics.avg_output_length = int(sum(toon_lengths) / len(toon_lengths))
            toon_metrics.avg_output_lines = int(sum(toon_lines) / len(toon_lines))
            toon_metrics.output_to_data_ratio = toon_metrics.avg_output_length / avg_data_size
        
        # 비교 계산
        input_reduction = (
            (toon_metrics.format_instructions_length - json_metrics.format_instructions_length)
            / json_metrics.format_instructions_length * 100
        )
        
        if json_metrics.avg_output_length > 0:
            output_reduction = (
                (toon_metrics.avg_output_length - json_metrics.avg_output_length)
                / json_metrics.avg_output_length * 100
            )
            
            json_total = json_metrics.format_instructions_length + json_metrics.avg_output_length
            toon_total = toon_metrics.format_instructions_length + toon_metrics.avg_output_length
            total_reduction = (toon_total - json_total) / json_total * 100
        else:
            output_reduction = 0.0
            total_reduction = input_reduction
        
        return FormatComparison(
            json_metrics=json_metrics,
            toon_metrics=toon_metrics,
            input_reduction_percent=input_reduction,
            output_reduction_percent=output_reduction,
            total_reduction_percent=total_reduction,
        )
    
    @staticmethod
    def estimate_cost_savings(
        comparison: FormatComparison,
        requests_per_day: int = 1000,
        cost_per_million_chars: float = 1.0,  # 예: $1 per 1M chars
    ) -> Dict[str, Any]:
        """비용 절감액을 추정합니다.
        
        Args:
            comparison: 포맷 비교 결과
            requests_per_day: 하루 요청 수
            cost_per_million_chars: 100만 문자당 비용 (USD)
            
        Returns:
            Dict: 비용 절감 추정치
        """
        json_total = (
            comparison.json_metrics.format_instructions_length +
            comparison.json_metrics.avg_output_length
        )
        toon_total = (
            comparison.toon_metrics.format_instructions_length +
            comparison.toon_metrics.avg_output_length
        )
        
        chars_saved_per_request = json_total - toon_total
        chars_saved_per_day = chars_saved_per_request * requests_per_day
        chars_saved_per_month = chars_saved_per_day * 30
        chars_saved_per_year = chars_saved_per_day * 365
        
        cost_saved_per_day = (chars_saved_per_day / 1_000_000) * cost_per_million_chars
        cost_saved_per_month = (chars_saved_per_month / 1_000_000) * cost_per_million_chars
        cost_saved_per_year = (chars_saved_per_year / 1_000_000) * cost_per_million_chars
        
        return {
            "chars_saved_per_request": chars_saved_per_request,
            "chars_saved_per_day": chars_saved_per_day,
            "chars_saved_per_month": chars_saved_per_month,
            "chars_saved_per_year": chars_saved_per_year,
            "cost_saved_per_day_usd": cost_saved_per_day,
            "cost_saved_per_month_usd": cost_saved_per_month,
            "cost_saved_per_year_usd": cost_saved_per_year,
        }
    
    @staticmethod
    def print_cost_savings(
        comparison: FormatComparison,
        requests_per_day: int = 1000,
        cost_per_million_chars: float = 1.0,
    ) -> None:
        """비용 절감액을 출력합니다."""
        savings = CostAnalyzer.estimate_cost_savings(
            comparison, requests_per_day, cost_per_million_chars
        )
        
        print("=" * 80)
        print("💰 비용 절감 추정")
        print("=" * 80)
        print(f"기준: {requests_per_day:,}회/일, ${cost_per_million_chars}/1M rough chars")
        print()
        
        if savings["chars_saved_per_request"] > 0:
            print("✅ TOON 사용 시 절감:")
            print(f"  요청당: {savings['chars_saved_per_request']:,} rough chars")
            print(f"  일간: {savings['chars_saved_per_day']:,} rough chars (${savings['cost_saved_per_day_usd']:.2f})")
            print(f"  월간: {savings['chars_saved_per_month']:,} rough chars (${savings['cost_saved_per_month_usd']:.2f})")
            print(f"  연간: {savings['chars_saved_per_year']:,} rough chars (${savings['cost_saved_per_year_usd']:.2f})")
        else:
            print("⚠️ TOON 사용 시 추가 비용:")
            print(f"  요청당: {abs(savings['chars_saved_per_request']):,} rough chars")
            print(f"  일간: {abs(savings['chars_saved_per_day']):,} rough chars (${abs(savings['cost_saved_per_day_usd']):.2f})")
            print(f"  월간: {abs(savings['chars_saved_per_month']):,} rough chars (${abs(savings['cost_saved_per_month_usd']):.2f})")
            print(f"  연간: {abs(savings['chars_saved_per_year']):,} rough chars (${abs(savings['cost_saved_per_year_usd']):.2f})")
        
        print()
        print("=" * 80)
