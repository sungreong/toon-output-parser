from __future__ import annotations

import json
import os

from eval_metrics import print_evaluation
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain_community.chat_models import ChatOpenAI

from toon_langchain_parser import CostAnalyzer, ToonOutputParser
from toon_langchain_parser.toon_parser_ultimate import ParserConfig


class IncidentWindow(BaseModel):
    start_time: str = ""
    end_time: str = ""
    duration_minutes: int | None = None


class TimelineEvent(BaseModel):
    time: str = ""
    actor: str = ""
    event: str = ""


class MitigationAction(BaseModel):
    owner: str = ""
    action: str = ""
    status: str = ""


class OpsIncidentReport(BaseModel):
    ticket_id: str = ""
    service: str = ""
    severity: str = ""
    incident_window: IncidentWindow = Field(default_factory=IncidentWindow)
    impacted_regions: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)
    probable_root_causes: list[str] = Field(default_factory=list)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    mitigation_actions: list[MitigationAction] = Field(default_factory=list)
    customer_message: str = ""
    next_update_eta: str = ""


def extract_ops_incident(document: str) -> tuple[str, OpsIncidentReport | None, str | None, dict | None]:
    cfg = ParserConfig(instructions_mode="adaptive")
    parser = ToonOutputParser(model=OpsIncidentReport, cfg=cfg)

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract incident report fields exactly from the document. Keep output schema-conformant.",
            ),
            (
                "human",
                "Incident note:\n{document}\n\n"
                "Output requirements:\n"
                "- timeline must be list[object] with time/actor/event\n"
                "- mitigation_actions must be list[object] with owner/action/status\n"
                "- use empty typed values when missing\n"
                "{format_instructions}",
            ),
        ]
    )

    raw_output = (prompt | llm | StrOutputParser()).invoke(
        {"document": document, "format_instructions": parser.get_format_instructions()}
    )

    try:
        result = parser.parse(raw_output)
        cost = CostAnalyzer.analyze_actual_usage(
            model=OpsIncidentReport,
            toon_raw_output=raw_output,
            parsed_result=result,
            cfg=cfg,
        )
        return raw_output, result, None, cost
    except Exception as e:
        return raw_output, None, str(e), None


def main() -> None:
    document = """
[INCIDENT] API latency spike on checkout
Ticket: INC-2026-0315
Service: checkout-api
Severity: SEV2

Window:
- Start: 2026-03-05 09:12 KST
- End: 2026-03-05 10:01 KST
- Duration: 49 minutes

Impacted regions: ap-northeast-2, us-west-2

Symptoms observed:
- p95 latency increased from 180ms to 2.4s
- 5xx error rate peaked at 3.1%
- payment authorization timeout increased

Probable causes:
- DB connection pool saturation after deploy
- cache miss surge due to key-namespace mismatch

Timeline:
09:12 - oncall-bot - alert fired (latency SLO burn)
09:18 - minji - rollback started for checkout-api v2.8.1
09:27 - sora - increased DB pool from 40 to 80
09:41 - junho - cache key hotfix applied
10:01 - oncall-bot - metrics recovered to baseline

Mitigation actions:
- owner: minji, action: rollback to v2.8.0, status: completed
- owner: sora, action: tune DB pool and timeout, status: completed
- owner: junho, action: patch cache key namespace, status: completed

Customer message:
"We are investigating elevated checkout latency and error rates. A mitigation is in progress."
Next update ETA: 2026-03-05 10:30 KST
""".strip()

    print("=" * 80)
    print("Real-world Ops Incident Extraction")
    print("=" * 80)
    print("\nINPUT DOCUMENT:\n")
    print(document)

    raw_output, result, parse_error, cost = extract_ops_incident(document)

    print("\n" + "=" * 80)
    print("RAW MODEL OUTPUT")
    print("=" * 80)
    print(raw_output)

    if parse_error:
        print("\n" + "=" * 80)
        print("PARSING ERROR")
        print("=" * 80)
        print(parse_error)
        return

    assert result is not None
    parsed = result.model_dump()
    print("\n" + "=" * 80)
    print("PARSED JSON")
    print("=" * 80)
    print(json.dumps(parsed, ensure_ascii=False, indent=2))

    expected = {
        "ticket_id": "INC-2026-0315",
        "service": "checkout-api",
        "severity": "SEV2",
        "incident_window": {"duration_minutes": 49},
        "impacted_regions": ["ap-northeast-2"],
    }
    print_evaluation("QUALITY", parsed, expected)

    print("\n" + "=" * 80)
    print("PRACTICAL CHECKS")
    print("=" * 80)
    print(f"- timeline items: {len(result.timeline)}")
    print(f"- mitigation action items: {len(result.mitigation_actions)}")
    print(f"- symptoms captured: {len(result.symptoms)}")

    if cost:
        print("\n" + "=" * 80)
        print("COST ANALYSIS")
        print("=" * 80)
        CostAnalyzer.print_actual_usage_analysis(cost)


if __name__ == "__main__":
    main()
