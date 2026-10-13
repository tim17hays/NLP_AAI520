"""Planning and routing for the financial research agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .data_tools import StockResearchTools
from .news_pipeline import NewsPipeline
from .llm import generate_draft

@dataclass(frozen=True)
class ResearchPlan:
    """A structured plan describing which specialists should be used."""

    ticker: str
    research_question: str
    specialists: list[str]
    steps: list[str]
    retrieved_lessons: list = field(default_factory=list)

class ResearchPlanner:
    """Create a research plan by routing a question to specialists."""

    def __init__(
        self,
        research_tools: StockResearchTools | None = None,
        news_pipeline: NewsPipeline | None = None,
    ) -> None:
        self._research_tools = research_tools or StockResearchTools()
        self._news_pipeline = news_pipeline or NewsPipeline()

        self._routing_keywords = {
            "market": (
                "stock price",
                "stock performance",
                "share price",
                "price",
                "market",
                "return",
                "volatility",
            ),
            "earnings": (
                "earnings",
                "revenue",
                "profit",
                "financial",
                "income",
                "margin",
                "guidance",
            ),
            "news": (
                "news",
                "recent events",
                "latest",
                "headline",
                "sentiment",
                "announcement",
            ),
            "company": (
                "company",
                "business",
                "industry",
                "sector",
                "employees",
            ),
        }

    @staticmethod
    def normalize_ticker(ticker: str) -> str:
        """Normalize a stock ticker for the research plan."""
        symbol = ticker.strip().upper()
        if not symbol:
            raise ValueError("ticker must not be empty")
        return symbol

    def route(self, research_question: str) -> list[str]:
        """Select specialists based on keywords in the research question."""
        question = research_question.strip().lower()

        if not question:
            raise ValueError("research_question must not be empty")

        selected: list[str] = []

        for specialist, keywords in self._routing_keywords.items():
            if any(keyword in question for keyword in keywords):
                selected.append(specialist)

        return selected
    
    def plan(self, ticker: str, research_question: str, lessons: list | None = None) -> ResearchPlan:
        """Create a research plan for the given stock and question."""
        symbol = self.normalize_ticker(ticker)
        specialists = self.route(research_question)

        if not specialists:
            specialists = ["company"]

        steps = [
            f"Retrieve evidence from the {specialist} specialist."
            for specialist in specialists
        ]

        if lessons:
            steps.insert(0, "Review retrieved lessons before collecting and drafting evidence: "
                         + str(lessons))

        steps.append("Combine the retrieved evidence into a draft analysis.")
        steps.append("Send the draft analysis and supporting evidence for evaluation.")

        return ResearchPlan(
            ticker=symbol,
            research_question=research_question.strip(),
            specialists=specialists,
            steps=steps,
            retrieved_lessons=list(lessons or []),
        )
    
    def execute(self, ticker: str, research_question: str, lessons: list | None = None) -> dict[str, Any]:
        """Execute the specialists selected for the research question."""
        symbol = self.normalize_ticker(ticker)
        specialists = self.route(research_question)

        if not specialists:
            specialists = ["company"]

        plan = self.plan(symbol, research_question, lessons=lessons)
        evidence: dict[str, Any] = {}

        for specialist in specialists:
            if specialist == "market":
                result = self._research_tools.market_snapshot(symbol)
                evidence["market"] = result.to_dict()

            elif specialist == "earnings":
                result = self._research_tools.financial_snapshot(symbol)
                evidence["earnings"] = result.to_dict()

            elif specialist == "company":
                result = self._research_tools.company_profile(symbol)
                evidence["company"] = result.to_dict()

            elif specialist == "news":
                result = self._news_pipeline.process(symbol)
                evidence["news"] = result

        draft_analysis = generate_draft(
            ticker=symbol,
            research_question=research_question.strip(),
            evidence=evidence,
            lessons=plan.retrieved_lessons,
        )

        return {
            "ticker": symbol,
            "research_question": research_question.strip(),
            "specialists": specialists,
            "evidence": evidence,
            "draft_analysis": draft_analysis,
            "plan": asdict(plan),
            "retrieved_lessons": plan.retrieved_lessons,
            "tool_trace": [
                {"specialist": name, "tool": payload.get("tool", "news_pipeline"),
                 "errors": payload.get("errors", [])}
                for name, payload in evidence.items()
            ],
        }