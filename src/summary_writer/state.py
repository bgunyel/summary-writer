from typing import Any, List
from dataclasses import dataclass

from pydantic import BaseModel
from ai_common import SearchQuery

@dataclass
class WriterClaim:
    """Represents a claim extracted by the Writer"""
    text: str
    source_ids: List[str]
    confidence: float
    start_position: int = 0
    end_position: int = 0


class SummaryState(BaseModel):
    """
    Represents the state of our research summary.

    Attributes:
        topic: research topic
        search_queries: list of search queries
        source_str: String of formatted source content from web search
        content: Content generated from sources
        steps: steps followed during graph run

    """
    content: str
    iteration: int = 0
    search_queries: list[SearchQuery]
    source_str: str
    steps: list[str]
    summary_exists: bool = False
    token_usage: dict
    topic: str
    unique_sources: dict[str, Any]
    cumulative_unique_sources: list[dict[str, Any]]
    cumulative_search_queries: list

    bibliography: str # reference list
    cited_content: str # summary with inline citations
    claims: List[WriterClaim] # for tracking claims
