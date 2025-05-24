from typing import Any

from pydantic import BaseModel
from ai_common import SearchQuery


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
    topic: str
    unique_sources: dict[str, Any]
