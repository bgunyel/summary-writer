from pydantic import Field
from ai_common import CfgBase, TavilySearchCategory


class Configuration(CfgBase):
    """The configurable fields for the workflow"""
    citation_style: str = Field(default="numeric")
    max_iterations: int = Field(gt = 0) # (0, inf)
    max_results_per_query: int = Field(gt = 0) # (0, inf)
    max_tokens_per_source: int = Field(gt = 0) # (0, inf)
    min_claim_confidence: float = Field(ge = 0.0, le = 1.0) # [0, 1]
    number_of_days_back: int = None
    number_of_queries: int = Field(gt = 0) # (0, inf)
    search_category: TavilySearchCategory = "general"
    strip_thinking_tokens: bool # = True
