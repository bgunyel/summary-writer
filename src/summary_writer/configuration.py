from pydantic import BaseModel
from langchain_core.runnables import RunnableConfig
from ai_common import CfgBase, TavilySearchCategory


class Configuration(CfgBase):
    """The configurable fields for the workflow"""
    # thread_id: str
    max_iterations: int # = 3
    max_results_per_query: int # = 5
    max_tokens_per_source: int # = 5000
    number_of_days_back: int # = None
    number_of_queries: int # = 3
    search_category: TavilySearchCategory # = "general"
    strip_thinking_tokens: bool # = True
