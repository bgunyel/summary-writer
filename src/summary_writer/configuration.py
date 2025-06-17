import os
import importlib

from pydantic import BaseModel
from langchain_core.runnables import RunnableConfig
from ai_common import CfgBase, TavilySearchCategory


class Configuration(CfgBase):
    """The configurable fields for the workflow"""
    # thread_id: str
    max_iterations: int
    # max_retries: int # in case LLM call fails, the number of retries
    max_results_per_query: int
    max_tokens_per_source: int
    number_of_days_back: int = None
    number_of_queries: int
    search_category: TavilySearchCategory = "general"
    strip_thinking_tokens: bool # = True
