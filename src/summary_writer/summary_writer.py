from uuid import uuid4
from pprint import pprint
from typing import Literal, Any

from langgraph.graph import START, END, StateGraph
from langchain_core.runnables import RunnableConfig
from ai_common import GraphBase, WebSearch, LlmServers, format_sources

from .state import SummaryState
from .configuration import Configuration
from .enums import Node
from .components.query_writer import QueryWriter
from .components.writer import Writer


def route_research(state: SummaryState, config: RunnableConfig) -> Literal["continue_research", "end_research"]:
    """ Route the research based on the follow-up query """

    configurable = Configuration.from_runnable_config(config=config)
    if state.iteration <= configurable.research_iterations:
        return "continue_research"
    else:
        return "end_research"


class SummaryWriter(GraphBase):

    """
    Reproduction from https://github.com/langchain-ai/ollama-deep-researcher
    """

    def __init__(self, llm_server: LlmServers, llm_config: dict[str, Any], web_search_api_key: str) -> None:
        config = Configuration()
        self.query_writer = QueryWriter(llm_server=llm_server, model_params=llm_config)
        self.web_search = WebSearch(api_key = web_search_api_key,
                                    search_category = config.search_category,
                                    number_of_days_back = config.number_of_days_back,
                                    include_raw_content = True)

        self.writer = Writer(llm_server=llm_server, model_params=llm_config)
        """
        self.summary_reviewer = SummaryReviewer(model_name=settings.REASONING_MODEL, context_window_length=config.context_window_length)
        """
        self.graph = self.build_graph()

    def web_search_run(self, state: SummaryState) -> SummaryState:
        unique_sources = self.web_search.search(search_queries=[query.search_query for query in state.search_queries])
        source_str = format_sources(unique_sources=unique_sources, max_tokens_per_source=5000, include_raw_content=True)
        state.steps.append(Node.WEB_SEARCH.value)
        state.source_str = source_str
        state.unique_sources = unique_sources
        return state

    def get_response(self, input_dict: dict[str, Any], verbose: bool = False) -> str:
        config = {"configurable": {"thread_id": str(uuid4())}}

        in_state = SummaryState(
            content = '',
            iteration=0,
            search_queries = [],
            source_str = '',
            steps = [],
            summary_exists=False,
            topic = input_dict['topic'],
            unique_sources={},
        )
        out_state = self.graph.invoke(in_state, config)
        return out_state['content']

    def build_graph(self):
        workflow = StateGraph(SummaryState, config_schema=Configuration)

        ## Nodes
        workflow.add_node(node=Node.QUERY_WRITER.value, action=self.query_writer.run)
        workflow.add_node(node=Node.WEB_SEARCH.value, action=self.web_search_run)
        workflow.add_node(node=Node.WRITER.value, action=self.writer.run)
        # workflow.add_node(node=Node.SUMMARY_REVIEWER.value, action=self.summary_reviewer.run)

        ## Edges
        workflow.add_edge(start_key=START, end_key=Node.QUERY_WRITER.value)
        workflow.add_edge(start_key=Node.QUERY_WRITER.value, end_key=Node.WEB_SEARCH.value)
        workflow.add_edge(start_key=Node.WEB_SEARCH.value, end_key=Node.WRITER.value)
        workflow.add_edge(start_key=Node.WRITER.value, end_key=END)

        """
        workflow.add_conditional_edges(
            source=Node.SUMMARY_REVIEWER.value,
            path=route_research,
            path_map={
                'continue_research': Node.WEB_SEARCH.value,
                'end_research': END,
            }
        )
        """

        compiled_graph = workflow.compile()
        return compiled_graph
