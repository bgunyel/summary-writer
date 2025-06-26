import asyncio
from uuid import uuid4
from typing import Literal, Any, Final

from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from ai_common import GraphBase
from ai_common.components import QueryWriter, WebSearchNode

from .state import SummaryState
from .enums import Node
from .configuration import Configuration
from .components import Writer, Reviewer, CitationsManager


def route_research(state: SummaryState, config: RunnableConfig) -> Literal["continue_research", "end_research"]:
    """ Route the research based on the follow-up query """

    configurable = Configuration.from_runnable(runnable=config)
    if state.iteration < configurable.max_iterations:
        return "continue_research"
    else:
        return "end_research"


class SummaryWriter(GraphBase):

    """
    Reproduction from https://github.com/langchain-ai/ollama-deep-researcher
    """

    def __init__(self, llm_config: dict[str, Any], web_search_api_key: str) -> None:
        self.memory_saver = MemorySaver()
        self.models = list({llm_config['language_model']['model'], llm_config['reasoning_model']['model']})
        self.configuration_module_prefix: Final = 'summary_writer.configuration'
        self.query_writer = QueryWriter(model_params = llm_config['language_model'],
                                        configuration_module_prefix = self.configuration_module_prefix)
        self.web_search_node = WebSearchNode(model_params = llm_config['language_model'],
                                             web_search_api_key = web_search_api_key,
                                             configuration_module_prefix = self.configuration_module_prefix)
        self.writer = Writer(model_params=llm_config['reasoning_model'],
                             configuration_module_prefix=self.configuration_module_prefix,
                             enable_citations = False,
                             citation_style = "numeric",
                             min_confidence = 0.6)
        self.citations_manager = CitationsManager(model_params=llm_config['reasoning_model'],
                                                  configuration_module_prefix=self.configuration_module_prefix)
        self.reviewer = Reviewer(model_params=llm_config['reasoning_model'],
                                 configuration_module_prefix=self.configuration_module_prefix)
        self.graph = self.build_graph()

    async def run(self, topic: str, config: RunnableConfig) -> dict[str, Any]:
        in_state = SummaryState(
            content='',
            iteration=0,
            search_queries=[],
            source_str='',
            steps=[],
            summary_exists=False,
            token_usage={m: {'input_tokens': 0, 'output_tokens': 0} for m in self.models},
            topic=topic,
            unique_sources={},
            cumulative_unique_sources=[],
            cumulative_search_queries=[],
            bibliography='',
            cited_content='',
            claims=[],
        )
        out_state = await self.graph.ainvoke(in_state, config)
        out_dict = {
            'content': out_state['content'],
            'unique_sources': {k: v for d in out_state['cumulative_unique_sources'] for k, v in d.items()},
            'token_usage': out_state['token_usage'],
        }
        return out_dict

    def get_response(self, input_dict: dict[str, Any], verbose: bool = False) -> str:
        config = {
            "configurable": {
                'thread_id': str(uuid4()),
                'max_iterations': 3,
                'max_results_per_query': 4,
                'max_tokens_per_source': 10000,
                'number_of_days_back': 1e6,
                'number_of_queries': 3,
                'search_category': 'general',
                'strip_thinking_tokens': True,
            }
        }
        event_loop = asyncio.new_event_loop()
        out_dict = event_loop.run_until_complete(self.run(topic=input_dict["topic"], config=config))
        event_loop.close()
        return out_dict['content']

    def build_graph(self):
        workflow = StateGraph(SummaryState, config_schema=Configuration)

        ## Nodes
        workflow.add_node(node=Node.QUERY_WRITER, action=self.query_writer.run)
        workflow.add_node(node=Node.WEB_SEARCH, action=self.web_search_node.run)
        workflow.add_node(node=Node.WRITER, action=self.writer.run)
        workflow.add_node(node=Node.CITATIONS_MANAGER, action=self.citations_manager.run)
        workflow.add_node(node=Node.REVIEWER, action=self.reviewer.run)

        ## Edges
        workflow.add_edge(start_key=START, end_key=Node.QUERY_WRITER)
        workflow.add_edge(start_key=Node.QUERY_WRITER, end_key=Node.WEB_SEARCH)
        workflow.add_edge(start_key=Node.WEB_SEARCH, end_key=Node.WRITER)
        workflow.add_edge(start_key=Node.WRITER, end_key=Node.CITATIONS_MANAGER)
        workflow.add_edge(start_key=Node.CITATIONS_MANAGER, end_key=Node.REVIEWER)

        workflow.add_conditional_edges(
            source=Node.REVIEWER,
            path=route_research,
            path_map={
                'continue_research': Node.WEB_SEARCH,
                'end_research': END,
            }
        )


        ## Compile Graph
        compiled_graph = workflow.compile(checkpointer=self.memory_saver)
        return compiled_graph
