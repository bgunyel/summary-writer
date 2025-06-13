from uuid import uuid4
from typing import Literal, Any, Final

from langgraph.graph import START, END, StateGraph
from langchain_core.runnables import RunnableConfig
from ai_common import GraphBase, LlmServers, format_sources, TavilySearchCategory
from ai_common.components import QueryWriter, WebSearchNode

from .state import SummaryState
from .enums import Node
from .configuration import Configuration
from .components.writer import Writer


def route_research(state: SummaryState, config: RunnableConfig) -> Literal["continue_research", "end_research"]:
    """ Route the research based on the follow-up query """

    configurable = Configuration.from_runnable_config(config=config)
    if state.iteration <= configurable.max_iterations:
        return "continue_research"
    else:
        return "end_research"


class SummaryWriter(GraphBase):

    """
    Reproduction from https://github.com/langchain-ai/ollama-deep-researcher
    """

    def __init__(self,
                 llm_server: LlmServers,
                 llm_config: dict[str, Any],
                 web_search_api_key: str) -> None:
        self.models = list({llm_config['language_model'], llm_config['reasoning_model']})
        self.configuration_module_prefix: Final = 'summary_writer.configuration'
        self.query_writer = QueryWriter(llm_server = llm_server,
                                        model_params = llm_config,
                                        configuration_module_prefix = self.configuration_module_prefix)
        self.web_search_node = WebSearchNode(llm_server = llm_server,
                                             model_params = llm_config,
                                             web_search_api_key = web_search_api_key,
                                             configuration_module_prefix = self.configuration_module_prefix)
        self.writer = Writer(llm_server=llm_server,
                             model_params=llm_config,
                             configuration_module_prefix=self.configuration_module_prefix)
        """
        self.summary_reviewer = SummaryReviewer(model_name=settings.REASONING_MODEL, context_window_length=config.context_window_length)
        """
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
        )
        out_state = await self.graph.ainvoke(in_state, config)
        out_dict = {
            'content': out_state['content'],
            'unique_sources': out_state['unique_sources'],
            'token_usage': out_state['token_usage'],
        }
        return out_dict

    def get_response(self, input_dict: dict[str, Any], verbose: bool = False) -> str:
        config = {"configurable": {"thread_id": str(uuid4())}}
        out_dict = self.run(topic = input_dict["topic"], config = config)
        return out_dict['content']

    def build_graph(self):
        workflow = StateGraph(SummaryState, config_schema=Configuration)

        ## Nodes
        workflow.add_node(node=Node.QUERY_WRITER.value, action=self.query_writer.run)
        workflow.add_node(node=Node.WEB_SEARCH.value, action=self.web_search_node.run)
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
