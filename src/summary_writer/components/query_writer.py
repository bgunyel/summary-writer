import datetime
from typing import Any
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import get_usage_metadata_callback

from ai_common import LlmServers, Queries, get_llm
from ..enums import Node
from ..state import SummaryState
# from ..configuration import Configuration
from ai_common.utils import get_config_from_runnable


QUERY_WRITER_INSTRUCTIONS = """
Your goal is to generate targeted web search queries that will gather comprehensive information for writing a summary about a topic.
You will generate exactly {number_of_queries} queries.

<topic>
{topic}
</topic>

Today's date is:
<today>
{today}
</today>

When generating the search queries:
1. Make sure to cover different aspects of the topic.
2. Make sure that your queries account for the most current information available as of today.

Your queries should be:
- Specific enough to avoid generic or irrelevant results.
- Targeted to gather specific information about the topic.
- Diverse enough to cover all aspects of the summary plan.

It is very important that you generate exactly {number_of_queries} queries.
Generate targeted web search queries that will gather specific information about the given topic.
"""


class QueryWriter:
    def __init__(self, llm_server: LlmServers, model_params: dict[str, Any], configuration_file_directory: str):
        self.model_name = model_params['language_model']
        self.configuration_file_directory = configuration_file_directory
        model_params['model_name'] = self.model_name
        self.base_llm = get_llm(llm_server=llm_server, model_params=model_params)
        self.structured_llm = self.base_llm.with_structured_output(Queries)

    def run(self, state: SummaryState, config: RunnableConfig) -> SummaryState:
        """
        Writes queries for comprehensive web search.
            :param state: The current flow state
            :param config: The configuration
        """

        # configurable = Configuration.from_runnable(runnable=config)
        configurable = get_config_from_runnable(configuration_module_prefix = 'src.summary_writer.configuration', config = config)
        state.steps.append(Node.QUERY_WRITER.value)
        instructions = QUERY_WRITER_INSTRUCTIONS.format(topic=state.topic,
                                                        today=datetime.date.today().isoformat(),
                                                        number_of_queries=configurable.number_of_queries)
        with get_usage_metadata_callback() as cb:
            results = self.structured_llm.invoke(instructions)
            state.token_usage[self.model_name]['input_tokens'] += cb.usage_metadata[self.model_name]['input_tokens']
            state.token_usage[self.model_name]['output_tokens'] += cb.usage_metadata[self.model_name]['output_tokens']
        state.search_queries = results.queries
        return state
