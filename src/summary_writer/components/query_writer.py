import datetime
from typing import Any
from langchain_core.runnables import RunnableConfig

from ai_common import LlmServers, Queries, get_llm
from ..enums import Node
from ..state import SummaryState
from ..configuration import Configuration


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
    def __init__(self, llm_server: LlmServers, model_params: dict[str, Any]):
        model_params['model_name'] = model_params['language_model']
        self.base_llm = get_llm(llm_server=llm_server, model_params=model_params)
        self.structured_llm = self.base_llm.with_structured_output(Queries)

    def run(self, state: SummaryState, config: RunnableConfig) -> SummaryState:
        """
        Writes queries for comprehensive web search.
            :param state: The current flow state
            :param config: The configuration
        """
        configurable = Configuration.from_runnable_config(config=config)
        state.steps.append(Node.QUERY_WRITER.value)

        instructions = QUERY_WRITER_INSTRUCTIONS.format(topic=state.topic,
                                                        today=datetime.date.today().isoformat(),
                                                        number_of_queries=configurable.number_of_queries)
        results = self.structured_llm.invoke(instructions)
        state.search_queries = results.queries
        return state
