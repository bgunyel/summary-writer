import copy
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import get_usage_metadata_callback
from langchain.chat_models import init_chat_model
from ai_common import strip_thinking_tokens, get_config_from_runnable

from ..enums import Node
from ..state import SummaryState
from .writer_with_citations import WriterWithCitations


WRITING_INSTRUCTIONS = """
You are an expert writer working on writing a summary about a given topic using the provided context.

<Goal>
Write a high quality summary of the provided context.
</Goal>

The topic you are writing about:
<topic>
{topic}
</topic>

The context you are going to use:
<context>
{context}
</context>

<Requirements>
1. Highlight every relevant information from each source.
2. Provide a detailed overview of the key points related to the topic.
3. Emphasize significant findings or insights.
4. Ensure a coherent flow of information.
</Requirements>

<Formatting>
- Start directly with the summary, without preamble or titles. Do not use XML tags in the output.  
</Formatting>

<Task>
Think carefully about the provided context first. Then write a summary using the provided context.
</Task>
"""

EXTENDING_INSTRUCTIONS = """
You are an expert writer working on extending summary with new search results:

<Goal>
Extend the existing summary with new search results.
</Goal>

The topic of the summary:
<topic>
{topic}
</topic>

The existing summary:
<summary>
{summary}
</summary>

The new search results:
<search_results>
{search_results}
</search_results>

<Requirements>
1. Read the existing summary and new search results carefully.                                                    
2. Compare the new information with the existing summary.                                                         
3. For each piece of new information:                                                                             
    a. If it's related to existing points, integrate it into the relevant paragraph.                               
    b. If it's entirely new but relevant, add a new paragraph with a smooth transition.                            
    c. If it's not relevant to the user topic, skip it.                                                            
4. Ensure all additions are relevant to the user's topic.                                                         
5. Verify that your final output differs from the input summary.
</Requirements>

<Formatting>
- Start directly with the updated summary, without preamble or titles. Do not use XML tags in the output.  
</Formatting>

<Task>
Think carefully about the provided search results first. Then update the existing summary accordingly.
</Task>
"""

# Backward compatibility wrapper
class Writer(WriterWithCitations):
    """
    Backward compatible Writer class.

    If you want to maintain the original behavior, set enable_citations=False.
    If you want citations, set enable_citations=True.
    """

    def __init__(self, model_params: dict[str, Any], configuration_module_prefix: str,
                 enable_citations: bool = True, citation_style: str = "numeric",
                 min_confidence: float = 0.6):

        if enable_citations:
            super().__init__(model_params, configuration_module_prefix, citation_style, min_confidence)
        else:
            # Original behavior
            self.model_name = model_params['model']
            self.configuration_module_prefix = configuration_module_prefix
            self.writer_llm = init_chat_model(
                model=model_params['model'],
                model_provider=model_params['model_provider'],
                api_key=model_params['api_key'],
                **model_params['model_args']
            )

        self.enable_citations = enable_citations

    def run(self, state: SummaryState, config: RunnableConfig) -> SummaryState:

        state.cumulative_unique_sources.append(state.unique_sources)
        state.cumulative_search_queries.append(state.search_queries)

        if self.enable_citations:
            return super().run(state, config)
        else:
            # Original implementation
            configurable = get_config_from_runnable(
                configuration_module_prefix=self.configuration_module_prefix,
                config=config
            )

            if state.summary_exists:
                instructions = EXTENDING_INSTRUCTIONS.format(
                    topic=state.topic,
                    summary=state.content,
                    search_results=state.source_str
                )
            else:
                instructions = WRITING_INSTRUCTIONS.format(
                    topic=state.topic,
                    context=state.source_str
                )

            with get_usage_metadata_callback() as cb:
                summary = self.writer_llm.invoke(instructions)
                state.token_usage[self.model_name]['input_tokens'] += cb.usage_metadata[self.model_name]['input_tokens']
                state.token_usage[self.model_name]['output_tokens'] += cb.usage_metadata[self.model_name][
                    'output_tokens']
                state.steps.append(Node.WRITER)
                state.summary_exists = True
                state.content = copy.deepcopy(summary.content)

            if configurable.strip_thinking_tokens:
                state.content = strip_thinking_tokens(text=state.content)

            state.content = state.content.lstrip('\n')
            return state
