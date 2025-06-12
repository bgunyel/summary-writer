import copy
from typing import Any, Final

from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import get_usage_metadata_callback
from ai_common import LlmServers, get_llm, strip_thinking_tokens, get_config_from_runnable

from ..enums import Node
from ..state import SummaryState


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
1. Highlight the most relevant information from each source.
2. Provide a concise overview of the key points related to the topic.
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

class Writer:
    def __init__(self, llm_server: LlmServers, model_params: dict[str, Any], configuration_module_prefix: str):
        self.model_name = model_params['reasoning_model']
        self.configuration_module_prefix: Final = configuration_module_prefix
        model_params['model_name'] = self.model_name
        self.writer_llm = get_llm(llm_server=llm_server, model_params=model_params)

    def run(self, state: SummaryState, config: RunnableConfig) -> SummaryState:

        configurable = get_config_from_runnable(
            configuration_module_prefix = self.configuration_module_prefix,
            config = config
        )

        if state.summary_exists: # Extending existing summary
            instructions = EXTENDING_INSTRUCTIONS.format(topic=state.topic, summary=state.content, search_results=state.source_str)
        else: # Writing a new summary
            instructions = WRITING_INSTRUCTIONS.format(topic=state.topic, context=state.source_str)

        with get_usage_metadata_callback() as cb:
            summary = self.writer_llm.invoke(instructions)
            state.token_usage[self.model_name]['input_tokens'] += cb.usage_metadata[self.model_name]['input_tokens']
            state.token_usage[self.model_name]['output_tokens'] += cb.usage_metadata[self.model_name]['output_tokens']
            state.steps.append(Node.WRITER.value)
            state.summary_exists = True
            state.content = copy.deepcopy(summary.content)

        if configurable.strip_thinking_tokens:
            state.content = strip_thinking_tokens(text=state.content)

        state.content = state.content.lstrip('\n')
        return state
