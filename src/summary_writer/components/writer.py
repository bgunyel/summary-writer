import copy
from typing import Any

from langchain_core.runnables import RunnableConfig
from ai_common import LlmServers, get_llm, strip_thinking_tokens

from ..enums import Node
from ..state import SummaryState
from ..configuration import Configuration


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
    def __init__(self, llm_server: LlmServers, model_params: dict[str, Any]):
        model_params['model_name'] = model_params['reasoning_model']
        self.writer_llm = get_llm(llm_server=llm_server, model_params=model_params)

    def run(self, state: SummaryState, config: RunnableConfig) -> SummaryState:

        configurable = Configuration.from_runnable_config(config=config)

        if state.summary_exists: # Extending existing summary
            instructions = EXTENDING_INSTRUCTIONS.format(topic=state.topic, summary=state.content, search_results=state.source_str)
        else: # Writing a new summary
            instructions = WRITING_INSTRUCTIONS.format(topic=state.topic, context=state.source_str)

        summary = self.writer_llm.invoke(instructions)
        state.steps.append(Node.WRITER.value)
        state.summary_exists = True
        state.content = copy.deepcopy(summary.content)

        if configurable.strip_thinking_tokens:
            state.content = strip_thinking_tokens(text=state.content)

        state.content = state.content.lstrip('\n')
        return state
