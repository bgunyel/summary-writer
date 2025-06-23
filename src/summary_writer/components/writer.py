import copy
from typing import Any, Final

from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import get_usage_metadata_callback
from langchain.chat_models import init_chat_model
from ai_common import strip_thinking_tokens, get_config_from_runnable

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

class Writer:
    def __init__(self, model_params: dict[str, Any], configuration_module_prefix: str):
        self.model_name = model_params['model']
        self.configuration_module_prefix: Final = configuration_module_prefix
        self.writer_llm = init_chat_model(
            model=model_params['model'],
            model_provider=model_params['model_provider'],
            api_key=model_params['api_key'],
            **model_params['model_args']
        )

    def run(self, state: SummaryState, config: RunnableConfig) -> SummaryState:
        """
        Execute the writing process to generate or extend a summary based on the current state.
        
        This method determines whether to create a new summary or extend an existing one based on
        the state.summary_exists flag. It uses different instruction templates depending on the
        operation type and tracks token usage throughout the process.
        
        Args:
            state (SummaryState): The current state containing:
                - topic: The subject matter for the summary
                - summary_exists: Boolean flag indicating if extending existing summary
                - content: Current summary content (if extending)
                - source_str: Context or search results to use for writing/extending
                - token_usage: Dictionary tracking token consumption by model
                - steps: List tracking the workflow steps completed
            config (RunnableConfig): Configuration object containing runnable parameters
                and configurable options for the writing process
        
        Returns:
            SummaryState: Updated state object with:
                - content: The newly generated or extended summary text
                - summary_exists: Set to True after successful writing
                - token_usage: Updated with tokens consumed during this operation
                - steps: Appended with Node.WRITER to track completion
        
        Process Flow:
            1. Extracts configuration parameters from the runnable config
            2. Selects appropriate instruction template (extending vs. new summary)
            3. Invokes the LLM with formatted instructions
            4. Captures and records token usage metadata
            5. Updates state with generated content and process tracking
            6. Optionally strips thinking tokens if configured
            7. Cleans up leading whitespace from the final content
        
        Side Effects:
            - Modifies the input state object in place
            - Records token usage for the configured model
            - Adds WRITER node to the steps tracking list
        """

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
            state.steps.append(Node.WRITER)
            state.summary_exists = True
            state.content = copy.deepcopy(summary.content)

        if configurable.strip_thinking_tokens:
            state.content = strip_thinking_tokens(text=state.content)

        state.content = state.content.lstrip('\n')
        return state
