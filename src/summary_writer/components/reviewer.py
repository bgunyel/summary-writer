import datetime
from typing import Any, Final
import json

from pydantic import BaseModel
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import get_usage_metadata_callback
from langchain.chat_models import init_chat_model
from ai_common import get_config_from_runnable, SearchQuery

from ..enums import Node


REVIEW_INSTRUCTIONS = """
You are an expert researcher analyzing a summary about a given topic.

<Goal>
1. Identify knowledge gaps or areas that need deeper exploration in the summary.
2. Focus on details and key insights about the topic that weren't fully covered.
3. Generate follow-up questions that would help expand your understanding.
4. Generate stand-alone web queries from the follow-up questions.
</Goal>

The topic of the summary is:
<topic>
{topic}
</topic>

The already written content of the summary:
<context>
{context}
</context>

Today's date is:
<today>
{today}
</today>

<Requirements>
* Ensure the follow-up questions are self-contained and include necessary context for web search.
* Convert each follow-up question to a stand-alone web search query.
</Requirements>

<Format>
Format your response as a JSON object with four fields:
- reasoning: All the reasoning you do.
- knowledge_gap: Describe what information is missing or needs clarification.
- follow-up questions: Write specific questions to address this gap.
- queries: Convert the follow-up questions to stand-alone web search queries.

Provide your analysis in JSON format:

{{
    "reasoning": "string"
    "knowledge_gap": "string",
    "follow-up questions": [
            {{
                "question": "string",                
            }}
    ]
    "queries": [
            {{
                "query": "string",                
            }}
    ]
}}
</Format>

<Task>
Think carefully about the provided summary context first.
Then, identify the knowledge gaps, produce follow-up questions and generate stand-alone web search queries.
</Task>
"""

class Reviewer:
    def __init__(self, model_params: dict[str, Any], configuration_module_prefix: str):
        self.model_name = model_params['model']
        self.configuration_module_prefix: Final = configuration_module_prefix
        self.base_llm = init_chat_model(
            model=model_params['model'],
            model_provider=model_params['model_provider'],
            api_key=model_params['api_key'],
            **model_params['model_args']
        )

    def run(self, state: BaseModel, config: RunnableConfig) -> BaseModel:
        """
        Execute the review process to analyze existing summary content and generate follow-up search queries.
        
        This method performs a critical analysis of the current summary to identify knowledge gaps,
        formulate follow-up questions, and convert them into web search queries for iterative research.
        It tracks the review process and manages cumulative data across multiple iterations.
        
        Args:
            state (BaseModel): The current state object containing:
                - topic: The subject matter being researched
                - content: Current summary content to be reviewed
                - steps: List tracking workflow steps completed
                - unique_sources: Sources used in current iteration
                - search_queries: Will be populated with new search queries
                - cumulative_unique_sources: Historical source tracking across iterations
                - cumulative_search_queries: Historical query tracking across iterations
                - iteration: Current iteration counter
                - token_usage: Dictionary tracking token consumption by model
            config (RunnableConfig): Configuration object containing runnable parameters
                and configurable options for the review process
        
        Returns:
            BaseModel: Updated state object with:
                - search_queries: New SearchQuery objects for follow-up research
                - steps: Appended with Node.REVIEWER to track completion
                - iteration: Incremented iteration counter
                - cumulative_unique_sources: Updated with current iteration's sources
                - cumulative_search_queries: Updated with current iteration's queries
                - token_usage: Updated with tokens consumed during this operation
        
        Process Flow:
            1. Extracts configuration parameters from the runnable config
            2. Updates state tracking (steps, cumulative data, iteration counter)
            3. Formats review instructions with current topic, content, and date
            4. Invokes LLM with structured JSON response format to analyze content
            5. Captures and records token usage metadata
            6. Parses JSON response to extract generated search queries
            7. Converts query strings to SearchQuery objects for downstream use
        
        Side Effects:
            - Modifies the input state object in place
            - Records token usage for the configured model
            - Adds REVIEWER node to the steps tracking list
            - Increments iteration counter
            - Updates cumulative tracking lists
        
        Expected JSON Response Format:
            {
                "reasoning": "Analysis of current content",
                "knowledge_gap": "Description of missing information",
                "follow-up questions": [{"question": "string"}],
                "queries": [{"query": "string"}]
            }
        """
        configurable = get_config_from_runnable(
            configuration_module_prefix=self.configuration_module_prefix,
            config=config
        )

        state.steps.append(Node.REVIEWER)
        state.iteration += 1

        instructions = REVIEW_INSTRUCTIONS.format(topic=state.topic,
                                                  context=state.content,
                                                  today=datetime.date.today().isoformat())

        with get_usage_metadata_callback() as cb:
            results = self.base_llm.invoke(instructions, response_format = {"type": "json_object"})
            state.token_usage[self.model_name]['input_tokens'] += cb.usage_metadata[self.model_name]['input_tokens']
            state.token_usage[self.model_name]['output_tokens'] += cb.usage_metadata[self.model_name]['output_tokens']
            json_dict = json.loads(results.content)
            state.search_queries = [SearchQuery(search_query=q['query']) for q in json_dict['queries']]

        return state
