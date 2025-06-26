from typing import Any, Final
import json

from pydantic import BaseModel
from langchain_core.callbacks import get_usage_metadata_callback
from langchain.chat_models import init_chat_model

from ..enums import Node
from .utils import format_context_with_source_ids


CLAIMS_INSTRUCTIONS = """
You are an expert writer working on extracting the claims in a given summary about a topic using the provided context.

<Goal>
Extract the claims in the given summary using the provided context.
</Goal>

The topic of the summary:
<topic>
{topic}
</topic>

The summary you are going to work on:
<summary>
{summary}
</summary>

The context you are going to use (with source IDs):
<context>
{context}
</context>

<Format>
* Format your response as a JSON object with one field:    
    - claims: Claims you extract from the given summary.
* Each claim should have the following five fields:
    - text: Text of the claim taken from the summary.
    - source_ids: The ids of sources that support your claim (e.g. src1, src2, etc.).
    - confidence: Your confidence level in your claim (between 0 and 1, floating point).
    - start_position: The start position of your claim in the given summary.
    - end_position: The end position of your claim in the given summary.

Provide your analysis in JSON format:

{{    
    "claims": [
            {{
                "text": str
                "source_ids": List[str]
                "confidence": float
                "start_position": int
                "end_position": int
            }}
    ],    
}}
</Format>

<Claim Instructions>
WHAT TO MARK AS CLAIMS:
✓ Statistics, numbers, percentages, measurements
✓ Specific facts, dates, events, names
✓ Direct quotes or paraphrases from sources
✓ Research findings, study results, survey data
✓ Any statement that could be disputed or needs verification

WHAT NOT TO MARK:
✗ General background information or common knowledge
✗ Your own analysis or connecting statements between facts
✗ Obvious conclusions that don't need citation

CONFIDENCE LEVELS:
- 1.0: Directly quoted or exactly stated in source
- 0.9: Clearly supported with specific evidence in source
- 0.8: Well-supported by source data with minor interpretation
- 0.7: Reasonably inferred from source information
- 0.6: Somewhat supported but requires interpretation
- Below 0.6: Don't include the claim

EXAMPLE:
"topic": "Renewable energy sector in 2024"
"summary": "The renewable energy sector experienced significant growth in 2024. Solar energy capacity increased by 23% globally, while wind energy installations grew by 18%. This expansion was driven by increased government subsidies totaling $12 billion.",
"context": [src1]\nThe global solar energy capacity increased by 23% due to increased government subsidies reaching $12 billion\n[src2]\nThe government subsidies in renewable energy reached $12 billion, leading a 18% increase in the global wind energy installations." 

{{        
    "claims": [
            {{
                "text": "Solar energy capacity increased by 23% globally"
                "source_ids": [src1]
                "confidence": 0.95
                "start_position": 68
                "end_position": 115
            }},
            {{
                "text": "wind energy installations grew by 18%"
                "source_ids": [src2]
                "confidence": 0.88
                "start_position": 123
                "end_position": 160
            }},
            {{
                "text": "increased government subsidies totaling $12 billion"
                "source_ids": [src1, src2]
                "confidence": 0.80
                "start_position": 191
                "end_position": 242
            }}
    ],    
}}
</Claim Instructions>

<Task>
List every factual claim with proper source attribution, confidence, start position, end position.
</Task>
"""


class CitationsManager:
    def __init__(self, model_params: dict[str, Any], configuration_module_prefix: str):
        self.model_name = model_params['model']
        self.configuration_module_prefix: Final = configuration_module_prefix
        self.base_llm = init_chat_model(
            model=model_params['model'],
            model_provider=model_params['model_provider'],
            api_key=model_params['api_key'],
            **model_params['model_args']
        )

    def run(self, state: BaseModel):
        # Prepare context with source IDs for citation
        formatted_context = format_context_with_source_ids(state)
        instructions = CLAIMS_INSTRUCTIONS.format(topic = state.topic,
                                                  summary = state.content,
                                                  context = formatted_context)

        with get_usage_metadata_callback() as cb:
            results = self.base_llm.invoke(instructions, response_format = {"type": "json_object"})
            state.token_usage[self.model_name]['input_tokens'] += cb.usage_metadata[self.model_name]['input_tokens']
            state.token_usage[self.model_name]['output_tokens'] += cb.usage_metadata[self.model_name]['output_tokens']
            json_dict = json.loads(results.content)

        return state