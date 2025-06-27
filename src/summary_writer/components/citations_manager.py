from typing import Any, Final, Dict, List, Tuple
import json

from pydantic import BaseModel
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import get_usage_metadata_callback
from langchain.chat_models import init_chat_model

from ai_common import get_config_from_runnable, strip_thinking_tokens
# from ..enums import Node
from ..state import WriterClaim


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

BIBLIOGRAPHY_GENERATION_INSTRUCTIONS = """
Generate a bibliography/reference list for the following sources used in the summary.

Sources:
<sources>
{sources}
</sources>

Citation style: {citation_style}

<Requirements>
1. Format each source according to the citation style
2. Order sources as appropriate for the style:
   - numeric: In order of first appearance [1], [2], [3]
   - author-year: Alphabetically by author's last name
   - footnote: In order of first appearance
3. Include all available metadata for each source
4. Use consistent formatting throughout
</Requirements>

<Output>
Provide only the formatted bibliography, one source per line.
</Output>
"""


def format_sources_for_extraction(sources: Dict[str, Any]) -> str:
    """
    Format sources with IDs for claim extraction.
    """
    formatted_sources = []
    for source_id, source_data in sources.items():
        source_text = f"[Source ID: {source_id}]\n"
        source_text += f"Title: {source_data.get('title', 'Unknown')}\n"
        source_text += f"Content: {source_data.get('content', '')}\n"
        formatted_sources.append(source_text)

    return "\n\n".join(formatted_sources)

def format_citation(citation_nums: List[int], citation_style: str) -> str:
    """
    Format citation numbers based on citation style.
    """
    if citation_style == "numeric":
        if len(citation_nums) == 1:
            return f"[{citation_nums[0]}]"
        else:
            return f"[{','.join(map(str, sorted(citation_nums)))}]"
    elif citation_style == "footnote":
        superscript_map = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
        if len(citation_nums) == 1:
            return str(citation_nums[0]).translate(superscript_map)
        else:
            return ','.join(str(num).translate(superscript_map) for num in sorted(citation_nums))
    else:  # author-year style would need author information
        return f"[{','.join(map(str, sorted(citation_nums)))}]"


def add_citations_to_content(content: str,
                             claims: List[WriterClaim],
                             citation_style: str) -> Tuple[str, Dict[str, int]]:

    """
    Add inline citations to content based on extracted claims.

    Returns:
        Tuple of (cited_content, citation_mapping)
    """

    citation_mapping = {}
    next_citation_num = 1

    # Sort claims by position to process in order
    sorted_claims = sorted(claims, key=lambda c: c.start_position)

    # Build cited content
    cited_content = content
    offset = 0

    for claim in sorted_claims:
        # Get citation numbers for this claim's sources
        citation_nums = []
        for source_id in claim.source_ids:
            if source_id not in citation_mapping:
                citation_mapping[source_id] = next_citation_num
                next_citation_num += 1
            citation_nums.append(citation_mapping[source_id])

        # Format citation based on style
        citation = format_citation(citation_nums=citation_nums, citation_style=citation_style)

        # Insert citation after the claim
        insert_pos = claim.end_position + offset
        cited_content = cited_content[:insert_pos] + citation + cited_content[insert_pos:]
        offset += len(citation)

    return cited_content, citation_mapping

def format_source_for_bibliography(source_data: Dict[str, Any], citation_num: int) -> str:
    """
    Format a source for bibliography generation.
    """
    parts = [f"[{citation_num}]"]

    if 'author' in source_data:
        parts.append(f"Author: {source_data['author']}")
    if 'title' in source_data:
        parts.append(f"Title: {source_data['title']}")
    if 'url' in source_data:
        parts.append(f"URL: {source_data['url']}")
    if 'date' in source_data:
        parts.append(f"Date: {source_data['date']}")

    return " | ".join(parts)


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

    def run(self, state: BaseModel, config: RunnableConfig):

        configurable = get_config_from_runnable(
            configuration_module_prefix=self.configuration_module_prefix,
            config=config
        )

        all_sources = {}
        for sources_dict in state.cumulative_unique_sources:  # Exclude current sources
            all_sources.update(sources_dict)
        all_sources = {f"src_{i}": {'title': v['title'], 'content': v['content'], 'url': k} for i, (k, v) in enumerate(all_sources.items(), start=1)}

        claims = self.extract_claims(state=state,
                                     sources=all_sources,
                                     min_claim_confidence=configurable.min_claim_confidence)

        cited_content, citation_mapping = add_citations_to_content(content = state.content,
                                                                   claims = claims,
                                                                   citation_style = configurable.citation_style)

        state.bibliography = self.generate_bibliography(sources=all_sources,
                                                        citation_mapping=citation_mapping,
                                                        citation_style=configurable.citation_style,
                                                        remove_thinking_tokens=configurable.strip_thinking_tokens)


        return state

    def extract_claims(self, state: BaseModel, sources: dict[str, Any], min_claim_confidence: float) -> list[WriterClaim]:
        """
        Extract claims from content with their supporting sources.
        """

        # Prepare context with source IDs for citation
        source_context = format_sources_for_extraction(sources=sources)
        instructions = CLAIMS_INSTRUCTIONS.format(topic=state.topic,
                                                  summary=state.content,
                                                  context=source_context)

        with get_usage_metadata_callback() as cb:
            results = self.base_llm.invoke(instructions, response_format={"type": "json_object"})
            state.token_usage[self.model_name]['input_tokens'] += cb.usage_metadata[self.model_name]['input_tokens']
            state.token_usage[self.model_name]['output_tokens'] += cb.usage_metadata[self.model_name]['output_tokens']
            json_dict = json.loads(results.content)


        # Parse the JSON response
        try:
            claims_data = json_dict['claims']

            claims = []
            for claim_data in claims_data:
                if claim_data['confidence'] >= min_claim_confidence:
                    claim_text = claim_data['text'].strip()

                    # Find the actual position of this claim in the content
                    start_pos = state.content.find(claim_text)
                    if start_pos == -1:
                        # Try finding a substring if exact match fails
                        # This handles cases where LLM slightly modified the text
                        words = claim_text.split()
                        if len(words) > 3:
                            # Try with first few words
                            partial_text = ' '.join(words[:3])
                            start_pos = state.content.find(partial_text)
                            if start_pos != -1:
                                # Find the end of the actual claim
                                # Look for sentence ending punctuation
                                end_search_start = start_pos + len(partial_text)
                                end_markers = ['. ', '.\n', '? ', '?\n', '! ', '!\n']
                                end_pos = len(state.content)
                                for marker in end_markers:
                                    marker_pos = state.content.find(marker, end_search_start)
                                    if marker_pos != -1 and marker_pos < end_pos:
                                        end_pos = marker_pos + 1
                                claim_text = state.content[start_pos:end_pos].strip()

                    if start_pos != -1:
                        end_pos = start_pos + len(claim_text)

                        claim = WriterClaim(
                            text=claim_text,
                            source_ids=claim_data['source_ids'],
                            confidence=claim_data['confidence'],
                            start_position=start_pos,
                            end_position=end_pos
                        )
                        claims.append(claim)
                    else:
                        # Skip claims we can't locate in the content
                        print(f"Warning: Could not locate claim in content: {claim_text[:50]}...")

            return claims
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing claims: {e}")
            return []

    def generate_bibliography(self,
                              sources: Dict[str, Any],
                              citation_mapping: Dict[str, int],
                              citation_style: str,
                              remove_thinking_tokens: bool) -> str:
        """
        Generate a bibliography from sources and citation mapping.
        """
        # Sort sources by citation number
        sorted_sources = sorted(
            [(source_id, sources[source_id], num) for source_id, num in citation_mapping.items()],
            key=lambda x: x[2]
        )

        # Format sources for bibliography
        formatted_sources = []
        for source_id, source_data, citation_num in sorted_sources:
            formatted_source = format_source_for_bibliography(source_data, citation_num)
            formatted_sources.append(formatted_source)

        instructions = BIBLIOGRAPHY_GENERATION_INSTRUCTIONS.format(
            sources="\n".join(formatted_sources),
            citation_style=citation_style
        )

        response = self.base_llm.invoke(instructions)

        bibliography = response.content
        if remove_thinking_tokens:
            bibliography = strip_thinking_tokens(text=bibliography)
        bibliography = bibliography.lstrip('\n')

        return bibliography

