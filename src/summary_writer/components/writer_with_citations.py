import copy
import re
from typing import Any, Dict, List, Tuple
from collections import defaultdict
import json

from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import get_usage_metadata_callback
from langchain.chat_models import init_chat_model
from ai_common import strip_thinking_tokens, get_config_from_runnable

from ..enums import Node
from ..state import SummaryState, WriterClaim

CLAIM_EXTRACTION_INSTRUCTIONS = """
You are an expert at analyzing text written about a topic and extracting specific claims with their sources.

<Goal>
Extract all factual claims from the provided text and identify which sources support each claim.
</Goal>

The topic of the text:
<topic>
{topic}
</topic>

The text you are analyzing:
<text>
{text}
</text>

The context with sources:
<context>
{context}
</context>

<Requirements>
1. Extract each distinct factual claim from the text
2. For each claim, identify ALL source IDs that support it
3. Assess confidence level (0.0-1.0) based on:
   - How directly the source supports the claim
   - Number of supporting sources
   - Clarity and specificity of the information
4. Record the position of each claim in the text
5. Only extract claims that are directly supported by the sources
</Requirements>

<Output Format>
* Format your response as a JSON object with one field:    
    - claims: Claims you extract from the given text.
* Each claim should have the following five fields:
    - "text": The claim text
    - "source_ids": Array of source IDs that support this claim
    - "confidence": Confidence score between 0.0 and 1.0
    - "start_position": Character position where claim starts
    - "end_position": Character position where claim ends

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

Example:
[
  {{
    "text": "Climate change is primarily caused by human activities",
    "source_ids": ["source_1", "source_3"],
    "confidence": 0.9,
    "start_position": 0,
    "end_position": 52
  }}
]
</Output Format>
"""

WRITING_WITH_CITATIONS_INSTRUCTIONS = """
You are an expert writer working on writing a summary about a given topic using the provided context.

<Goal>
Write a high-quality summary of the provided context.
</Goal>

The topic you are writing about:
<topic>
{topic}
</topic>

The context you are going to use:
<context>
{context}
</context>

Citation style: {citation_style}

<Requirements>
1. Include ALL relevant information from each source
2. Add inline citations immediately after each claim or statement
3. Use the specified citation style:
   - numeric: [1], [2], [3] or [1,3,5] for multiple sources
   - author-year: (Smith, 2023), (Jones & Brown, 2022)
   - footnote: ¹, ², ³
4. Ensure every factual statement has a citation
5. Group related information logically
6. Maintain coherent flow between paragraphs
7. Be comprehensive but concise
</Requirements>

<Formatting>
- Start directly with the summary content
- Place citations immediately after the relevant claim
- Use consistent citation format throughout
- No XML tags in the output
</Formatting>
"""

EXTENDING_WITH_CITATIONS_INSTRUCTIONS = """
You are an expert writer extending an existing summary with new information.

<Goal>
Extend the existing summary with new search results.
</Goal>

The topic of the summary:
<topic>
{topic}
</topic>

The existing summary with citations:
<summary>
{summary}
</summary>

The new search results:
<search_results>
{search_results}
</search_results>

Citation style: {citation_style}
Next citation number: {next_citation_number}

<Requirements>
1. Integrate new information seamlessly into existing content
2. Continue citation numbering from where the existing summary left off
3. Add citations for all new claims
4. Maintain the same citation style as the existing summary
5. Update existing sections if new information provides better context
6. Ensure no duplicate information
7. Preserve the logical flow and structure
</Requirements>

<Formatting>
- Start directly with the updated summary
- Maintain consistent citation format
- No XML tags in the output
</Formatting>
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


class WriterWithCitations:
    """
    Writer that extracts claims, adds inline citations, and generates bibliographies.
    """

    def __init__(self, model_params: dict[str, Any], configuration_module_prefix: str,
                 citation_style: str = "numeric", min_confidence: float = 0.6):
        """
        Initialize the WriterWithCitations.

        Args:
            model_params: Model configuration parameters
            configuration_module_prefix: Configuration module prefix
            citation_style: Citation style ("numeric", "author-year", "footnote")
            min_confidence: Minimum confidence threshold for including citations
        """
        self.model_name = model_params['model']
        self.configuration_module_prefix = configuration_module_prefix
        self.citation_style = citation_style
        self.min_confidence = min_confidence

        # Initialize the LLM
        self.writer_llm = init_chat_model(
            model=model_params['model'],
            model_provider=model_params['model_provider'],
            api_key=model_params['api_key'],
            **model_params['model_args']
        )

    def extract_claims(self, content: str, sources: Dict[str, Any], topic: str) -> List[WriterClaim]:
        """
        Extract claims from content with their supporting sources.
        """
        # Format sources for extraction
        source_context = self._format_sources_for_extraction(sources)

        instructions = CLAIM_EXTRACTION_INSTRUCTIONS.format(
            topic=topic,
            text=content,
            context=source_context
        )

        response = self.writer_llm.invoke(instructions, response_format = {"type": "json_object"})

        # Parse the JSON response
        try:
            claims_data = json.loads(response.content)['claims']

            claims = []
            for claim_data in claims_data:
                if claim_data['confidence'] >= self.min_confidence:
                    claim_text = claim_data['text'].strip()

                    # Find the actual position of this claim in the content
                    start_pos = content.find(claim_text)
                    if start_pos == -1:
                        # Try finding a substring if exact match fails
                        # This handles cases where LLM slightly modified the text
                        words = claim_text.split()
                        if len(words) > 3:
                            # Try with first few words
                            partial_text = ' '.join(words[:3])
                            start_pos = content.find(partial_text)
                            if start_pos != -1:
                                # Find the end of the actual claim
                                # Look for sentence ending punctuation
                                end_search_start = start_pos + len(partial_text)
                                end_markers = ['. ', '.\n', '? ', '?\n', '! ', '!\n']
                                end_pos = len(content)
                                for marker in end_markers:
                                    marker_pos = content.find(marker, end_search_start)
                                    if marker_pos != -1 and marker_pos < end_pos:
                                        end_pos = marker_pos + 1
                                claim_text = content[start_pos:end_pos].strip()

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

    def add_citations_to_content(self, content: str, claims: List[WriterClaim],
                                 existing_citations: Dict[str, int] = None) -> Tuple[str, Dict[str, int]]:
        """
        Add inline citations to content based on extracted claims.

        Returns:
            Tuple of (cited_content, citation_mapping)
        """
        if existing_citations is None:
            existing_citations = {}

        citation_mapping = existing_citations.copy()
        next_citation_num = max(citation_mapping.values()) + 1 if citation_mapping else 1

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
            citation = self._format_citation(citation_nums)

            # Insert citation after the claim
            insert_pos = claim.end_position + offset
            cited_content = cited_content[:insert_pos] + citation + cited_content[insert_pos:]
            offset += len(citation)

        return cited_content, citation_mapping

    def generate_bibliography(self, sources: Dict[str, Any], citation_mapping: Dict[str, int], remove_thinking_tokens: bool) -> str:
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
            formatted_source = self._format_source_for_bibliography(source_data, citation_num)
            formatted_sources.append(formatted_source)

        instructions = BIBLIOGRAPHY_GENERATION_INSTRUCTIONS.format(
            sources="\n".join(formatted_sources),
            citation_style=self.citation_style
        )

        response = self.writer_llm.invoke(instructions)

        bibliography = response.content
        if remove_thinking_tokens:
            bibliography = strip_thinking_tokens(text=bibliography)
        bibliography = bibliography.lstrip('\n')

        return bibliography

    def _format_sources_for_extraction(self, sources: Dict[str, Any]) -> str:
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

    def _format_citation(self, citation_nums: List[int]) -> str:
        """
        Format citation numbers based on citation style.
        """
        if self.citation_style == "numeric":
            if len(citation_nums) == 1:
                return f"[{citation_nums[0]}]"
            else:
                return f"[{','.join(map(str, sorted(citation_nums)))}]"
        elif self.citation_style == "footnote":
            superscript_map = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
            if len(citation_nums) == 1:
                return str(citation_nums[0]).translate(superscript_map)
            else:
                return ','.join(str(num).translate(superscript_map) for num in sorted(citation_nums))
        else:  # author-year style would need author information
            return f"[{','.join(map(str, sorted(citation_nums)))}]"

    def _format_source_for_bibliography(self, source_data: Dict[str, Any], citation_num: int) -> str:
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

    def _get_next_citation_number(self, cited_content: str) -> int:
        """
        Extract the highest citation number from existing content.
        """
        if self.citation_style == "numeric":
            # Find all [n] patterns
            matches = re.findall(r'\[(\d+)\]', cited_content)
            if matches:
                return max(int(m) for m in matches) + 1
        elif self.citation_style == "footnote":
            # Find all superscript numbers
            superscript_digits = "⁰¹²³⁴⁵⁶⁷⁸⁹"
            normal_digits = "0123456789"
            trans_table = str.maketrans(superscript_digits, normal_digits)

            # Find all sequences of superscript digits
            matches = re.findall(f'[{superscript_digits}]+', cited_content)
            if matches:
                numbers = [int(m.translate(trans_table)) for m in matches]
                return max(numbers) + 1

        return 1

    def run(self, state: SummaryState, config: RunnableConfig) -> SummaryState:
        """
        Run the writer with citations.
        """

        configurable = get_config_from_runnable(
            configuration_module_prefix=self.configuration_module_prefix,
            config=config
        )

        with get_usage_metadata_callback() as cb:
            if state.summary_exists:
                # Extending existing summary
                next_citation_num = self._get_next_citation_number(state.cited_content or state.content)

                instructions = EXTENDING_WITH_CITATIONS_INSTRUCTIONS.format(
                    topic=state.topic,
                    summary=state.cited_content or state.content,
                    search_results=state.source_str,
                    citation_style=self.citation_style,
                    next_citation_number=next_citation_num
                )
            else:
                # Creating new summary with citations
                instructions = WRITING_WITH_CITATIONS_INSTRUCTIONS.format(
                    topic=state.topic,
                    context=state.source_str,
                    citation_style=self.citation_style
                )

            response = self.writer_llm.invoke(instructions)
            new_content = response.content

            if configurable.strip_thinking_tokens:
                new_content = strip_thinking_tokens(text=new_content)
            new_content = new_content.lstrip('\n')

            # Extract claims from the new content
            claims = self.extract_claims(
                new_content,
                state.unique_sources,
                state.topic
            )
            state.claims.extend(claims)
            #######################################
            # Get existing citation mapping if we're extending
            existing_citations = {}
            if state.summary_exists and state.cited_content:
                # Build mapping from existing cited content
                all_previous_sources = {}
                for sources_dict in state.cumulative_unique_sources[:-1]:  # Exclude current sources
                    all_previous_sources.update(sources_dict)

                for i, source_id in enumerate(all_previous_sources.keys(), 1):
                    existing_citations[source_id] = i

            # Add citations to the new content based on extracted claims
            cited_new_content, citation_mapping = self.add_citations_to_content(
                new_content,
                claims,
                existing_citations
            )

            # Update the full cited content
            if state.summary_exists:
                state.cited_content = state.cited_content + "\n\n" + cited_new_content
            else:
                state.cited_content = cited_new_content

            # Update content without citations (for backward compatibility)
            if state.summary_exists:
                state.content = state.content + "\n\n" + new_content
            else:
                state.content = new_content

            # Generate bibliography with all sources
            all_sources = {}
            for sources_dict in state.cumulative_unique_sources:
                all_sources.update(sources_dict)

            all_sources = {k: {'title': v['title'], 'content': v['content'], 'url': k} for k, v in all_sources.items()}
            state.bibliography = self.generate_bibliography(all_sources, citation_mapping, configurable.strip_thinking_tokens)

            ########################################
            """
            # Generate bibliography
            all_sources = {}
            for sources_dict in state.cumulative_unique_sources:
                all_sources.update(sources_dict)

            all_sources = {k:{'title': v['title'], 'content': v['content'], 'url': k}  for k, v in all_sources.items()}

            # Build citation mapping from cited content
            citation_mapping = {}
            for i, source_id in enumerate(all_sources.keys(), 1):
                citation_mapping[source_id] = i

            state.bibliography = self.generate_bibliography(all_sources, citation_mapping, configurable.strip_thinking_tokens)

            # Update content (for backward compatibility)
            state.content = state.cited_content
            """

            # Update token usage
            if self.model_name not in state.token_usage:
                state.token_usage[self.model_name] = {'input_tokens': 0, 'output_tokens': 0}

            state.token_usage[self.model_name]['input_tokens'] += cb.usage_metadata.get(self.model_name, {}).get(
                'input_tokens', 0)
            state.token_usage[self.model_name]['output_tokens'] += cb.usage_metadata.get(self.model_name, {}).get(
                'output_tokens', 0)

            state.steps.append(Node.WRITER)
            state.summary_exists = True

        if configurable.strip_thinking_tokens:
            state.content = strip_thinking_tokens(text=state.content)
            state.cited_content = strip_thinking_tokens(text=state.cited_content)

        state.content = state.content.lstrip('\n')
        state.cited_content = state.cited_content.lstrip('\n')

        return state