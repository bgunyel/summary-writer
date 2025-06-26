import copy
import re
from typing import Any, List, Tuple

from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import get_usage_metadata_callback
from langchain.chat_models import init_chat_model
from ai_common import strip_thinking_tokens, get_config_from_runnable

from ..enums import Node
from ..state import SummaryState, WriterClaim


# Enhanced prompts with citation instructions
WRITING_INSTRUCTIONS_WITH_CITATIONS = """
You are an expert writer working on writing a summary about a given topic using the provided context.

<Goal>
Write a high quality summary of the provided context with precise source attribution.
</Goal>

The topic you are writing about:
<topic>
{topic}
</topic>

The context you are going to use (with source IDs):
<context>
{context}
</context>

<Requirements>
1. Highlight every relevant information from each source.
2. Provide a detailed overview of the key points related to the topic.
3. Emphasize significant findings or insights.
4. Ensure a coherent flow of information.
5. CRITICAL: Mark every factual claim with proper source attribution.
</Requirements>

<Format>
* Format your response as a JSON object with two fields:
    - summary: The summary that is generated about the topic using the given context. Start directly with the summary, without preamble or titles.
    - claims: Claims in your summary.
* Each claim should have the following five fileds:
    - text: Text of the claim taken from the summary.
    - source_ids: The ids of sources that support your claim (e.g. src1, src2, etc.).
    - confidence: Your confidence level in your claim (between 0 and 1, floating point)
    - start_position: The start position of your claim in the generated summary.
    - end_position: The end position of your claim in the generated summary.

Provide your analysis in JSON format:

{{
    "summary": "string",
    "claims": [
            {{
                "text": str
                "source_ids": List[str]
                "confidence": str
                "start_position": str
                "end_position": str
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
{{
    "summary": "The renewable energy sector experienced significant growth in 2024. Solar energy capacity increased by 23% globally, while wind energy installations grew by 18%. This expansion was driven by increased government subsidies totaling $12 billion.",
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
                "source_ids": List[src1, src2]
                "confidence": 0.80
                "start_position": 191
                "end_position": 242
            }}
    ],    
}}
</Claim Instructions>

<Task>
Think carefully about the provided context first. 
Then write a comprehensive summary using the provided context. 
List every factual claim with proper source attribution, confidence, start position, end position.
</Task>
"""

EXTENDING_INSTRUCTIONS_WITH_CITATIONS = """
You are an expert writer working on extending summary with new search results while maintaining precise source attribution.

<Goal>
Extend the existing summary with new search results, adding proper citations for all new factual claims.
</Goal>

The topic of the summary:
<topic>
{topic}
</topic>

The existing summary (may already contain citations):
<summary>
{summary}
</summary>

The new search results (with source IDs):
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
5. Mark ALL new factual claims with proper source attribution.
6. Preserve existing citations in the summary.
7. Verify that your final output differs from the input summary.
</Requirements>

<Citation Instructions>
For every NEW factual statement you add, use this exact format:
<CLAIM sources="source_id1,source_id2" confidence="0.0-1.0">"your factual claim here"</CLAIM>

Apply the same citation rules as in the initial writing phase.
Keep existing <CLAIM> tags unchanged - only add new ones for new information.
</Citation Instructions>

<Formatting>
- Start directly with the updated summary, without preamble or titles.
</Formatting>

<Task>
Think carefully about the provided search results first. Then update the existing summary accordingly, adding proper citations for all new factual claims.
</Task>
"""


class WriterWithCitations:
    """Enhanced Writer class that generates summaries with claim-level citations"""

    def __init__(self, model_params: dict[str, Any], configuration_module_prefix: str,
                 citation_style: str = "numeric", min_confidence: float = 0.6):
        self.model_name = model_params['model']
        self.configuration_module_prefix = configuration_module_prefix
        self.citation_style = citation_style
        self.min_confidence = min_confidence

        self.writer_llm = init_chat_model(
            model=model_params['model'],
            model_provider=model_params['model_provider'],
            api_key=model_params['api_key'],
            **model_params['model_args']
        )

        # Pattern for extracting CLAIM tags
        self.claim_pattern = r'<CLAIM\s+sources="([^"]*?)"\s+confidence="([^"]*?)">(.*?)</CLAIM>'

    def run(self, state: SummaryState, config: RunnableConfig) -> SummaryState:
        """
        Enhanced version of the Writer run method with citation support.

        This method extends the original functionality to:
        1. Generate summaries with embedded claim markers
        2. Extract and validate claims from the LLM response
        3. Produce both clean and cited versions of the summary
        4. Track citation metadata in the state

        Args:
            state (SummaryState): Enhanced state that should include:
                - All original fields from your SummaryState
                - Optional: claims (List[WriterClaim]) - for tracking claims
                - Optional: cited_content (str) - summary with inline citations
                - Optional: bibliography (str) - reference list
            config (RunnableConfig): Same as original

        Returns:
            SummaryState: Enhanced state with citation data
        """

        configurable = get_config_from_runnable(
            configuration_module_prefix=self.configuration_module_prefix,
            config=config
        )

        # Prepare context with source IDs for citation
        formatted_context = self._format_context_with_source_ids(state)

        # Select appropriate instruction template
        if state.summary_exists:  # Extending existing summary
            instructions = EXTENDING_INSTRUCTIONS_WITH_CITATIONS.format(
                topic=state.topic,
                summary=state.content,
                search_results=formatted_context
            )
        else:  # Writing a new summary
            instructions = WRITING_INSTRUCTIONS_WITH_CITATIONS.format(
                topic=state.topic,
                context=formatted_context
            )

        # Call LLM with citation-enhanced instructions
        with get_usage_metadata_callback() as cb:
            summary_response = self.writer_llm.invoke(instructions)

            # Update token usage
            state.token_usage[self.model_name]['input_tokens'] += cb.usage_metadata[self.model_name]['input_tokens']
            state.token_usage[self.model_name]['output_tokens'] += cb.usage_metadata[self.model_name]['output_tokens']

            # Track step completion
            state.steps.append(Node.WRITER)
            state.summary_exists = True

            # Get raw content
            raw_content = copy.deepcopy(summary_response.content)

        # Strip thinking tokens if configured
        if configurable.strip_thinking_tokens:
            raw_content = strip_thinking_tokens(text=raw_content)

        raw_content = raw_content.lstrip('\n')

        # Process citations
        clean_content, claims, cited_content = self._process_citations(raw_content, state)

        # Update state with both versions
        state.content = clean_content  # Original behavior - clean summary

        # Add citation-enhanced fields
        state.cited_content = cited_content
        state.claims.extend(claims)
        state.bibliography = self._generate_bibliography(state)
        return state

    def _format_context_with_source_ids(self, state: SummaryState) -> str:
        """
        Format the context to include source IDs for citation.

        This assumes your state.source_str contains source information.
        You may need to adapt this based on your actual data structure.
        """

        number_of_previous_sources = sum(
            [len(state.cumulative_unique_sources[i]) for i in range(state.iteration)]
        ) if state.iteration > 0 else 0

        formatted_sources = []
        for i, (k, v) in enumerate(iterable=state.unique_sources.items(), start=number_of_previous_sources+1):
            source_id = f"src{i}"
            title = v['title']
            content = v['content']
            url = k
            formatted_sources.append(f"[{source_id}] {title}\nURL: {url}\nContent: {content}")

        return "\n".join(formatted_sources)



    def _process_citations(self, raw_content: str, state: SummaryState) -> Tuple[str, List[WriterClaim], str]:
        """
        Process the LLM response to extract claims and generate both clean and cited versions.

        Returns:
            Tuple of (clean_content, extracted_claims, cited_content)
        """

        # Extract claims from the raw content
        claims = self._extract_claims(raw_content)

        # Validate claims (check confidence, source existence, etc.)
        validated_claims = self._validate_claims(claims, state)

        # Generate clean content (remove CLAIM tags)
        clean_content = self._generate_clean_content(raw_content)

        # Generate cited content (replace CLAIM tags with inline citations)
        cited_content = self._generate_cited_content(raw_content, validated_claims)

        return clean_content, validated_claims, cited_content

    def _extract_claims(self, content: str) -> List[WriterClaim]:
        """Extract claims from content containing CLAIM tags"""
        claims = []

        for match in re.finditer(self.claim_pattern, content, re.DOTALL):
            sources_str = match.group(1)
            confidence_str = match.group(2)
            claim_text = match.group(3).strip()

            # Parse source IDs
            source_ids = [s.strip() for s in sources_str.split(',') if s.strip()]

            # Parse confidence
            try:
                confidence = float(confidence_str)
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                confidence = 0.5

            claims.append(WriterClaim(
                text=claim_text,
                source_ids=source_ids,
                confidence=confidence,
                start_position=match.start(),
                end_position=match.end()
            ))

        return claims

    def _validate_claims(self, claims: List[WriterClaim], state: SummaryState) -> List[WriterClaim]:
        """Validate claims against confidence threshold and source availability"""
        validated_claims = []

        # Get available source IDs (you may need to adapt this)
        available_sources = self._get_available_source_ids(state)

        for claim in claims:
            # Check confidence threshold
            if claim.confidence < self.min_confidence:
                continue

            # Check if sources exist
            valid_source_ids = [sid for sid in claim.source_ids if sid in available_sources]
            if not valid_source_ids:
                continue

            # Update claim with valid sources only
            claim.source_ids = valid_source_ids
            validated_claims.append(claim)

        return validated_claims

    def _get_available_source_ids(self, state: SummaryState) -> set:
        """Get set of available source IDs from the state"""
        # This is a heuristic - adapt based on your actual data structure

        if hasattr(state, 'sources') and state.sources:
            return {f"src{i + 1}" for i in range(len(state.sources))}
        else:
            # Estimate from source_str - count likely sources
            source_count = len([s for s in state.source_str.split('\n\n') if s.strip()])
            return {f"src{i + 1}" for i in range(source_count)}

    def _generate_clean_content(self, raw_content: str) -> str:
        """Generate clean content by removing CLAIM tags"""
        clean_content = re.sub(self.claim_pattern, r'\3', raw_content, flags=re.DOTALL)
        return re.sub(r'\s+', ' ', clean_content).strip()

    def _generate_cited_content(self, raw_content: str, validated_claims: List[WriterClaim]) -> str:
        """Generate content with inline citations"""
        cited_content = raw_content

        # Process in reverse order to maintain text positions
        matches = list(re.finditer(self.claim_pattern, raw_content, re.DOTALL))

        for match in reversed(matches):
            claim_text = match.group(3).strip()
            sources_str = match.group(1)
            source_ids = [s.strip() for s in sources_str.split(',') if s.strip()]

            # Find corresponding validated claim
            validated_claim = None
            for claim in validated_claims:
                if claim.text == claim_text:
                    validated_claim = claim
                    break

            if validated_claim:
                # Replace with claim + citation
                citation = self._format_citation(validated_claim.source_ids)
                replacement = claim_text + citation
            else:
                # Claim was filtered out, just use the text
                replacement = claim_text

            cited_content = cited_content[:match.start()] + replacement + cited_content[match.end():]

        return re.sub(r'\s+', ' ', cited_content).strip()

    def _format_citation(self, source_ids: List[str]) -> str:
        """Format inline citation based on style"""
        if not source_ids:
            return ""

        if self.citation_style == "numeric":
            return f" [{','.join(source_ids)}]"
        elif self.citation_style == "superscript":
            superscript_map = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵',
                               '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'}
            result = ""
            for sid in source_ids:
                superscript = ''.join(superscript_map.get(c, c) for c in sid)
                result += superscript
            return result
        else:
            return f" [{', '.join(source_ids)}]"

    def _generate_bibliography(self, state: SummaryState) -> str:
        """Generate bibliography from claims and sources"""
        if not hasattr(state, 'claims') or not state.claims:
            return ""

        # Collect used source IDs
        used_sources = set()
        for claim in state.claims:
            used_sources.update(claim.source_ids)

        # Generate bibliography entries
        bibliography_entries = []

        # If you have access to source metadata
        if hasattr(state, 'sources') and state.sources:
            for i, source in enumerate(state.sources):
                source_id = f"src{i + 1}"
                if source_id in used_sources:
                    title = getattr(source, 'title', 'Untitled')
                    url = getattr(source, 'url', '')
                    entry = f"[{source_id}] {title}. {url}"
                    bibliography_entries.append(entry)
        else:
            # Fallback - basic source list
            for source_id in sorted(used_sources):
                bibliography_entries.append(f"[{source_id}] Source {source_id}")

        return "\n".join(bibliography_entries)
