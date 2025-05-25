from ai_common import ConfigurationBase, TavilySearchCategory


DEFAULT_REPORT_STRUCTURE = """The report structure should focus on breaking-down the user-provided topic:

1. Introduction (no research needed)
   - Brief overview of the topic area

2. Main Body Sections:
   - Each section should focus on a sub-topic of the user-provided topic
   - Include any key concepts and definitions
   - Provide real-world examples or case studies where applicable

3. Conclusion
   - Aim for 1 structural element (either a list of table) that distills the main body sections 
   - Provide a concise summary of the report"""


class Configuration(ConfigurationBase):
    """The configurable fields for the workflow"""
    context_window_length: int = int(12 * 1024)
    max_iterations: int = 3
    number_of_days_back: int = None
    number_of_queries: int = 3
    search_category: TavilySearchCategory = "general"
    strip_thinking_tokens: bool = True
