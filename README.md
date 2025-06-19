# Summary Writer

An AI-powered research tool that automatically generates comprehensive summaries on any topic by intelligently searching the web and synthesizing information using large language models.

## Overview

Summary Writer uses a multi-step workflow to research topics and create high-quality summaries:

1. **Query Generation** - Creates targeted web search queries for comprehensive coverage
2. **Web Search** - Searches the internet using Tavily API for relevant, up-to-date information  
3. **Summary Writing** - Synthesizes search results into coherent summaries using reasoning models

## Features

- **Multi-LLM Support**: Compatible with Groq, OpenAI, VLLM, and Ollama
- **Intelligent Query Generation**: Automatically creates diverse search queries to cover all aspects of a topic
- **Web Search Integration**: Uses Tavily API for comprehensive web search capabilities
- **Advanced Reasoning**: Leverages reasoning models for high-quality summary generation
- **Configurable Workflow**: Customizable research iterations and query parameters
- **LangGraph Orchestration**: Uses LangGraph for robust workflow management

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd summary-writer

# Install dependencies using uv
uv sync
```

## Configuration

Set up your API keys in environment variables:

```bash
export GROQ_API_KEY="your-groq-api-key"
export OPENAI_API_KEY="your-openai-api-key" 
export TAVILY_API_KEY="your-tavily-api-key"
```

## Usage

### Basic Usage

```python
import asyncio
from summary_writer import SummaryWriter
from ai_common import LlmServers

# Configure LLM models
llm_config = {
    'language_model': {
        'model': 'llama-3.3-70b-versatile',
        'model_provider': LlmServers.GROQ.value,
        'api_key': 'your-groq-api-key',
        'model_args': {
            'temperature': 0,
            'max_retries': 5,
            'max_tokens': 32768,
            'model_kwargs': {
                'top_p': 0.95,
                'service_tier': "auto",
            }
        }
    },
    'reasoning_model': {
        'model': 'deepseek-r1-distill-llama-70b',
        'model_provider': LlmServers.GROQ.value,
        'api_key': 'your-groq-api-key',
        'model_args': {
            'temperature': 0,
            'max_retries': 5,
            'max_tokens': 32768,
            'model_kwargs': {
                'top_p': 0.95,
                'service_tier': "auto",
            }
        }
    }
}

# Initialize the summary writer
writer = SummaryWriter(
    llm_config=llm_config,
    web_search_api_key='your-tavily-api-key'
)

# Configure the research workflow
config = {
    "configurable": {
        'max_iterations': 3,
        'max_results_per_query': 4,
        'max_tokens_per_source': 10000,
        'number_of_days_back': 30,  # Search within last 30 days
        'number_of_queries': 3,
        'search_category': 'general',
        'strip_thinking_tokens': True,
    }
}

# Generate a summary (async)
async def generate_summary():
    result = await writer.run(
        topic='artificial intelligence trends 2024',
        config=config
    )
    return result

# Run the async function
loop = asyncio.get_event_loop()
response = loop.run_until_complete(generate_summary())
print(response['content'])
```

### Configuration Options

The workflow can be customized through the config parameter:

- **max_iterations**: Maximum number of research iterations (default: 3)
- **max_results_per_query**: Maximum search results per query (default: 4)
- **max_tokens_per_source**: Maximum tokens to extract from each source (default: 10000)
- **number_of_days_back**: Search within X days (None for no limit)
- **number_of_queries**: Number of search queries to generate (default: 3)
- **search_category**: Tavily search category ('general', 'news', etc.)
- **strip_thinking_tokens**: Remove reasoning tokens from output (default: True)

### Running the Development Script

```bash
python src/main_dev.py
```

This script demonstrates a complete workflow with cost tracking and timing information.

## Project Structure

```
src/
├── summary_writer/
│   ├── summary_writer.py      # Main SummaryWriter class and workflow
│   ├── state.py               # State management for the workflow
│   ├── configuration.py       # Configuration settings
│   ├── enums.py              # Enums and constants
│   └── components/
│       ├── __init__.py        # Component exports
│       └── writer.py          # Creates summaries from search results
├── main_dev.py               # Development entry point
└── config.py                 # Application configuration
```

## Dependencies

- **LangChain**: For LLM integration and workflow orchestration
- **LangGraph**: For building the research workflow graph
- **Tavily**: For web search capabilities
- **AI Common**: Shared utilities and base classes
- **Rich**: For enhanced console output

## Workflow Details

The Summary Writer follows a sophisticated multi-step process:

### 1. Query Generation
- Uses a language model to generate diverse, targeted search queries
- Ensures comprehensive coverage of the research topic
- Optimizes queries for web search effectiveness

### 2. Web Search & Information Gathering
- Executes multiple search queries using Tavily API
- Aggregates results from various sources
- Filters and processes content for relevance
- Tracks unique sources to avoid duplication

### 3. Iterative Research
- Performs multiple research iterations for thorough coverage
- Routes between continued research and summary generation
- Maintains state across iterations for consistency

### 4. Summary Generation
- Uses reasoning models for high-quality synthesis
- Combines information from all gathered sources
- Produces coherent, comprehensive summaries
- Tracks token usage and costs

## State Management

The workflow maintains a comprehensive state object (`SummaryState`) that includes:

- **topic**: The research topic
- **search_queries**: Generated search queries
- **source_str**: Formatted source content from web searches
- **content**: Generated summary content
- **steps**: Workflow steps executed
- **token_usage**: Detailed token consumption tracking
- **unique_sources**: De-duplicated source tracking
- **iteration**: Current research iteration count

## Cost Tracking

The system provides detailed cost analysis:
- Tracks input/output tokens for each model
- Calculates costs based on current pricing
- Supports multiple LLM providers
- Provides per-model and total cost breakdowns

## Error Handling

- Configurable retry mechanisms for LLM calls
- Graceful handling of search API failures
- State preservation across component failures
- Comprehensive logging and debugging support

## License

This project is licensed under the MIT License.