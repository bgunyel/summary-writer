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
from summary_writer import SummaryWriter
from ai_common import LlmServers

# Initialize the summary writer
writer = SummaryWriter(
    llm_server=LlmServers.GROQ,
    llm_config={
        'language_model': 'llama-3.3-70b-versatile',
        'reasoning_model': 'deepseek-r1-distill-llama-70b',
        'groq_api_key': 'your-api-key'
    },
    web_search_api_key='your-tavily-api-key'
)

# Generate a summary
response = writer.get_response({'topic': 'artificial intelligence trends 2024'})
print(response)
```

### Running the Development Script

```bash
python src/main_dev.py
```

## Project Structure

```
src/
├── summary_writer/
│   ├── summary_writer.py      # Main SummaryWriter class and workflow
│   ├── state.py               # State management for the workflow
│   ├── configuration.py       # Configuration settings
│   ├── enums.py              # Enums and constants
│   └── components/
│       ├── query_writer.py    # Generates search queries
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

## License

This project is licensed under the MIT License.