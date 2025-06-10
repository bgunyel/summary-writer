import datetime
import time
from uuid import uuid4
from typing import TypedDict

from config import settings
from src.summary_writer import SummaryWriter
from src.summary_writer.configuration import Configuration
from ai_common import LlmServers, TavilySearchCategory



def main():

    llm_server = LlmServers.GROQ

    llm_config = {
        LlmServers.GROQ.value: {
            'model_name': None,
            'groq_api_key': settings.GROQ_API_KEY,
            'language_model': 'llama-3.3-70b-versatile',
            'reasoning_model': 'deepseek-r1-distill-llama-70b',
        },
        LlmServers.OPENAI.value: {
            'model_name': None,
            'openai_api_key': settings.OPENAI_API_KEY,
            'language_model': 'gpt-4.1-2025-04-14',
            'reasoning_model': 'o3',
        },
        LlmServers.VLLM.value: {
            'llm_base_url': None,
            'vllm_api_key': None
        },
        LlmServers.OLLAMA.value: {
            'model_name': None,
            'llm_base_url': None,
            'format': None,  # Literal['', 'json']
            'context_window_length': None,
        }
    }

    language_model = llm_config[llm_server.value].get('language_model', '')
    reasoning_model = llm_config[llm_server.value].get('reasoning_model', '')

    """
    engine = Engine(
        responder=SummaryWriter(
            llm_server = llm_server,
            llm_config = llm_config[llm_server.value],
            web_search_api_key = settings.TAVILY_API_KEY
        ),
        llm_server=llm_server,
        models=[language_model, reasoning_model],
        llm_base_url=llm_config[llm_server.value].get('llm_base_url', ''),
        save_to_folder=settings.OUT_FOLDER
    )
    """

    topic = 'first 100 days of Trump administration'
    print(f'LLM Server: {llm_server.value}')
    print(f'Language Model: {language_model}')
    print(f'Reasoning Model: {reasoning_model}')
    print('\n')
    print(f'Topic: {topic}')
    print('\n\n\n')

    config = {
        "configurable": {
            'thread_id': str(uuid4()),
            'max_iterations': 3,
            'max_results_per_query': 4,
            'max_tokens_per_source': 10000,
            'number_of_days_back': 1e6,
            'number_of_queries': 3,
            'search_category': 'general',
            'strip_thinking_tokens': True,
            }
        }

    summary_writer = SummaryWriter(llm_server = llm_server,
                                   llm_config = llm_config[llm_server.value],
                                   web_search_api_key = settings.TAVILY_API_KEY,
                                   search_category=config['configurable']['search_category'],
                                   number_of_days_back=config['configurable']['number_of_days_back'])


    out_dict = summary_writer.run(topic=topic, config=config)


    dummy = -32



if __name__ == '__main__':
    time_now = datetime.datetime.now().replace(microsecond=0).astimezone(
        tz=datetime.timezone(offset=datetime.timedelta(hours=3), name='UTC+3'))

    print(f'{settings.APPLICATION_NAME} started at {time_now}')
    time1 = time.time()
    main()
    time2 = time.time()

    time_now = datetime.datetime.now().replace(microsecond=0).astimezone(
        tz=datetime.timezone(offset=datetime.timedelta(hours=3), name='UTC+3'))
    print(f'{settings.APPLICATION_NAME} finished at {time_now}')
    print(f'{settings.APPLICATION_NAME} took {(time2 - time1):.2f} seconds')
