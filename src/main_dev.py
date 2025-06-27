import asyncio
import datetime
import time
from uuid import uuid4

from config import settings
from src.summary_writer import SummaryWriter
from ai_common import LlmServers, PRICE_USD_PER_MILLION_TOKENS


def main():

    llm_config = {
        'language_model': {
            'model': 'llama-3.3-70b-versatile',
            'model_provider': LlmServers.GROQ.value,
            'api_key': settings.GROQ_API_KEY,
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
            'model': 'qwen/qwen3-32b', #'deepseek-r1-distill-llama-70b',
            'model_provider': LlmServers.GROQ.value,
            'api_key': settings.GROQ_API_KEY,
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

    language_model = llm_config['language_model'].get('model', '')
    reasoning_model = llm_config['reasoning_model'].get('model', '')

    topic = 'Life and Philosophy of Marcus Aurelius'
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
            'min_claim_confidence': 0.7,
            'number_of_days_back': 1e6,
            'number_of_queries': 3,
            'search_category': 'general',
            'strip_thinking_tokens': True,
            }
        }

    summary_writer = SummaryWriter(llm_config = llm_config, web_search_api_key = settings.TAVILY_API_KEY)

    event_loop = asyncio.new_event_loop()
    out_dict = event_loop.run_until_complete(summary_writer.run(topic=topic, config=config))
    event_loop.close()

    # out_dict = summary_writer.run(topic=topic, config=config)

    total_cost = 0
    for model_type, params in llm_config.items():
        model_provider = params['model_provider']
        model = params['model']
        price_dict = PRICE_USD_PER_MILLION_TOKENS[model_provider][model]
        cost = sum([price_dict[k] * out_dict['token_usage'][model][k] for k in price_dict.keys()]) / 1e6
        total_cost += cost
        print(f'Cost for {model_provider}: {model} --> {cost:.4f} USD')
    print(f'Total Token Usage Cost: {total_cost:.4f} USD')
    print('\n\n\n')

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
