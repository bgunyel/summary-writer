from pydantic import BaseModel


def format_context_with_source_ids(state: BaseModel) -> str:
    """
    Format the context to include source IDs for citation.
    """

    number_of_previous_sources = sum(
        [len(state.cumulative_unique_sources[i]) for i in range(state.iteration)]
    ) if state.iteration > 0 else 0

    formatted_sources = []
    for i, (k, v) in enumerate(iterable=state.unique_sources.items(), start=number_of_previous_sources +1):
        source_id = f"src{i}"
        title = v['title']
        content = v['content']
        url = k
        formatted_sources.append(f"[{source_id}] {title}\nURL: {url}\nContent: {content}")

    return "\n".join(formatted_sources)
