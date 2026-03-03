"""
Category exploration functions for CatVader.

This module provides raw category extraction from text inputs,
returning unprocessed category lists for frequency/saturation analysis.
"""

import pandas as pd

from ._social_media import fetch_social_media, SUPPORTED_SOURCES

__all__ = [
    "explore",
]

from .text_functions import explore_common_categories


def _build_social_media_context(platform, handle, hashtags, post_metadata):
    """Build a context block string from social media metadata fields."""
    parts = []
    if platform:
        parts.append(f"Platform: {platform}")
    if handle:
        parts.append(f"Author: {handle}")
    if hashtags:
        tags = hashtags if isinstance(hashtags, str) else " ".join(hashtags)
        parts.append(f"Hashtags: {tags}")
    if post_metadata:
        for k, v in post_metadata.items():
            parts.append(f"{k.capitalize()}: {v}")
    return "\n".join(parts)


def explore(
    input_data=None,
    api_key=None,
    description="",
    # Social media source
    sm_source: str = None,
    sm_limit: int = 50,
    sm_months: int = None,
    sm_credentials: dict = None,
    # Social media context fields
    platform: str = None,
    handle: str = None,
    hashtags=None,
    post_metadata: dict = None,
    max_categories=12,
    categories_per_chunk=10,
    divisions=12,
    user_model="gpt-5",
    creativity=None,
    specificity="broad",
    research_question=None,
    filename=None,
    model_source="auto",
    iterations=8,
    random_state=None,
    focus=None,
    progress_callback=None,
):
    """
    Explore categories in text data, returning the raw extracted list.

    Unlike extract(), which normalizes, deduplicates, and semantically merges
    categories, explore() returns every category string from every chunk across
    every iteration — with duplicates intact. This is useful for analyzing
    category stability and saturation across repeated extraction runs.

    Args:
        input_data: List of text responses or pandas Series.
            Omit when using sm_source (text is fetched automatically).
        api_key (str): API key for the model provider.
        sm_source (str): Social media platform to pull feed text from.
            Supported: "threads"
        sm_limit (int): Number of posts to fetch. Default 50.
        sm_credentials (dict): Platform credentials. Falls back to env vars.
        description (str): The survey question or description of the data.
        max_categories (int): Maximum categories per chunk (passed through).
        categories_per_chunk (int): Categories to extract per chunk.
        divisions (int): Number of chunks to divide data into.
        user_model (str): Model name to use. Default "gpt-5".
        creativity (float): Temperature setting. None uses model default.
        specificity (str): "broad" or "specific" category granularity.
        research_question (str): Optional research context.
        filename (str): Optional CSV filename to save raw category list.
        model_source (str): Provider - "auto", "openai", "anthropic", etc.
        iterations (int): Number of passes over the data.
        random_state (int): Random seed for reproducibility.
        focus (str): Optional focus instruction for category extraction.
        progress_callback (callable): Optional callback for progress updates.

    Returns:
        list[str]: Every category string extracted from every chunk across
        every iteration. Length ≈ iterations × divisions × categories_per_chunk.

    Examples:
        >>> import catvader as cat
        >>>
        >>> raw_categories = cat.explore(
        ...     input_data=df['responses'],
        ...     description="Why did you move?",
        ...     api_key="your-api-key",
        ...     iterations=3,
        ...     divisions=5,
        ... )
        >>> print(len(raw_categories))  # ~150
        >>> print(raw_categories[:5])
    """
    # Fetch feed text from social media when sm_source is set
    if sm_source is not None:
        if input_data is not None:
            raise ValueError("Pass either input_data or sm_source, not both.")
        print(f"[CatVader] Fetching feed from '{sm_source}' (limit={sm_limit})...")
        _sm_df = fetch_social_media(sm_source, limit=sm_limit, months=sm_months, credentials=sm_credentials)
        input_data = _sm_df["text"].tolist()
        print(f"[CatVader] Fetched {len(input_data)} posts.")
    elif input_data is None:
        raise ValueError(
            f"Provide either input_data or sm_source={SUPPORTED_SOURCES}."
        )

    # Prepend social media context to description if any fields provided
    sm_context = _build_social_media_context(platform, handle, hashtags, post_metadata)
    if sm_context:
        description = f"{sm_context}\n{description}".strip() if description else sm_context

    raw_items = explore_common_categories(
        survey_input=input_data,
        api_key=api_key,
        survey_question=description,
        max_categories=max_categories,
        categories_per_chunk=categories_per_chunk,
        divisions=divisions,
        user_model=user_model,
        creativity=creativity,
        specificity=specificity,
        research_question=research_question,
        filename=None,  # We handle saving ourselves
        model_source=model_source,
        iterations=iterations,
        random_state=random_state,
        focus=focus,
        progress_callback=progress_callback,
        return_raw=True,
    )

    if filename:
        df = pd.DataFrame(raw_items, columns=["Category"])
        df.to_csv(filename, index=False)
        print(f"Raw categories saved to {filename}")

    return raw_items
