"""
Category exploration functions for CatVader.

Thin wrapper around cat_stack.explore() that adds social media fetching
and context injection. Returns raw category lists (with duplicates) for
frequency/saturation analysis.
"""

import cat_stack

from ._social_media import fetch_social_media, SUPPORTED_SOURCES

__all__ = [
    "explore",
]


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
    # Everything else passed through to cat_stack.explore()
    **kwargs,
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
            Supported: "threads", "bluesky", "reddit", "mastodon", "youtube"
        sm_limit (int): Number of posts to fetch. Default 50.
        sm_months (int): Fetch all posts from the last N months.
        sm_credentials (dict): Platform credentials. Falls back to env vars.
        platform (str): Social media platform name for prompt context.
        handle (str): Post author handle for prompt context.
        hashtags (str or list): Hashtags for prompt context.
        post_metadata (dict): Additional metadata for prompt context.
        description (str): The survey question or description of the data.
        **kwargs: All additional parameters are passed through to
            cat_stack.explore(). This includes: max_categories,
            categories_per_chunk, divisions, user_model, creativity,
            specificity, research_question, filename, model_source,
            iterations, random_state, focus, progress_callback, etc.

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

    return cat_stack.explore(
        input_data=input_data,
        api_key=api_key,
        description=description,
        **kwargs,
    )
