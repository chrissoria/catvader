"""
Category extraction functions for CatVader.

Thin wrapper around cat_stack.extract() that adds social media fetching
and context injection.
"""

import cat_stack

from ._social_media import fetch_social_media, SUPPORTED_SOURCES

__all__ = [
    "extract",
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


def extract(
    input_data=None,
    api_key=None,
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
    description="",
    # Everything else passed through to cat_stack.extract()
    **kwargs,
):
    """
    Unified category extraction function for text, image, and PDF inputs.

    This function dispatches to cat_stack.extract() after handling social
    media fetching and context injection.

    Args:
        input_data: The data to explore. Can be:
            - For text: list of text responses or pandas Series
            - For image: directory path, single file, or list of image paths
            - For pdf: directory path, single file, or list of PDF paths
            - Omit when using sm_source (text is fetched automatically).
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
        description (str): Description of the input data. Maps to
            cat_stack's survey_question parameter.
        **kwargs: All additional parameters are passed through to
            cat_stack.extract(). This includes: input_type, max_categories,
            categories_per_chunk, divisions, user_model, creativity,
            specificity, research_question, mode, filename, model_source,
            iterations, random_state, focus, progress_callback, etc.

    Returns:
        dict with keys:
            - counts_df: DataFrame of categories with counts
            - top_categories: List of top category names
            - raw_top_text: Raw model output from final merge step

    Examples:
        >>> import catvader as cat
        >>>
        >>> # Extract categories from text
        >>> results = cat.extract(
        ...     input_data=df['responses'],
        ...     description="Why did you move?",
        ...     api_key="your-api-key"
        ... )
        >>> print(results['top_categories'])
        >>>
        >>> # Extract from social media
        >>> results = cat.extract(
        ...     sm_source="threads",
        ...     sm_limit=100,
        ...     api_key="your-api-key"
        ... )
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

    return cat_stack.extract(
        input_data=input_data,
        api_key=api_key,
        survey_question=description,
        **kwargs,
    )
