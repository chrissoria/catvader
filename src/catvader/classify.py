"""
Classification functions for CatVader.

Thin wrapper around cat_stack.classify() that adds social media fetching,
context injection, and engagement metric attachment.
"""

from typing import Union
import pandas as pd

import cat_stack

from ._social_media import fetch_social_media, SUPPORTED_SOURCES

__all__ = [
    "classify",
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


def classify(
    input_data=None,
    categories=None,
    api_key=None,
    # Social media source — when set, input_data is fetched automatically
    sm_source: str = None,
    sm_limit: int = 50,
    sm_months: int = None,
    sm_days: int = None,
    sm_credentials: dict = None,
    sm_handle: str = None,
    sm_timezone: str = "UTC",
    sm_youtube_content: str = "video",
    sm_youtube_transcript: bool = False,
    sm_comments_per_video: int = 20,
    sm_youtube_transcript_max_chars: int = 10_000,
    # Social media context fields — injected into the classification prompt
    platform: str = None,
    handle: str = None,
    hashtags=None,
    post_metadata: dict = None,
    # Mapped params
    description="",
    feed_question: str = "",
    # Everything else passed through to cat_stack.classify()
    **kwargs,
):
    """
    Unified classification function for text, image, and PDF inputs.

    Supports single-model and multi-model (ensemble) classification. Input type
    is auto-detected from the data (text strings, image paths, or PDF paths).

    Social media parameters (sm_*) allow automatic fetching from platforms.
    Context fields (platform, handle, hashtags, post_metadata) are injected
    into the classification prompt. All other parameters are passed through
    to cat_stack.classify().

    Args:
        input_data: The data to classify. Can be:
            - For text: list of text responses or pandas Series
            - For image: directory path or list of image file paths
            - For pdf: directory path or list of PDF file paths
            - Omit when using sm_source (data is fetched automatically).
        categories (list): List of category names for classification.
        api_key (str): API key for the model provider (single-model mode).
        platform (str): Social media platform (e.g., "Twitter/X", "Reddit").
            Injected into the classification prompt as context.
        handle (str): Post author handle (e.g., "@username", "r/subreddit").
        hashtags (str or list): Hashtags associated with the post(s).
        post_metadata (dict): Additional post metadata injected into the prompt
            (e.g., {"likes": 1200, "shares": 450}).
        sm_source (str): Social media platform to pull feed data from.
            Supported: "threads", "bluesky", "reddit", "mastodon", "youtube"
        sm_limit (int): Number of posts to fetch. Default 50.
        sm_months (int): Fetch all posts from the last N months.
        sm_days (int): Fetch all posts from the last N days (Reddit only).
        sm_handle (str): Account handle for platforms that require one.
        sm_youtube_content (str): "video" or "comments". Default "video".
        sm_youtube_transcript (bool): Use transcript instead of description.
        sm_comments_per_video (int): Max comments per video. Default 20.
        sm_youtube_transcript_max_chars (int): Max transcript chars. Default 10000.
        sm_credentials (dict): Platform credentials. Falls back to env vars.
        sm_timezone (str): Timezone for day/month columns. Default "UTC".
        description (str): Description of the input data context.
        feed_question (str): Context for category discovery (maps to
            cat_stack's survey_question parameter).
        **kwargs: All additional parameters are passed through to
            cat_stack.classify(). This includes: user_model, model_source,
            creativity, safety, chain_of_thought, step_back_prompt,
            thinking_budget, models, consensus_threshold, batch_mode,
            embeddings, json_formatter, multi_label, add_other,
            check_verbosity, filename, save_directory, etc.

    Returns:
        pd.DataFrame: Results with classification columns.

    Examples:
        >>> import catvader as cat
        >>>
        >>> # Single model classification
        >>> results = cat.classify(
        ...     input_data=df['responses'],
        ...     categories=["Positive", "Negative", "Neutral"],
        ...     description="Customer feedback survey",
        ...     api_key="your-api-key"
        ... )
        >>>
        >>> # Multi-model ensemble
        >>> results = cat.classify(
        ...     input_data=df['responses'],
        ...     categories=["Positive", "Negative"],
        ...     models=[
        ...         ("gpt-5", "openai", "sk-..."),
        ...         ("claude-sonnet-4-5-20250929", "anthropic", "sk-ant-..."),
        ...     ],
        ...     consensus_threshold="majority",
        ... )
        >>>
        >>> # Social media classification
        >>> results = cat.classify(
        ...     sm_source="threads",
        ...     sm_limit=100,
        ...     categories=["Positive", "Negative", "Neutral"],
        ...     api_key="your-api-key"
        ... )
    """
    # =========================================================================
    # Social media source — fetch input_data automatically when sm_source set
    # =========================================================================
    _sm_df = None  # holds the full fetched DataFrame (text + metrics)
    if sm_source is not None:
        if input_data is not None:
            raise ValueError(
                "Pass either input_data or sm_source, not both."
            )
        if sm_handle:
            target = sm_handle
        elif sm_source == "reddit" and sm_credentials:
            target = sm_credentials.get("subreddit") or sm_credentials.get("username") or "your account"
        else:
            target = "your account"
        window = f"last {sm_days}d" if sm_days else f"last {sm_months}mo" if sm_months else f"limit={sm_limit}"
        print(f"[CatVader] Fetching feed from '{sm_source}' ({target}, {window})...")
        _sm_df = fetch_social_media(
            sm_source,
            limit=sm_limit,
            months=sm_months,
            days=sm_days,
            credentials=sm_credentials,
            handle=sm_handle,
            youtube_content=sm_youtube_content,
            youtube_transcript=sm_youtube_transcript,
            comments_per_video=sm_comments_per_video,
            youtube_transcript_max_chars=sm_youtube_transcript_max_chars,
        )
        input_data = _sm_df["text"].tolist()
        print(f"[CatVader] Fetched {len(input_data)} posts.")
        if not feed_question:
            feed_question = "What topics are discussed in these social media posts?"
    elif input_data is None:
        raise ValueError(
            "Provide either input_data or sm_source="
            f"{SUPPORTED_SOURCES}."
        )

    # Prepend social media context to description if any fields provided
    sm_context = _build_social_media_context(platform, handle, hashtags, post_metadata)
    if sm_context:
        description = f"{sm_context}\n{description}".strip() if description else sm_context

    # =========================================================================
    # Delegate to cat_stack.classify() — it handles models list building,
    # add_other, check_verbosity, strategy warnings, batch mode, etc.
    # =========================================================================
    result = cat_stack.classify(
        input_data=input_data,
        categories=categories,
        api_key=api_key,
        description=description,
        survey_question=feed_question,
        **kwargs,
    )

    # =========================================================================
    # Attach social media metrics to the result when sm_source was used
    # =========================================================================
    if _sm_df is not None:
        metric_cols = [c for c in _sm_df.columns if c not in ("text",)]
        result = result.reset_index(drop=True)
        for col in metric_cols:
            result[col] = _sm_df[col].values
        # Derive day-of-week and month name from timestamp
        if "timestamp" in _sm_df.columns:
            ts = pd.to_datetime(_sm_df["timestamp"], utc=True, errors="coerce")
            if sm_timezone != "UTC":
                ts = ts.dt.tz_convert(sm_timezone)
            result["day"] = ts.dt.day_name().values
            result["month"] = ts.dt.month_name().values
            result["hour"] = ts.dt.hour.values
            date_only = ts.dt.date
            result["n_posts_that_day"] = date_only.map(date_only.value_counts()).values
        # Flag reposts
        if "media_type" in _sm_df.columns:
            result["is_repost"] = (_sm_df["media_type"].str.upper().str.contains("REPOST", na=False)).astype(int).values
        # Post length in characters
        if "text" in _sm_df.columns:
            result["post_length"] = _sm_df["text"].fillna("").str.len().values
            result["contains_url"] = _sm_df["text"].fillna("").str.contains(r"https?://", regex=True).astype(int).values
        # Contains image (any media type with an image or video thumbnail)
        if "image_url" in _sm_df.columns:
            result["contains_image"] = (_sm_df["image_url"].fillna("") != "").astype(int).values

    return result
