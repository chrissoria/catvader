"""
_social_media.py
Fetch social media feed data for use as catvader input.

Usage via classify():
    classify(
        feed_input=None,          # omit when using sm_source
        categories=[...],
        sm_source="threads",      # platform to pull from
        sm_limit=50,              # number of posts to fetch
        sm_credentials={...},     # optional; falls back to env vars
    )

Supported sources: "threads"
"""

import os
import requests
import pandas as pd
from dotenv import load_dotenv

_ENV_PATH = os.path.expanduser(
    "~/Documents/Important_Docs/social_media_bot/.env"
)

# Metrics available per-post from the Threads Insights API
_THREADS_METRICS = "likes,replies,reposts,quotes,views,shares,clicks"


def _load_threads_credentials(credentials: dict = None) -> tuple[str, str]:
    """Return (access_token, user_id), preferring explicit credentials dict."""
    load_dotenv(_ENV_PATH, override=True)
    token = (credentials or {}).get("access_token") or os.getenv("THREADS_ACCESS_TOKEN")
    user_id = (credentials or {}).get("user_id") or os.getenv("THREADS_USER_ID")
    if not token:
        raise ValueError(
            "No Threads access token found. Pass sm_credentials={'access_token': '...'} "
            f"or set THREADS_ACCESS_TOKEN in {_ENV_PATH}"
        )
    if not user_id:
        raise ValueError(
            "No Threads user ID found. Pass sm_credentials={'user_id': '...'} "
            f"or set THREADS_USER_ID in {_ENV_PATH}"
        )
    return token, user_id


def _get_threads_posts(token: str, user_id: str, limit: int) -> list[dict]:
    """Fetch recent posts from the Threads API."""
    url = f"https://graph.threads.net/v1.0/{user_id}/threads"
    params = {
        "fields": "id,text,timestamp,media_type",
        "limit": limit,
        "access_token": token,
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json().get("data", [])


def _get_threads_insights(token: str, post_id: str) -> dict:
    """Fetch engagement metrics for a single Threads post."""
    url = f"https://graph.threads.net/v1.0/{post_id}/insights"
    params = {
        "metric": _THREADS_METRICS,
        "access_token": token,
    }
    r = requests.get(url, params=params)
    if r.status_code != 200:
        return {}
    data = r.json().get("data", [])
    return {item["name"]: item["values"][0]["value"] for item in data}


def fetch_threads(limit: int = 50, credentials: dict = None) -> pd.DataFrame:
    """
    Fetch recent Threads posts with engagement metrics.

    Args:
        limit (int): Number of posts to fetch. Default 50.
        credentials (dict): Optional dict with keys 'access_token' and 'user_id'.
            Falls back to THREADS_ACCESS_TOKEN / THREADS_USER_ID env vars.

    Returns:
        pd.DataFrame with columns:
            post_id, timestamp, text, likes, replies, reposts,
            quotes, views, shares, clicks
    """
    token, user_id = _load_threads_credentials(credentials)
    posts = _get_threads_posts(token, user_id, limit)
    rows = []
    for post in posts:
        metrics = _get_threads_insights(token, post["id"])
        rows.append({
            "post_id":   post["id"],
            "timestamp": post.get("timestamp"),
            "text":      post.get("text", ""),
            "likes":     metrics.get("likes", 0),
            "replies":   metrics.get("replies", 0),
            "reposts":   metrics.get("reposts", 0),
            "quotes":    metrics.get("quotes", 0),
            "views":     metrics.get("views", 0),
            "shares":    metrics.get("shares", 0),
            "clicks":    metrics.get("clicks", 0),
        })
    return pd.DataFrame(rows)


# Dispatcher — extend this dict as new platforms are added
_FETCHERS = {
    "threads": fetch_threads,
}

SUPPORTED_SOURCES = list(_FETCHERS.keys())


def fetch_social_media(sm_source: str, limit: int = 50, credentials: dict = None) -> pd.DataFrame:
    """
    Fetch feed data from a social media platform.

    Args:
        sm_source (str): Platform name. Currently supported: "threads".
        limit (int): Number of posts/items to fetch.
        credentials (dict): Platform-specific credentials. Falls back to env vars.

    Returns:
        pd.DataFrame with a 'text' column (the feed content) plus
        platform-specific engagement metric columns.
    """
    key = sm_source.lower().strip()
    if key not in _FETCHERS:
        raise ValueError(
            f"sm_source='{sm_source}' is not supported. "
            f"Supported sources: {SUPPORTED_SOURCES}"
        )
    return _FETCHERS[key](limit=limit, credentials=credentials)
