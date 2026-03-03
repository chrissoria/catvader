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
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

_ENV_PATH = os.path.expanduser(
    "~/Documents/Important_Docs/social_media_bot/.env"
)

# Metrics available per-post from the Threads Insights API
_THREADS_METRICS = "likes,replies,reposts,quotes,views,shares"


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
    """Fetch posts from the Threads API, paginating as needed."""
    url = f"https://graph.threads.net/v1.0/{user_id}/threads"
    page_size = min(limit, 100)  # API max per page is 100
    fields = "id,text,timestamp,media_type,media_url,thumbnail_url,children{media_url,media_type}"
    params = {
        "fields": fields,
        "limit": page_size,
        "access_token": token,
    }
    posts = []
    while len(posts) < limit:
        r = requests.get(url, params=params)
        r.raise_for_status()
        body = r.json()
        posts.extend(body.get("data", []))
        cursor = body.get("paging", {}).get("cursors", {}).get("after")
        if not cursor or not body.get("paging", {}).get("next"):
            break
        params = {
            "fields": fields,
            "limit": min(page_size, limit - len(posts)),
            "after": cursor,
            "access_token": token,
        }
    return posts[:limit]


def _extract_image_url(post: dict) -> str:
    """Extract the first image URL from a post, handling carousels."""
    media_type = post.get("media_type", "")
    if media_type == "IMAGE":
        return post.get("media_url", "")
    if media_type == "VIDEO":
        return post.get("thumbnail_url", "")
    if media_type == "CAROUSEL_ALBUM":
        children = post.get("children", {}).get("data", [])
        for child in children:
            if child.get("media_type") == "IMAGE" and child.get("media_url"):
                return child["media_url"]
    return ""


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


def fetch_threads(limit: int = 50, months: int = None, credentials: dict = None) -> pd.DataFrame:
    """
    Fetch recent Threads posts with engagement metrics.

    Args:
        limit (int): Maximum number of posts to fetch. Default 50.
            Ignored when months is set (all posts within the window are returned).
        months (int): If set, only return posts from the last N months.
            Overrides limit — fetches until the cutoff date is reached.
        credentials (dict): Optional dict with keys 'access_token' and 'user_id'.
            Falls back to THREADS_ACCESS_TOKEN / THREADS_USER_ID env vars.

    Returns:
        pd.DataFrame with columns:
            post_id, timestamp, media_type, text, image_url,
            likes, replies, reposts, quotes, views, shares
    """
    token, user_id = _load_threads_credentials(credentials)

    cutoff = None
    if months is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=months * 30)
        fetch_limit = 1000  # paginate until cutoff
    else:
        fetch_limit = limit

    posts = _get_threads_posts(token, user_id, fetch_limit)

    rows = []
    for post in posts:
        ts_str = post.get("timestamp", "")
        if cutoff and ts_str:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if ts < cutoff:
                break  # posts are newest-first; stop once past the window
        metrics = _get_threads_insights(token, post["id"])
        rows.append({
            "post_id":    post["id"],
            "timestamp":  ts_str,
            "media_type": post.get("media_type", ""),
            "text":       post.get("text", ""),
            "image_url":  _extract_image_url(post),
            "likes":      metrics.get("likes", 0),
            "replies":    metrics.get("replies", 0),
            "reposts":    metrics.get("reposts", 0),
            "quotes":     metrics.get("quotes", 0),
            "views":      metrics.get("views", 0),
            "shares":     metrics.get("shares", 0),
        })
    return pd.DataFrame(rows)


# Dispatcher — extend this dict as new platforms are added
_FETCHERS = {
    "threads": fetch_threads,
}

SUPPORTED_SOURCES = list(_FETCHERS.keys())


def fetch_social_media(sm_source: str, limit: int = 50, months: int = None, credentials: dict = None) -> pd.DataFrame:
    """
    Fetch feed data from a social media platform.

    Args:
        sm_source (str): Platform name. Currently supported: "threads".
        limit (int): Number of posts/items to fetch. Ignored when months is set.
        months (int): If set, fetch all posts from the last N months.
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
    return _FETCHERS[key](limit=limit, months=months, credentials=credentials)
