"""
_social_media.py
Fetch social media feed data for use as catvader input.

Usage via classify():
    classify(
        feed_input=None,          # omit when using sm_source
        categories=[...],
        sm_source="threads",      # platform to pull from ("threads", "bluesky", "reddit")
        sm_limit=50,              # number of posts to fetch
        sm_credentials={...},     # optional; falls back to env vars
    )

Supported sources: "threads", "bluesky", "reddit"
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


_BLUESKY_BASE_URL = "https://bsky.social/xrpc"


def _load_bluesky_credentials(credentials: dict = None) -> tuple[str, str]:
    """Return (handle, app_password), preferring explicit credentials dict."""
    load_dotenv(_ENV_PATH, override=True)
    handle = (credentials or {}).get("handle") or os.getenv("BLUESKY_HANDLE")
    password = (credentials or {}).get("app_password") or os.getenv("BLUESKY_APP_PASSWORD")
    if not handle:
        raise ValueError(
            "No Bluesky handle found. Pass sm_credentials={'handle': '...'} "
            f"or set BLUESKY_HANDLE in {_ENV_PATH}"
        )
    if not password:
        raise ValueError(
            "No Bluesky app password found. Pass sm_credentials={'app_password': '...'} "
            f"or set BLUESKY_APP_PASSWORD in {_ENV_PATH}"
        )
    return handle, password


def _bluesky_create_session(handle: str, password: str) -> dict:
    """Authenticate with Bluesky and return session dict (accessJwt, did)."""
    r = requests.post(
        f"{_BLUESKY_BASE_URL}/com.atproto.server.createSession",
        json={"identifier": handle, "password": password},
    )
    r.raise_for_status()
    return r.json()


def _bluesky_extract_image_url(post: dict) -> str:
    """Extract first image URL from a Bluesky post embed."""
    embed = post.get("embed", {})
    embed_type = embed.get("$type", "")
    if "images" in embed_type:
        images = embed.get("images", [])
        if images:
            return images[0].get("fullsize") or images[0].get("thumb", "")
    if "recordWithMedia" in embed_type:
        media = embed.get("media", {})
        images = media.get("images", [])
        if images:
            return images[0].get("fullsize") or images[0].get("thumb", "")
    return ""


def _bluesky_media_type(item: dict) -> str:
    """Derive a media_type string comparable to Threads media_type."""
    if item.get("reason", {}).get("$type", "").endswith("reasonRepost"):
        return "REPOST_FACADE"
    embed_type = item.get("post", {}).get("embed", {}).get("$type", "")
    if "images" in embed_type or "recordWithMedia" in embed_type:
        return "IMAGE"
    if "video" in embed_type:
        return "VIDEO"
    return "TEXT_POST"


def fetch_bluesky(limit: int = 50, months: int = None, credentials: dict = None, handle: str = None) -> pd.DataFrame:
    """
    Fetch recent Bluesky posts with engagement metrics.

    Args:
        limit (int): Maximum number of posts to fetch. Default 50.
            Ignored when months is set.
        months (int): If set, only return posts from the last N months.
        credentials (dict): Optional dict with keys 'handle' and 'app_password'.
            Falls back to BLUESKY_HANDLE / BLUESKY_APP_PASSWORD env vars.
            Not required when fetching another user's public posts.
        handle (str): Bluesky handle of the account to fetch posts from
            (e.g. "user.bsky.social"). When omitted, fetches the
            authenticated user's own posts (credentials required).

    Returns:
        pd.DataFrame with columns:
            post_id, timestamp, media_type, text, image_url,
            likes, replies, reposts, quotes, views, shares
        (views and shares are 0 — not exposed by the Bluesky API)
    """
    if handle:
        # Fetching another user's public posts — auth is optional
        actor = handle
        headers = {}
        try:
            auth_handle, password = _load_bluesky_credentials(credentials)
            session = _bluesky_create_session(auth_handle, password)
            headers = {"Authorization": f"Bearer {session['accessJwt']}"}
        except Exception:
            pass  # public posts are readable without auth
    else:
        # Fetching own posts — auth required
        auth_handle, password = _load_bluesky_credentials(credentials)
        session = _bluesky_create_session(auth_handle, password)
        actor = session["did"]
        headers = {"Authorization": f"Bearer {session['accessJwt']}"}

    cutoff = None
    if months is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=months * 30)

    rows = []
    cursor = None
    page_size = min(limit if months is None else 100, 100)

    while True:
        params = {"actor": actor, "limit": page_size}
        if cursor:
            params["cursor"] = cursor
        r = requests.get(
            f"{_BLUESKY_BASE_URL}/app.bsky.feed.getAuthorFeed",
            headers=headers,
            params=params,
        )
        r.raise_for_status()
        body = r.json()
        feed = body.get("feed", [])

        for item in feed:
            post = item.get("post", {})
            record = post.get("record", {})
            ts_str = record.get("createdAt", post.get("indexedAt", ""))

            if cutoff and ts_str:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if ts < cutoff:
                    return pd.DataFrame(rows)

            rows.append({
                "post_id":    post.get("uri", ""),
                "timestamp":  ts_str,
                "media_type": _bluesky_media_type(item),
                "text":       record.get("text", ""),
                "image_url":  _bluesky_extract_image_url(post),
                "likes":      post.get("likeCount", 0),
                "replies":    post.get("replyCount", 0),
                "reposts":    post.get("repostCount", 0),
                "quotes":     post.get("quoteCount", 0),
                "views":      0,
                "shares":     0,
            })

            if months is None and len(rows) >= limit:
                return pd.DataFrame(rows)

        cursor = body.get("cursor")
        if not cursor or not feed:
            break

    return pd.DataFrame(rows)


# =============================================================================
# Reddit
# =============================================================================

_REDDIT_BASE_URL = "https://www.reddit.com"
_REDDIT_OAUTH_URL = "https://oauth.reddit.com"
_REDDIT_USER_AGENT = "catvader/1.9.0"


def _load_reddit_credentials(credentials: dict = None) -> dict:
    """Return credentials dict, merging in env vars where not explicitly set."""
    load_dotenv(_ENV_PATH, override=True)
    creds = dict(credentials or {})
    for key, env_var in [
        ("client_id",     "REDDIT_CLIENT_ID"),
        ("client_secret", "REDDIT_CLIENT_SECRET"),
        ("username",      "REDDIT_USERNAME"),
        ("subreddit",     "REDDIT_SUBREDDIT"),
    ]:
        if key not in creds and os.getenv(env_var):
            creds[key] = os.getenv(env_var)
    return creds


def _reddit_oauth_token(client_id: str, client_secret: str) -> str:
    """Obtain an app-only OAuth2 bearer token (no user login required)."""
    r = requests.post(
        f"{_REDDIT_BASE_URL}/api/v1/access_token",
        auth=(client_id, client_secret),
        headers={"User-Agent": _REDDIT_USER_AGENT},
        data={"grant_type": "client_credentials"},
    )
    r.raise_for_status()
    return r.json()["access_token"]


def _reddit_extract_image_url(post: dict) -> str:
    """Extract the best available image URL from a Reddit post."""
    if post.get("post_hint") == "image":
        return post.get("url", "")
    preview_images = post.get("preview", {}).get("images", [])
    if preview_images:
        url = preview_images[0].get("source", {}).get("url", "")
        return url.replace("&amp;", "&")
    return ""


def _reddit_media_type(post: dict) -> str:
    """Derive a media_type string consistent with other platform fetchers."""
    if post.get("crosspost_parent"):
        return "REPOST_FACADE"  # consistent with Bluesky repost detection
    hint = post.get("post_hint", "")
    if hint == "image":
        return "IMAGE"
    if hint in ("rich:video", "hosted:video"):
        return "VIDEO"
    if post.get("is_self", False):
        return "TEXT_POST"
    return "LINK"


def _reddit_post_to_row(post: dict) -> dict:
    """Convert a Reddit API post object to the standard 11-column row dict."""
    title = post.get("title", "")
    selftext = post.get("selftext", "")
    if selftext in ("[deleted]", "[removed]"):
        selftext = ""
    text = f"{title}\n\n{selftext}".strip() if selftext else title

    created_utc = post.get("created_utc", 0)
    ts_str = (
        datetime.fromtimestamp(created_utc, tz=timezone.utc).isoformat()
        if created_utc
        else ""
    )

    return {
        "post_id":    f"t3_{post.get('id', '')}",
        "timestamp":  ts_str,
        "media_type": _reddit_media_type(post),
        "text":       text,
        "image_url":  _reddit_extract_image_url(post),
        "likes":      post.get("score", 0),
        "replies":    post.get("num_comments", 0),
        "reposts":    post.get("num_crossposts", 0),
        "quotes":     0,
        "views":      0,
        "shares":     0,
    }


def _reddit_paginate(url: str, headers: dict, limit: int, months: int = None) -> list:
    """Paginate a Reddit listing endpoint and return post rows."""
    cutoff = None
    if months is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=months * 30)

    rows = []
    after = None

    while True:
        params = {"limit": 100, "sort": "new"}
        if after:
            params["after"] = after

        r = requests.get(url, headers=headers, params=params)
        r.raise_for_status()

        body = r.json().get("data", {})
        children = body.get("children", [])
        if not children:
            break

        for child in children:
            post = child.get("data", {})
            created_utc = post.get("created_utc", 0)
            if cutoff and created_utc:
                ts = datetime.fromtimestamp(created_utc, tz=timezone.utc)
                if ts < cutoff:
                    return rows  # posts are newest-first; stop once past window

            rows.append(_reddit_post_to_row(post))

            if months is None and len(rows) >= limit:
                return rows

        after = body.get("after")
        if not after:
            break

    return rows


def fetch_reddit(limit: int = 50, months: int = None, credentials: dict = None) -> pd.DataFrame:
    """
    Fetch Reddit posts from a subreddit or user profile.

    Supports two access modes:
    - **Public** (no OAuth): Pass ``subreddit`` or ``username`` in credentials.
      Uses the unauthenticated JSON API — no app registration required.
      Rate limit: ~10 requests/minute.
    - **OAuth** (higher rate limits): Also pass ``client_id`` and
      ``client_secret`` from reddit.com/prefs/apps. No user password needed
      for public profiles. Rate limit: 60 requests/minute.

    Args:
        limit (int): Maximum number of posts to fetch. Default 50.
            Ignored when months is set.
        months (int): If set, fetch all posts from the last N months.
        credentials (dict): Dict with access info. Keys:
            - ``subreddit`` (str): Subreddit name, e.g. ``"MachineLearning"``
              or ``"r/MachineLearning"`` (the ``r/`` prefix is stripped).
            - ``username`` (str): Reddit username, e.g. ``"chrissoria"``
              or ``"u/chrissoria"`` (the ``u/`` prefix is stripped).
            - ``client_id`` (str): OAuth app client ID (optional).
            - ``client_secret`` (str): OAuth app client secret (optional).
            Falls back to env vars: REDDIT_SUBREDDIT, REDDIT_USERNAME,
            REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET.

    Returns:
        pd.DataFrame with columns:
            post_id, timestamp, media_type, text, image_url,
            likes, replies, reposts, quotes, views, shares
        (quotes, views, shares are always 0 — not exposed by Reddit's API)

    Examples:
        >>> # Public subreddit (no credentials needed)
        >>> df = fetch_reddit(limit=25, credentials={"subreddit": "MachineLearning"})

        >>> # Public user profile (no credentials needed)
        >>> df = fetch_reddit(limit=50, credentials={"username": "chrissoria"})

        >>> # OAuth for higher rate limits
        >>> df = fetch_reddit(
        ...     limit=200,
        ...     credentials={
        ...         "username": "chrissoria",
        ...         "client_id": "abc123",
        ...         "client_secret": "xyz789",
        ...     },
        ... )
    """
    creds = _load_reddit_credentials(credentials)
    has_oauth = "client_id" in creds and "client_secret" in creds

    raw_username = creds.get("username", "")
    raw_subreddit = creds.get("subreddit", "")
    username = (raw_username[2:] if raw_username.startswith("u/") else raw_username).strip()
    subreddit = (raw_subreddit[2:] if raw_subreddit.startswith("r/") else raw_subreddit).strip()

    if not username and not subreddit:
        raise ValueError(
            "Reddit: provide 'subreddit' or 'username' in sm_credentials.\n"
            "  Subreddit:   sm_credentials={'subreddit': 'MachineLearning'}\n"
            "  User posts:  sm_credentials={'username': 'chrissoria'}\n"
            "  OAuth:       also add 'client_id' and 'client_secret'"
        )

    headers = {"User-Agent": _REDDIT_USER_AGENT}

    if has_oauth:
        token = _reddit_oauth_token(creds["client_id"], creds["client_secret"])
        headers["Authorization"] = f"Bearer {token}"
        base = _REDDIT_OAUTH_URL
        url = (
            f"{base}/user/{username}/submitted"
            if username
            else f"{base}/r/{subreddit}/new"
        )
    else:
        url = (
            f"{_REDDIT_BASE_URL}/user/{username}/submitted.json"
            if username
            else f"{_REDDIT_BASE_URL}/r/{subreddit}/new.json"
        )

    rows = _reddit_paginate(url, headers, limit=limit, months=months)
    if not rows:
        return pd.DataFrame(
            columns=["post_id", "timestamp", "media_type", "text", "image_url",
                     "likes", "replies", "reposts", "quotes", "views", "shares"]
        )
    return pd.DataFrame(rows)


# Dispatcher — extend this dict as new platforms are added
_FETCHERS = {
    "threads": fetch_threads,
    "bluesky": fetch_bluesky,
    "reddit":  fetch_reddit,
}

SUPPORTED_SOURCES = list(_FETCHERS.keys())


def fetch_social_media(sm_source: str, limit: int = 50, months: int = None, credentials: dict = None, handle: str = None) -> pd.DataFrame:
    """
    Fetch feed data from a social media platform.

    Args:
        sm_source (str): Platform name. Currently supported: "threads", "bluesky".
        limit (int): Number of posts/items to fetch. Ignored when months is set.
        months (int): If set, fetch all posts from the last N months.
        credentials (dict): Platform-specific credentials. Falls back to env vars.
        handle (str): Bluesky only. Handle of the account to fetch posts from
            (e.g. "user.bsky.social"). When omitted, fetches the authenticated
            user's own posts.

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
    if key == "bluesky":
        return fetch_bluesky(limit=limit, months=months, credentials=credentials, handle=handle)
    return _FETCHERS[key](limit=limit, months=months, credentials=credentials)
