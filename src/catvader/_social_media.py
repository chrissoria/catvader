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

Supported sources: "threads", "bluesky", "reddit", "mastodon", "youtube"
"""

import html
import os
import time
from html.parser import HTMLParser
from urllib.parse import urlparse

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
# Mastodon
# =============================================================================

class _HTMLStripper(HTMLParser):
    """Minimal HTML stripper used for Mastodon post content."""
    def __init__(self):
        super().__init__()
        self._parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("br", "p"):
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "p":
            self._parts.append("\n")

    def handle_data(self, data):
        self._parts.append(data)

    def result(self):
        import re as _re
        text = html.unescape("".join(self._parts))
        return _re.sub(r"\n{3,}", "\n\n", text).strip()


def _mastodon_strip_html(text: str) -> str:
    """Strip HTML tags from Mastodon post content and decode entities."""
    if not text:
        return ""
    stripper = _HTMLStripper()
    stripper.feed(text)
    return stripper.result()


def _mastodon_parse_handle(handle: str) -> tuple:
    """
    Parse a Mastodon handle into (instance, username).

    Accepts:
        "user@mastodon.social"
        "@user@mastodon.social"
        "https://mastodon.social/@user"
    Returns:
        ("mastodon.social", "user")
    """
    handle = handle.strip()
    if handle.startswith("http"):
        parsed = urlparse(handle)
        instance = parsed.netloc
        username = parsed.path.lstrip("/@")
        return instance, username
    handle = handle.lstrip("@")
    if "@" not in handle:
        raise ValueError(
            f"Invalid Mastodon handle: '{handle}'. "
            "Use the format 'user@instance.social' (e.g. 'user@mastodon.social')."
        )
    username, instance = handle.split("@", 1)
    return instance, username


def _mastodon_lookup_account(instance: str, username: str) -> dict:
    """Look up a Mastodon account by username on the given instance."""
    r = requests.get(
        f"https://{instance}/api/v1/accounts/lookup",
        params={"acct": username},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def _mastodon_extract_image_url(status: dict) -> str:
    """Extract the first image or video thumbnail URL from a Mastodon status."""
    for attachment in status.get("media_attachments", []):
        if attachment.get("type") in ("image", "gifv"):
            return attachment.get("url", "")
        if attachment.get("type") == "video":
            return attachment.get("preview_url", "")
    return ""


def _mastodon_media_type(status: dict) -> str:
    """Derive a media_type string consistent with other platform fetchers."""
    if status.get("reblog"):
        return "REPOST_FACADE"
    attachments = status.get("media_attachments", [])
    if any(a.get("type") == "video" for a in attachments):
        return "VIDEO"
    if any(a.get("type") in ("image", "gifv") for a in attachments):
        return "IMAGE"
    return "TEXT_POST"


def fetch_mastodon(limit: int = 50, months: int = None, handle: str = None) -> pd.DataFrame:
    """
    Fetch recent posts from a public Mastodon account (no authentication needed).

    Args:
        limit (int): Maximum number of posts to fetch. Default 50.
            Ignored when months is set.
        months (int): If set, only return posts from the last N months.
        handle (str): Mastodon handle in the format 'user@instance.social'
            (e.g. 'Gargron@mastodon.social'). Leading '@' is optional.

    Returns:
        pd.DataFrame with columns:
            post_id, timestamp, media_type, text, image_url,
            likes, replies, reposts, quotes, views, shares
        (quotes, views, shares are 0 — not exposed by the Mastodon API)
    """
    if not handle:
        raise ValueError(
            "Mastodon requires a handle. Pass sm_handle='user@instance.social'."
        )
    instance, username = _mastodon_parse_handle(handle)
    account = _mastodon_lookup_account(instance, username)
    account_id = account["id"]

    cutoff = None
    if months is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=months * 30)

    rows = []
    max_id = None
    page_size = min(limit if months is None else 40, 40)  # Mastodon API max is 40

    while True:
        params = {"limit": page_size}
        if max_id:
            params["max_id"] = max_id

        r = requests.get(
            f"https://{instance}/api/v1/accounts/{account_id}/statuses",
            params=params,
            timeout=15,
        )
        r.raise_for_status()
        statuses = r.json()

        if not statuses:
            break

        for status in statuses:
            ts_str = status.get("created_at", "")
            if cutoff and ts_str:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if ts < cutoff:
                    return pd.DataFrame(rows)

            # For reblogs use the original content
            reblog = status.get("reblog")
            raw_content = reblog.get("content", "") if reblog else status.get("content", "")

            rows.append({
                "post_id":    status.get("id", ""),
                "timestamp":  ts_str,
                "media_type": _mastodon_media_type(status),
                "text":       _mastodon_strip_html(raw_content),
                "image_url":  _mastodon_extract_image_url(status),
                "likes":      status.get("favourites_count", 0),
                "replies":    status.get("replies_count", 0),
                "reposts":    status.get("reblogs_count", 0),
                "quotes":     0,
                "views":      0,
                "shares":     0,
            })

            if months is None and len(rows) >= limit:
                return pd.DataFrame(rows)

        max_id = statuses[-1]["id"]

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


def _reddit_estimate_votes(score: int, ratio: float) -> tuple:
    """
    Estimate raw upvotes and downvotes from net score and upvote ratio.

    Reddit only exposes the net score (upvotes - downvotes) and the ratio
    (upvotes / total votes). From these we can derive approximate counts.
    Reddit fuzzes both values slightly to prevent brigading, so results
    are estimates, not exact figures.

    Returns (upvotes_raw, downvotes_est) as ints.
    """
    if ratio is None or score is None:
        return 0, 0
    denom = 2 * ratio - 1
    if abs(denom) < 0.01:   # ratio ~0.5 — can't reliably estimate
        return max(0, score), 0
    upvotes = round(score * ratio / denom)
    downvotes = max(0, upvotes - score)
    return max(0, upvotes), downvotes


def _reddit_post_to_row(post: dict) -> dict:
    """Convert a Reddit API post object to the standard row dict."""
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

    score = post.get("score", 0)
    ratio = post.get("upvote_ratio")
    upvotes_raw, downvotes_est = _reddit_estimate_votes(score, ratio)

    return {
        "post_id":        f"t3_{post.get('id', '')}",
        "timestamp":      ts_str,
        "media_type":     _reddit_media_type(post),
        "text":           text,
        "image_url":      _reddit_extract_image_url(post),
        "likes":          score,
        "replies":        post.get("num_comments", 0),
        "reposts":        post.get("num_crossposts", 0),
        "quotes":         0,
        "views":          0,
        "shares":         0,
        "upvote_ratio":   ratio if ratio is not None else 0.0,
        "upvotes_raw":    upvotes_raw,
        "downvotes_est":  downvotes_est,
    }


def _reddit_paginate(url: str, headers: dict, limit: int, months: int = None, days: int = None, request_delay: float = 6.0) -> list:
    """Paginate a Reddit listing endpoint and return post rows.

    Args:
        days: If set, fetch posts from the last N days (overrides months).
        request_delay: Seconds to sleep between paginated requests.
            Default 6.0 (10 req/min, safe for unauthenticated access).
            Pass 1.0 for OAuth (60 req/min).
    """
    cutoff = None
    if days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    elif months is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=months * 30)

    rows = []
    after = None
    first_request = True

    while True:
        if not first_request:
            time.sleep(request_delay)
        first_request = False

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


def fetch_reddit(limit: int = 50, months: int = None, days: int = None, credentials: dict = None) -> pd.DataFrame:
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

    # TODO: Add comment fetching support.
    # Each post's top-level comments are at /r/{sub}/comments/{post_id}.json
    # (or /comments/{post_id} on oauth.reddit.com). Design decisions needed:
    #   - How many top-level comments per post to fetch (e.g. top N by score)?
    #   - Whether to include nested reply threads or flatten to top-level only.
    #   - How to represent comments in the output DataFrame (one row per comment,
    #     with parent post_id as a foreign key column).
    # Would enable analysis of r/AskReddit answers, not just the questions.

    request_delay = 1.0 if has_oauth else 6.0
    rows = _reddit_paginate(url, headers, limit=limit, months=months, days=days, request_delay=request_delay)
    if not rows:
        return pd.DataFrame(
            columns=["post_id", "timestamp", "media_type", "text", "image_url",
                     "likes", "replies", "reposts", "quotes", "views", "shares"]
        )
    return pd.DataFrame(rows)


# =============================================================================
# YouTube
# =============================================================================

_YOUTUBE_BASE_URL = "https://www.googleapis.com/youtube/v3"


def _load_youtube_credentials(credentials: dict = None) -> str:
    """Return YouTube API key, preferring explicit credentials dict.

    Falls back in order:
      1. sm_credentials={'api_key': '...'}
      2. YOUTUBE_API_KEY env var
      3. GOOGLE_API_KEY env var (if YouTube Data API v3 is enabled for that key)
    """
    load_dotenv(_ENV_PATH, override=True)
    api_key = (
        (credentials or {}).get("api_key")
        or os.getenv("YOUTUBE_API_KEY")
        or os.getenv("YOUTUBE_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if not api_key:
        raise ValueError(
            "No YouTube API key found. Pass sm_credentials={'api_key': '...'} "
            f"or set YOUTUBE_API_KEY (or GOOGLE_API_KEY) in {_ENV_PATH}\n"
            "  Get a free key at https://console.cloud.google.com/ "
            "(enable YouTube Data API v3)"
        )
    return api_key


def _youtube_get_channel_id(handle: str, api_key: str) -> str:
    """Resolve a YouTube channel handle or URL to a channel ID."""
    handle = handle.strip()

    # Strip full URL if given
    if "youtube.com" in handle:
        parsed = urlparse(handle)
        path = parsed.path.lstrip("/")
        if path.startswith("@"):
            handle = path                              # e.g. "@h3productions"
        elif path.startswith("channel/"):
            return path.split("channel/")[1].split("/")[0]
        elif path.startswith("user/"):
            handle = path.split("user/")[1].split("/")[0]

    # Already a channel ID
    if handle.startswith("UC") and len(handle) == 24:
        return handle

    # forHandle lookup (new @ handles and bare names)
    lookup_handle = handle if handle.startswith("@") else f"@{handle}"
    r = requests.get(
        f"{_YOUTUBE_BASE_URL}/channels",
        params={"part": "id", "forHandle": lookup_handle, "key": api_key},
        timeout=10,
    )
    if r.status_code == 200:
        items = r.json().get("items", [])
        if items:
            return items[0]["id"]

    raise ValueError(
        f"Could not resolve YouTube channel handle: '{handle}'.\n"
        "Use formats like '@h3productions', 'h3productions', or 'UCxxxxxx'."
    )


def _youtube_get_uploads_playlist(channel_id: str, api_key: str) -> str:
    """Get the uploads playlist ID for a channel."""
    r = requests.get(
        f"{_YOUTUBE_BASE_URL}/channels",
        params={"part": "contentDetails", "id": channel_id, "key": api_key},
        timeout=10,
    )
    r.raise_for_status()
    items = r.json().get("items", [])
    if not items:
        raise ValueError(f"No YouTube channel found with ID: {channel_id}")
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def _youtube_parse_duration(iso_duration: str) -> int:
    """Convert ISO 8601 duration (e.g. 'PT1H23M45S') to total seconds."""
    import re as _re
    if not iso_duration:
        return 0
    m = _re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso_duration)
    if not m:
        return 0
    h, mn, s = (int(x or 0) for x in m.groups())
    return h * 3600 + mn * 60 + s


def _youtube_get_video_stats(video_ids: list, api_key: str) -> dict:
    """
    Batch-fetch statistics, duration, and tags for up to 50 video IDs.
    Returns {video_id: dict} with keys: likeCount, viewCount, commentCount,
    duration_seconds, tags.
    """
    r = requests.get(
        f"{_YOUTUBE_BASE_URL}/videos",
        params={
            "part": "statistics,contentDetails,snippet",
            "id": ",".join(video_ids),
            "key": api_key,
        },
        timeout=15,
    )
    r.raise_for_status()
    result = {}
    for item in r.json().get("items", []):
        stats   = item.get("statistics",     {})
        details = item.get("contentDetails", {})
        snip    = item.get("snippet",        {})
        result[item["id"]] = {
            **stats,
            "duration_seconds": _youtube_parse_duration(details.get("duration", "")),
            "tags":             snip.get("tags", []),
        }
    return result


def _youtube_get_transcript(video_id: str, max_chars: int = 10_000) -> str | None:
    """
    Fetch the auto-generated or manual transcript for a YouTube video.

    Returns the joined transcript text (up to max_chars), or None if
    transcripts are unavailable (disabled, private, or not generated yet).

    Requires: pip install youtube-transcript-api>=1.0
    Supports both v0.x (get_transcript class method) and v1.x (fetch instance method).
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        # v1.x uses instance method; v0.x used class method get_transcript
        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            segments = YouTubeTranscriptApi.get_transcript(video_id)
            text = " ".join(seg["text"] for seg in segments)
        else:
            api = YouTubeTranscriptApi()
            fetched = api.fetch(video_id)
            text = " ".join(s.text for s in fetched)
        if max_chars is not None and len(text) > max_chars:
            text = text[:max_chars] + "..."
        return text
    except Exception:
        return None


def _youtube_fetch_video_comments(
    video_id: str,
    video_title: str,
    api_key: str,
    max_comments: int = 20,
    video_stats: dict = None,
) -> list:
    """
    Fetch the top N comments for a single video (ordered by relevance).

    Each comment row carries the parent video's aggregate stats as
    'video_*' columns (video_likes, video_views, video_comment_count,
    video_title, video_id), making it easy to use video-level features
    as covariates in comment-level analyses.

    Returns a list of row dicts compatible with the standard column schema.
    Comments disabled on a video are silently skipped (returns []).
    """
    vstats = video_stats or {}
    rows = []
    next_page_token = None

    while len(rows) < max_comments:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "order": "relevance",
            "maxResults": min(max_comments - len(rows), 100),
            "key": api_key,
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        r = requests.get(
            f"{_YOUTUBE_BASE_URL}/commentThreads",
            params=params,
            timeout=15,
        )
        if r.status_code in (403, 404):
            break  # comments disabled or video unavailable
        r.raise_for_status()
        body = r.json()

        for item in body.get("items", []):
            top = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
            rows.append({
                # Standard columns — refer to the comment itself
                "post_id":              item.get("id", ""),
                "timestamp":            top.get("publishedAt", ""),
                "media_type":           "COMMENT",
                "text":                 _mastodon_strip_html(top.get("textDisplay", "")),
                "image_url":            "",
                "likes":                top.get("likeCount", 0),
                "replies":              item.get("snippet", {}).get("totalReplyCount", 0),
                "reposts":              0,
                "quotes":               0,
                "views":                0,
                "shares":               0,
                # Video-level context columns — treat as covariates
                "video_id":               video_id,
                "video_title":            video_title,
                "video_likes":            vstats.get("likes",            0),
                "video_views":            vstats.get("views",            0),
                "video_comment_count":    vstats.get("comment_count",    0),
                "video_duration_seconds": vstats.get("duration_seconds", 0),
                "video_tags":             vstats.get("tags",             []),
            })
            if len(rows) >= max_comments:
                return rows

        next_page_token = body.get("nextPageToken")
        if not next_page_token:
            break

    return rows


def fetch_youtube(
    limit: int = 50,
    months: int = None,
    credentials: dict = None,
    handle: str = None,
    content: str = "video",
    use_transcript: bool = False,
    comments_per_video: int = 20,
    transcript_max_chars: int = 10_000,
) -> pd.DataFrame:
    """
    Fetch content from a YouTube channel.

    Args:
        limit (int): Number of videos to fetch. In "comments" mode, the number
            of videos to pull comments from. Default 50. Ignored when months is set.
        months (int): If set, only return content from the last N months.
        credentials (dict): Dict with 'api_key' key.
            Falls back to YOUTUBE_API_KEY env var.
        handle (str): YouTube channel handle, e.g. '@h3productions',
            'h3productions', a channel ID ('UCxxxxxx'), or a full channel URL.
        content (str): Unit of analysis:
            - "video" (default): one row per video.
            - "comments": one row per comment; video-level stats travel as
              video_* covariate columns (video_likes, video_views, etc.).
        use_transcript (bool): Video mode only. When True, use the
            auto-generated transcript as the text column instead of the
            description. Falls back to description if unavailable. Default False.
        comments_per_video (int): Comments mode only. Max top-level comments
            per video. Default 20.
        transcript_max_chars (int): Video transcript mode only. Maximum number of
            characters to include from the transcript. Default 10,000. Set to None
            for the full transcript (can be 100k+ chars for long videos).

    Returns:
        pd.DataFrame with columns:
            post_id, timestamp, media_type, text, image_url,
            likes, replies, reposts, quotes, views, shares
        When content="comments": also includes video_id, video_title.
        (reposts, quotes, shares are always 0 — not in the YouTube API)
    """
    if not handle:
        raise ValueError(
            "YouTube requires a channel handle. "
            "Pass sm_handle='@h3productions' or sm_handle='UCxxxxxx'."
        )
    api_key = _load_youtube_credentials(credentials)
    channel_id = _youtube_get_channel_id(handle, api_key)
    uploads_playlist = _youtube_get_uploads_playlist(channel_id, api_key)

    cutoff = None
    if months is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=months * 30)

    rows = []
    next_page_token = None
    page_size = min(limit if months is None else 50, 50)

    while True:
        params = {
            "part": "snippet",
            "playlistId": uploads_playlist,
            "maxResults": page_size,
            "key": api_key,
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        r = requests.get(
            f"{_YOUTUBE_BASE_URL}/playlistItems",
            params=params,
            timeout=15,
        )
        r.raise_for_status()
        body = r.json()
        items = body.get("items", [])
        if not items:
            break

        page_rows = []
        video_ids = []

        for item in items:
            snippet = item.get("snippet", {})
            ts_str = snippet.get("publishedAt", "")

            if cutoff and ts_str:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if ts < cutoff:
                    if video_ids and content != "comments":
                        stats_map = _youtube_get_video_stats(video_ids, api_key)
                        for row in page_rows:
                            s = stats_map.get(row["post_id"], {})
                            row["likes"]            = int(s.get("likeCount",       0) or 0)
                            row["replies"]          = int(s.get("commentCount",    0) or 0)
                            row["views"]            = int(s.get("viewCount",       0) or 0)
                            row["duration_seconds"] = int(s.get("duration_seconds", 0) or 0)
                            row["tags"]             = s.get("tags", [])
                    rows.extend(page_rows)
                    return pd.DataFrame(rows)

            video_id = snippet.get("resourceId", {}).get("videoId", "")
            title    = snippet.get("title", "")
            thumbnails = snippet.get("thumbnails", {})
            image_url = (
                thumbnails.get("maxres", {}).get("url")
                or thumbnails.get("high", {}).get("url")
                or thumbnails.get("default", {}).get("url")
                or ""
            )

            if content == "comments":
                # Defer to comment fetching below — just record video metadata
                page_rows.append({"_video_id": video_id, "_title": title})
            else:
                description = snippet.get("description", "")
                if use_transcript:
                    transcript = _youtube_get_transcript(video_id, max_chars=transcript_max_chars)
                    if transcript:
                        text = f"{title}\n\n{transcript}"
                    else:
                        if len(description) > 500:
                            description = description[:500] + "..."
                        text = f"{title}\n\n{description}".strip() if description else title
                else:
                    if len(description) > 500:
                        description = description[:500] + "..."
                    text = f"{title}\n\n{description}".strip() if description else title

                page_rows.append({
                    "post_id":          video_id,
                    "timestamp":        ts_str,
                    "media_type":       "VIDEO",
                    "text":             text,
                    "image_url":        image_url,
                    "likes":            0,
                    "replies":          0,
                    "reposts":          0,
                    "quotes":           0,
                    "views":            0,
                    "shares":           0,
                    "duration_seconds": 0,
                    "tags":             [],
                })
                video_ids.append(video_id)

        if content == "comments":
            # Batch-fetch video stats so they can travel with each comment row
            page_video_ids = [m["_video_id"] for m in page_rows]
            page_stats_map = _youtube_get_video_stats(page_video_ids, api_key) if page_video_ids else {}

            for meta in page_rows:
                vid = meta["_video_id"]
                s = page_stats_map.get(vid, {})
                vstats = {
                    "likes":            int(s.get("likeCount",       0) or 0),
                    "views":            int(s.get("viewCount",        0) or 0),
                    "comment_count":    int(s.get("commentCount",     0) or 0),
                    "duration_seconds": int(s.get("duration_seconds", 0) or 0),
                    "tags":             s.get("tags", []),
                }
                comment_rows = _youtube_fetch_video_comments(
                    vid, meta["_title"], api_key,
                    max_comments=comments_per_video,
                    video_stats=vstats,
                )
                rows.extend(comment_rows)
            # For comments mode, limit is interpreted as number of videos
            if months is None and len(page_rows) >= limit:
                return pd.DataFrame(rows)
        else:
            # Batch-fetch stats for video/transcript modes
            if video_ids:
                stats_map = _youtube_get_video_stats(video_ids, api_key)
                for row in page_rows:
                    s = stats_map.get(row["post_id"], {})
                    row["likes"]            = int(s.get("likeCount",       0) or 0)
                    row["replies"]          = int(s.get("commentCount",    0) or 0)
                    row["views"]            = int(s.get("viewCount",       0) or 0)
                    row["duration_seconds"] = int(s.get("duration_seconds", 0) or 0)
                    row["tags"]             = s.get("tags", [])

            rows.extend(page_rows)

            if months is None and len(rows) >= limit:
                return pd.DataFrame(rows[:limit])

        next_page_token = body.get("nextPageToken")
        if not next_page_token:
            break

    return pd.DataFrame(rows)


# =============================================================================
# LinkedIn
# =============================================================================

_LINKEDIN_AUTH_URL      = "https://www.linkedin.com/oauth/v2/authorization"
_LINKEDIN_TOKEN_URL     = "https://www.linkedin.com/oauth/v2/accessToken"
_LINKEDIN_API_BASE      = "https://api.linkedin.com/v2"
_LINKEDIN_REDIRECT_PORT = 8765
_LINKEDIN_REDIRECT_URI  = f"http://localhost:{_LINKEDIN_REDIRECT_PORT}/callback"
_LINKEDIN_SCOPES        = "r_liteprofile r_member_social"


def _save_env_var(key: str, value: str) -> None:
    """Append or update a KEY="value" line in the project .env file."""
    lines = []
    found = False
    if os.path.exists(_ENV_PATH):
        with open(_ENV_PATH, "r") as fh:
            lines = fh.readlines()
        for i, line in enumerate(lines):
            if line.startswith(f"{key}="):
                lines[i] = f'{key}="{value}"\n'
                found = True
                break
    if not found:
        lines.append(f'{key}="{value}"\n')
    with open(_ENV_PATH, "w") as fh:
        fh.writelines(lines)


def _linkedin_oauth_flow(client_id: str, client_secret: str) -> str:
    """Open a browser for LinkedIn OAuth 2.0 and return an access token."""
    import secrets as _secrets
    import threading
    import webbrowser
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import parse_qs, urlencode, urlparse as _urlparse

    state = _secrets.token_urlsafe(16)
    _result: dict = {}

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            qs = parse_qs(_urlparse(self.path).query)
            if "code" in qs:
                _result["code"] = qs["code"][0]
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                b"<h2>LinkedIn authorization complete.</h2>"
                b"<p>You can close this tab and return to your terminal.</p>"
            )
            threading.Thread(target=self.server.shutdown, daemon=True).start()

        def log_message(self, *args):
            pass  # suppress request logs

    auth_url = (
        _LINKEDIN_AUTH_URL
        + "?"
        + urlencode({
            "response_type": "code",
            "client_id":     client_id,
            "redirect_uri":  _LINKEDIN_REDIRECT_URI,
            "scope":         _LINKEDIN_SCOPES,
            "state":         state,
        })
    )

    print(
        f"\n[LinkedIn] Opening browser for OAuth authorization...\n"
        f"  If it doesn't open automatically, visit:\n  {auth_url}\n"
    )
    webbrowser.open(auth_url)

    server = HTTPServer(("localhost", _LINKEDIN_REDIRECT_PORT), _Handler)
    server.serve_forever()  # blocks until handler calls shutdown()

    if "code" not in _result:
        raise RuntimeError(
            "LinkedIn OAuth failed: no authorization code received.\n"
            "Make sure the redirect URI in your LinkedIn app settings is:\n"
            f"  {_LINKEDIN_REDIRECT_URI}"
        )

    r = requests.post(
        _LINKEDIN_TOKEN_URL,
        data={
            "grant_type":   "authorization_code",
            "code":         _result["code"],
            "redirect_uri": _LINKEDIN_REDIRECT_URI,
            "client_id":    client_id,
            "client_secret": client_secret,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    r.raise_for_status()
    token_data = r.json()
    access_token = token_data.get("access_token")
    if not access_token:
        raise RuntimeError(f"LinkedIn token exchange failed: {token_data}")
    return access_token


def _load_linkedin_credentials(credentials: dict = None) -> str:
    """Return a valid LinkedIn access token.

    On first run (no stored token), opens a browser OAuth flow and saves
    the resulting token to the project .env file.
    """
    load_dotenv(_ENV_PATH, override=True)
    creds = dict(credentials or {})

    client_id     = creds.get("client_id")     or os.getenv("LINKEDIN_CLIENT_ID")
    client_secret = creds.get("client_secret") or os.getenv("LINKEDIN_CLIENT_SECRET")
    access_token  = creds.get("access_token")  or os.getenv("LINKEDIN_ACCESS_TOKEN")

    if not client_id or not client_secret:
        raise ValueError(
            "LinkedIn requires a client_id and client_secret.\n"
            "  1. Register an app at https://www.linkedin.com/developers/apps\n"
            "  2. Add the 'Share on LinkedIn' and 'Sign In with LinkedIn' products\n"
            "     (grants r_liteprofile + r_member_social scopes)\n"
            f"  3. Add to {_ENV_PATH}:\n"
            "       LINKEDIN_CLIENT_ID=\"your-client-id\"\n"
            "       LINKEDIN_CLIENT_SECRET=\"your-client-secret\"\n"
            f"  4. Set redirect URI to: {_LINKEDIN_REDIRECT_URI}\n"
            "     in your app's OAuth 2.0 settings on the LinkedIn Developer Portal."
        )

    if not access_token:
        print("[LinkedIn] No access token found — starting OAuth flow...")
        access_token = _linkedin_oauth_flow(client_id, client_secret)
        _save_env_var("LINKEDIN_ACCESS_TOKEN", access_token)
        print(f"[LinkedIn] Access token saved to {_ENV_PATH} (valid ~60 days).")

    return access_token


def _linkedin_headers(access_token: str) -> dict:
    return {
        "Authorization":              f"Bearer {access_token}",
        "X-Restli-Protocol-Version":  "2.0.0",
        "LinkedIn-Version":           "202304",
    }


def _linkedin_media_type(post: dict) -> str:
    specific = post.get("specificContent", {})
    share    = specific.get("com.linkedin.ugc.ShareContent", {})
    category = share.get("shareMediaCategory", "NONE")
    if category == "IMAGE":
        return "IMAGE"
    if category == "VIDEO":
        return "VIDEO"
    if category == "ARTICLE":
        return "ARTICLE"
    return "TEXT_POST"


def _linkedin_extract_text(post: dict) -> str:
    specific   = post.get("specificContent", {})
    share      = specific.get("com.linkedin.ugc.ShareContent", {})
    commentary = share.get("shareCommentary", {})
    return commentary.get("text", "")


def _linkedin_extract_image_url(post: dict) -> str:
    specific   = post.get("specificContent", {})
    share      = specific.get("com.linkedin.ugc.ShareContent", {})
    media_list = share.get("media", [])
    if not media_list:
        return ""
    first      = media_list[0]
    thumbnails = first.get("thumbnails", [])
    if thumbnails:
        return thumbnails[0].get("url", "")
    return first.get("originalUrl", "")


def _linkedin_social_actions(post_urn: str, headers: dict, delay: float = 0.3) -> tuple:
    """Fetch (likes, comments) for a single post URN. Returns (0, 0) on error."""
    from urllib.parse import quote
    time.sleep(delay)
    encoded = quote(post_urn, safe="")
    r = requests.get(
        f"{_LINKEDIN_API_BASE}/socialActions/{encoded}",
        headers=headers,
    )
    if r.status_code != 200:
        return 0, 0
    data     = r.json()
    likes    = data.get("likesSummary",    {}).get("totalLikes",              0)
    comments = data.get("commentsSummary", {}).get("totalFirstLevelComments", 0)
    return likes, comments


def fetch_linkedin(limit: int = 50, months: int = None, credentials: dict = None) -> pd.DataFrame:
    """
    Fetch your own LinkedIn posts with engagement metrics.

    On first run, opens a browser window for OAuth 2.0 authorization and
    saves the access token to your .env file (valid ~60 days).

    Args:
        limit (int): Maximum number of posts to fetch. Default 50.
            Ignored when months is set.
        months (int): If set, fetch all posts from the last N months.
        credentials (dict): Optional. Keys:
            - 'client_id', 'client_secret': LinkedIn OAuth app credentials.
              Falls back to LINKEDIN_CLIENT_ID / LINKEDIN_CLIENT_SECRET env vars.
            - 'access_token': Use a pre-existing token and skip the OAuth flow.
              Falls back to LINKEDIN_ACCESS_TOKEN env var.

    Returns:
        pd.DataFrame with columns:
            post_id, timestamp, media_type, text, image_url,
            likes, replies, reposts, quotes, views, shares
        (reposts, quotes, views, shares are 0 — not exposed by LinkedIn's API)

    Notes:
        - Only your own posts are accessible via the personal LinkedIn API.
          Cross-account analysis is not supported.
        - Engagement metrics require one API call per post (~0.3 s/post).
        - If you see a 401 error, your token has expired (~60 days). Delete
          LINKEDIN_ACCESS_TOKEN from your .env and re-run to re-authorize.
    """
    from urllib.parse import quote

    access_token = _load_linkedin_credentials(credentials)
    headers      = _linkedin_headers(access_token)

    # Resolve the current user's person ID
    r = requests.get(f"{_LINKEDIN_API_BASE}/me", headers=headers)
    if r.status_code == 401:
        raise RuntimeError(
            "LinkedIn access token is invalid or expired.\n"
            f"Delete LINKEDIN_ACCESS_TOKEN from {_ENV_PATH} and re-run to re-authorize."
        )
    r.raise_for_status()
    person_id  = r.json()["id"]
    author_urn = f"urn:li:person:{person_id}"

    cutoff    = None
    if months is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=months * 30)

    rows      = []
    start     = 0
    page_size = min(limit if months is None else 50, 50)

    while True:
        encoded_urn = quote(author_urn, safe="")
        params = {
            "q":       "authors",
            "authors": f"List({encoded_urn})",
            "count":   page_size,
            "start":   start,
            "sortBy":  "LAST_MODIFIED",
        }
        r = requests.get(
            f"{_LINKEDIN_API_BASE}/ugcPosts",
            headers=headers,
            params=params,
        )
        if r.status_code == 401:
            raise RuntimeError(
                "LinkedIn access token expired.\n"
                f"Delete LINKEDIN_ACCESS_TOKEN from {_ENV_PATH} and re-run."
            )
        r.raise_for_status()
        body     = r.json()
        elements = body.get("elements", [])
        if not elements:
            break

        for post in elements:
            created_ms = post.get("created", {}).get("time", 0)
            ts_str = (
                datetime.fromtimestamp(created_ms / 1000, tz=timezone.utc).isoformat()
                if created_ms else ""
            )

            if cutoff and ts_str:
                ts = datetime.fromisoformat(ts_str)
                if ts < cutoff:
                    return pd.DataFrame(rows)

            post_urn       = post.get("id", "")
            likes, replies = _linkedin_social_actions(post_urn, headers) if post_urn else (0, 0)

            rows.append({
                "post_id":    post_urn,
                "timestamp":  ts_str,
                "media_type": _linkedin_media_type(post),
                "text":       _linkedin_extract_text(post),
                "image_url":  _linkedin_extract_image_url(post),
                "likes":      likes,
                "replies":    replies,
                "reposts":    0,
                "quotes":     0,
                "views":      0,
                "shares":     0,
            })

            if months is None and len(rows) >= limit:
                return pd.DataFrame(rows)

        paging = body.get("paging", {})
        total  = paging.get("total", 0)
        start += page_size
        if start >= total or len(elements) < page_size:
            break

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["post_id", "timestamp", "media_type", "text", "image_url",
                 "likes", "replies", "reposts", "quotes", "views", "shares"]
    )


# Dispatcher — extend this dict as new platforms are added
_FETCHERS = {
    "threads":   fetch_threads,
    "bluesky":   fetch_bluesky,
    "reddit":    fetch_reddit,
    "mastodon":  fetch_mastodon,
    "youtube":   fetch_youtube,
    "linkedin":  fetch_linkedin,
}

SUPPORTED_SOURCES = list(_FETCHERS.keys())


def fetch_social_media(
    sm_source: str,
    limit: int = 50,
    months: int = None,
    days: int = None,
    credentials: dict = None,
    handle: str = None,
    youtube_content: str = "video",
    youtube_transcript: bool = False,
    comments_per_video: int = 20,
    youtube_transcript_max_chars: int = 10_000,
) -> pd.DataFrame:
    """
    Fetch feed data from a social media platform.

    Args:
        sm_source (str): Platform name. Supported: "threads", "bluesky", "reddit",
            "mastodon", "youtube", "linkedin".
        limit (int): Number of posts/items to fetch. Ignored when months or days is set.
            For YouTube "comments" mode, interpreted as number of videos to pull
            comments from.
        months (int): If set, fetch all posts from the last N months.
        days (int): If set, fetch all posts from the last N days (overrides months).
        credentials (dict): Platform-specific credentials. Falls back to env vars.
        handle (str): Account handle for platforms that require one.
        youtube_content (str): YouTube only. Unit of analysis: "video" or "comments".
        youtube_transcript (bool): YouTube video mode only. Use auto-generated
            transcript as the text column instead of description. Default False.
        comments_per_video (int): YouTube comments mode only. Default 20.
        youtube_transcript_max_chars (int): YouTube transcript mode only. Max characters
            to include. Default 10,000. Set to None for full transcript. Default 10,000.

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
    if key == "reddit":
        return fetch_reddit(limit=limit, months=months, days=days, credentials=credentials)
    if key == "mastodon":
        return fetch_mastodon(limit=limit, months=months, handle=handle)
    if key == "youtube":
        return fetch_youtube(
            limit=limit,
            months=months,
            credentials=credentials,
            handle=handle,
            content=youtube_content,
            use_transcript=youtube_transcript,
            comments_per_video=comments_per_video,
            transcript_max_chars=youtube_transcript_max_chars,
        )
    return _FETCHERS[key](limit=limit, months=months, credentials=credentials)
