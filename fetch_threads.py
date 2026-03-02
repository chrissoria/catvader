"""
fetch_threads.py
Fetch your Threads posts with engagement metrics into a DataFrame.
"""

import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv("/Users/chrissoria/Documents/Important_Docs/social_media_bot/.env")

ACCESS_TOKEN = os.getenv("THREADS_ACCESS_TOKEN")
USER_ID      = os.getenv("THREADS_USER_ID")
BASE_URL     = "https://graph.threads.net/v1.0"


def get_posts(limit=5):
    """Fetch the most recent posts."""
    url = f"{BASE_URL}/{USER_ID}/threads"
    params = {
        "fields": "id,text,timestamp,media_type",
        "limit": limit,
        "access_token": ACCESS_TOKEN,
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json().get("data", [])


def get_insights(post_id):
    """Fetch engagement metrics for a single post."""
    url = f"{BASE_URL}/{post_id}/insights"
    params = {
        "metric": "likes,replies,reposts,quotes,views,shares,clicks",
        "access_token": ACCESS_TOKEN,
    }
    r = requests.get(url, params=params)
    if r.status_code != 200:
        print(f"  Insights error for {post_id}: {r.status_code} {r.text}")
        return {}
    data = r.json().get("data", [])
    return {item["name"]: item["values"][0]["value"] for item in data}


def fetch_posts_df(limit=5):
    """Return a DataFrame of recent posts with engagement metrics."""
    posts = get_posts(limit=limit)
    rows = []
    for post in posts:
        print(f"Fetching insights for post {post['id']}...")
        metrics = get_insights(post["id"])
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


if __name__ == "__main__":
    print("Fetching 5 most recent Threads posts...\n")
    df = fetch_posts_df(limit=5)
    pd.set_option("display.max_colwidth", 60)
    print(df.to_string(index=False))
