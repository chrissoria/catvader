"""
Test Reddit sm_source — r/AskReddit, today's posts.
"""

import os
import catvader as cat
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/Documents/Research/Categorization_AI_experiments/.env"))

OPENAI_KEY = os.environ["OPENAI_API_KEY"]

print(f"catvader version: {cat.__version__}\n")
print("=" * 60)
print("Test: r/AskReddit — posts from last 24 hours, 10 posts")
print("=" * 60)

results = cat.classify(
    sm_source="reddit",
    sm_credentials={"subreddit": "AskReddit"},
    sm_days=1,
    sm_limit=10,
    categories=["Relationships", "Career/Work", "Health", "Advice", "Hypothetical/Fun", "Other"],
    api_key=OPENAI_KEY,
    user_model="gpt-4o-mini",
    add_other=False,
    check_verbosity=False,
)

print(f"\nShape: {results.shape}")
print(f"Columns: {list(results.columns)}\n")

display_cols = [
    "social_media_input", "post_id", "media_type",
    "likes", "upvotes_raw", "downvotes_est", "upvote_ratio",
    "replies", "reposts",
    "is_repost", "post_length", "contains_url",
    "day", "month", "hour",
]
print(results[display_cols].to_string())
