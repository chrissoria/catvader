"""
Test Reddit sm_source — r/AskReddit, 1000 posts.
"""

import os
import catvader as cat
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/Documents/Research/Categorization_AI_experiments/.env"))

OPENAI_KEY = os.environ["OPENAI_API_KEY"]

print(f"catvader version: {cat.__version__}\n")
print("=" * 60)
print("Test: r/AskReddit — 1000 posts")
print("=" * 60)

results = cat.classify(
    sm_source="reddit",
    sm_credentials={"subreddit": "AskReddit"},
    sm_limit=1000,
    categories=[
        "Relationships & Dating",
        "Career & Money",
        "Health & Body",
        "Hypothetical & Fun",
        "Advice & Life",
        "Other",
    ],
    api_key=OPENAI_KEY,
    user_model="gpt-4o-mini",
    add_other=False,
    check_verbosity=False,
)

print(f"\nShape: {results.shape}")
print(f"Columns: {list(results.columns)}\n")

print("--- Category distribution ---")
print(results["category_1"].value_counts())

print("\n--- Media type breakdown ---")
print(results["media_type"].value_counts())

print("\n--- Engagement stats ---")
print(results[["likes", "upvotes_raw", "downvotes_est", "replies"]].describe())

print("\n--- Sample (first 5 rows) ---")
print(results[["social_media_input", "category_1", "likes", "replies", "day", "hour"]].head().to_string())
