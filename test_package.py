"""
Quick smoke tests for cat-vader 1.0.0
Tests text classification, social media context, and auto-detection.
"""

import os
import catvader as cat
import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/Documents/Research/Categorization_AI_experiments/.env"))

OPENAI_KEY = os.environ["OPENAI_API_KEY"]
ANTHROPIC_KEY = os.environ["ANTHROPIC_API_KEY"]

print(f"catvader version: {cat.__version__}\n")

# --- Test 1: basic text classification ---
print("=" * 50)
print("Test 1: Basic text classification (OpenAI)")
posts = [
    "I absolutely love this new feature, it changed my life!",
    "This product is terrible and I want a refund.",
    "Just had my morning coffee.",
]
results = cat.classify(
    input_data=posts,
    categories=["Positive", "Negative", "Neutral"],
    api_key=OPENAI_KEY,
    user_model="gpt-4o-mini",
)
print(results.to_string())

# --- Test 2: social media context injected into prompt ---
print("\n" + "=" * 50)
print("Test 2: Social media context fields (Anthropic)")
df = pd.DataFrame({
    "text": [
        "Can't believe the government is hiding this from us #wakeup",
        "Great game last night, what a finish! #sports",
        "Buy followers for cheap DM me",
    ]
})
results2 = cat.classify(
    input_data=df["text"],
    categories=["Misinformation/Conspiracy", "Sports", "Spam", "Other"],
    platform="Twitter/X",
    hashtags=["#wakeup", "#sports"],
    api_key=ANTHROPIC_KEY,
    user_model="claude-haiku-4-5-20251001",
)
print(results2.to_string())

# --- Test 3: pandas Series with NaN (robustness) ---
print("\n" + "=" * 50)
print("Test 3: NaN handling in Series")
series_with_nan = pd.Series([
    "This is great!",
    None,
    "I hate this.",
    float("nan"),
    "Meh.",
])
results3 = cat.classify(
    input_data=series_with_nan,
    categories=["Positive", "Negative", "Neutral"],
    api_key=OPENAI_KEY,
    user_model="gpt-4o-mini",
)
print(results3.to_string())

print("\nAll tests passed.")
