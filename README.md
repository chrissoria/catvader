# cat-vader

<p align="center">
  <img src="https://raw.githubusercontent.com/chrissoria/catvader/main/logo.png" alt="CatVader logo" width="180"/>
</p>

CatVader: An AI Pipeline for Classifying and Exploring Social Media Data

[![PyPI - Version](https://img.shields.io/pypi/v/cat-vader.svg)](https://pypi.org/project/cat-vader)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cat-vader.svg)](https://pypi.org/project/cat-vader)

-----

## The Problem

Social media data is messy and vast: millions of posts, comments, and threads that need to be categorized before any quantitative analysis can begin. Manual coding doesn't scale, and generic text classifiers miss the nuance of platform-specific language, hashtags, and metadata.

## The Solution

CatVader is a Python package designed specifically for social media research that uses LLMs to automate the classification and exploration of posts, comments, and threads. It handles both:

- **Category Assignment**: Classify posts into your predefined categories (multi-label supported)
- **Category Extraction**: Automatically discover and extract categories from your data when you don't have a predefined scheme
- **Category Exploration**: Analyze category stability and saturation through repeated raw extraction

Social media context (platform, author handle, hashtags, engagement metrics) can be injected directly into the classification prompt for richer, more accurate results.

-----

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Best Practices for Classification](#best-practices-for-classification)
- [Configuration](#configuration)
- [Supported Models](#supported-models)
- [API Reference](#api-reference)
  - [classify()](#classify) - Unified function for text, image, and PDF (auto-detects input type)
  - [extract()](#extract) - Unified function for category extraction
  - [explore()](#explore) - Raw category extraction for saturation analysis
  - [image_features()](#image_features)
- [Deprecated Functions](#deprecated-functions)
- [Contributing & Support](#contributing--support)
- [License](#license)

## Installation

```console
pip install cat-vader
```

For PDF support:
```console
pip install cat-vader[pdf]
```

-----

## Quick Start

**This package is designed for building datasets at scale**, not one-off queries. Its primary purpose is batch processing entire social media datasets into structured, analysis-ready DataFrames.

Simply provide your posts and category list — the package handles the rest and outputs clean data ready for statistical analysis. It works with single or multiple categories per post and automatically skips missing data to save API costs.

Also supports **image and PDF classification** using the same methodology.

All outputs are formatted for immediate statistical analysis and can be exported directly to CSV.

## Best Practices for Classification

These recommendations are based on empirical testing across multiple tasks and models (7B to frontier-class).

### What works

- **Detailed category descriptions**: The single biggest lever for accuracy. Instead of short labels like `"Anger"`, use verbose descriptions like `"Post expresses anger or frustration toward a person, institution, or situation."` This consistently improves accuracy across all models.
- **Include an "Other" category**: Adding a catch-all category prevents the model from forcing ambiguous posts into ill-fitting categories, improving precision.
- **Low temperature** (`creativity=0`): For classification tasks, deterministic output is generally preferable.
- **Social media context fields** (`platform`, `handle`, `hashtags`): Providing platform context helps the model interpret slang, tone, and conventions accurately.

### What doesn't help (or hurts)

- **Chain of Thought** (`chain_of_thought`): In testing, enabling CoT did not improve classification accuracy and slightly degraded it for some models.
- **Step-back prompting** (`step_back_prompt`): Results are inconsistent — slight gains for weaker models but slight losses for stronger models.
- **Context prompting** (`context_prompt`): Adds generic expert context. No consistent benefit observed.

### Summary

The most effective approach: **write detailed category descriptions, include an "Other" category, provide social media context, and use a capable model at low temperature.**

-----

## Configuration

### Get Your API Key

Get an API key from your preferred provider:

- **OpenAI**: [platform.openai.com](https://platform.openai.com)
- **Anthropic**: [console.anthropic.com](https://console.anthropic.com)
- **Google**: [aistudio.google.com](https://aistudio.google.com)
- **Huggingface**: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- **xAI**: [console.x.ai](https://console.x.ai)
- **Mistral**: [console.mistral.ai](https://console.mistral.ai)
- **Perplexity**: [perplexity.ai/settings/api](https://www.perplexity.ai/settings/api)

## Supported Models

- **OpenAI**: GPT-4o, GPT-4, GPT-5, etc.
- **Anthropic**: Claude Sonnet 4, Claude 3.5 Sonnet, Claude Haiku, etc.
- **Google**: Gemini 2.5 Flash, Gemini 2.5 Pro, etc.
- **Huggingface**: Qwen, Llama 4, DeepSeek, and thousands of community models
- **xAI**: Grok models
- **Mistral**: Mistral Large, Pixtral, etc.
- **Perplexity**: Sonar Large, Sonar Small, etc.

## API Reference

### `classify()`

Unified classification function for text, image, and PDF inputs. **Input type is auto-detected** from your data.

Supports both **single-model** and **multi-model ensemble** classification for improved accuracy through consensus voting.

**Parameters:**
- `input_data`: The data to classify (text list/Series, image paths, or PDF paths). Omit when using `sm_source`.
- `categories` (list): List of category names for classification. Use `"auto"` to discover categories automatically.
- `api_key` (str): API key for the LLM service (single-model mode)
- `sm_source` (str, optional): Social media platform to pull posts from automatically. Supported: `"threads"`, `"bluesky"`, `"reddit"`, `"mastodon"`, `"youtube"`. When set, `input_data` is fetched and does not need to be provided.
- `sm_limit` (int, default=`50`): Number of posts/videos to fetch. For YouTube comments mode, number of videos to pull comments from.
- `sm_months` (int, optional): Fetch all posts from the last N months instead of using `sm_limit`.
- `sm_days` (int, optional): Fetch all posts from the last N days (overrides `sm_months`).
- `sm_credentials` (dict, optional): Platform credentials (e.g., `{"access_token": "...", "user_id": "..."}`). Falls back to env vars.
- `sm_handle` (str, optional): Account handle for platforms that require one. Bluesky: `"user.bsky.social"`. Mastodon: `"user@instance.social"`. YouTube: `"@ChannelHandle"` or channel ID.
- `sm_timezone` (str, default=`"UTC"`): Timezone for the `day`, `month`, `hour`, and `n_posts_that_day` output columns. Any IANA timezone string (e.g., `"America/Los_Angeles"`, `"America/New_York"`, `"Europe/London"`).
- `sm_youtube_content` (str, default=`"video"`): YouTube only. Unit of analysis — `"video"` (one row per video) or `"comments"` (one row per comment, with video-level covariates).
- `sm_youtube_transcript` (bool, default=`False`): YouTube video mode only. Use the auto-generated transcript as the `text` column instead of the description. Falls back to description if unavailable. Requires `pip install youtube-transcript-api`.
- `sm_youtube_transcript_max_chars` (int, default=`10_000`): Max characters from the transcript. Set to `None` for the full transcript (can be 100k+ chars for long videos).
- `sm_comments_per_video` (int, default=`20`): YouTube comments mode only. Max top-level comments per video.
- `platform` (str, optional): Social media platform label (e.g., `"Twitter/X"`, `"Reddit"`, `"TikTok"`). Injected into the classification prompt as context.
- `handle` (str, optional): Author handle (e.g., `"@username"`, `"r/subreddit"`). Injected into prompt.
- `hashtags` (str or list, optional): Hashtags associated with the posts. Injected into prompt.
- `post_metadata` (dict, optional): Additional post metadata injected into prompt (e.g., `{"avg_likes": 1200}`).
- `description` (str): Additional context about the input data.
- `feed_question` (str, default=`""`): Context describing what to look for in the feed. Used when `categories="auto"`. When `sm_source` is set and this is omitted, defaults to `"What topics are discussed in these social media posts?"`.
- `user_model` (str, default=`"gpt-5"`): Model to use
- `mode` (str, default=`"image"`): PDF processing mode — `"image"`, `"text"`, or `"both"`
- `creativity` (float, optional): Temperature setting (0.0–1.0)
- `chain_of_thought` (bool, default=`False`): Enable step-by-step reasoning
- `step_back_prompt` (bool, default=`False`): Enable step-back prompting (results inconsistent — see Best Practices)
- `context_prompt` (bool, default=`False`): Add generic expert context to prompts (no consistent benefit observed)
- `filename` (str, optional): Output filename for CSV
- `save_directory` (str, optional): Directory to save results
- `model_source` (str, default=`"auto"`): Provider — `"auto"`, `"openai"`, `"anthropic"`, `"google"`, `"mistral"`, `"perplexity"`, `"huggingface"`, `"xai"`
- `models` (list, optional): For multi-model ensemble, list of `(model, provider, api_key)` tuples
- `consensus_threshold` (str or float, default=`"majority"`): Agreement threshold for ensemble mode
- `thinking_budget` (int, default=`0`): Token budget for model reasoning/thinking. Behavior varies by provider:

| Provider | `thinking_budget=0` | `thinking_budget > 0` |
|----------|---------------------|-----------------------|
| **OpenAI** | `reasoning_effort="minimal"` | `reasoning_effort="high"` |
| **Anthropic** | Thinking disabled | Extended thinking enabled (min 1024 tokens) |
| **Google** | Thinking disabled | `thinkingConfig: {thinkingBudget: N}` |

**Returns:**
- `pandas.DataFrame`: Classification results with category columns. When using `sm_source`, the DataFrame also includes:

| Column | Description |
|--------|-------------|
| `post_id` | Platform post ID |
| `timestamp` | Raw post datetime (UTC) |
| `media_type` | `TEXT`, `IMAGE`, `VIDEO`, `COMMENT`, or `CAROUSEL_ALBUM` |
| `image_url` | Image or video thumbnail URL |
| `likes` | Like count |
| `replies` | Reply / comment count |
| `reposts` | Repost count |
| `quotes` | Quote count |
| `views` | View count |
| `shares` | Share count |
| `day` | Day of week (e.g., `"Monday"`) |
| `month` | Month name (e.g., `"June"`) |
| `hour` | Hour of day, 24-hour scale (e.g., `23`) |
| `n_posts_that_day` | Total posts made on that calendar date |
| `duration_seconds` | *(YouTube video mode)* Video length in seconds |
| `tags` | *(YouTube video mode)* Creator-specified tags as a list |
| `video_id` | *(YouTube comments mode)* Parent video ID |
| `video_title` | *(YouTube comments mode)* Parent video title |
| `video_likes` | *(YouTube comments mode)* Parent video like count |
| `video_views` | *(YouTube comments mode)* Parent video view count |
| `video_comment_count` | *(YouTube comments mode)* Parent video total comments |
| `video_duration_seconds` | *(YouTube comments mode)* Parent video length in seconds |
| `video_tags` | *(YouTube comments mode)* Parent video tags as a list |

**Examples:**

```python
import catvader as cat

# Basic text classification
results = cat.classify(
    input_data=df['post_text'],
    categories=["Positive sentiment", "Negative sentiment", "Neutral"],
    description="Twitter posts about a product launch",
    api_key=api_key
)

# Pull directly from Threads — no input_data needed
results = cat.classify(
    sm_source="threads",
    sm_limit=100,
    categories=["Opinion/Commentary", "News/Information", "Humor/Satire", "Other"],
    platform="Threads",
    api_key=api_key
)

# With social media context injected into prompt
results = cat.classify(
    input_data=df['post_text'],
    categories=["Misinformation", "Opinion", "Factual", "Satire"],
    platform="Twitter/X",
    hashtags=["#Election2024", "#Politics"],
    post_metadata={"avg_likes": 450, "avg_shares": 120},
    api_key=api_key
)

# Auto-discover categories from a feed (categories="auto")
results = cat.classify(
    sm_source="threads",
    sm_limit=50,
    categories="auto",
    feed_question="What topics and themes appear in these posts?",
    api_key=api_key
)

# Pull from a Mastodon account (no credentials needed)
results = cat.classify(
    sm_source="mastodon",
    sm_handle="Gargron@mastodon.social",
    sm_limit=50,
    categories=["Tech & Open Source", "Politics", "Personal", "Other"],
    api_key=api_key
)

# YouTube — classify videos by title + description
results = cat.classify(
    sm_source="youtube",
    sm_handle="@H3Podcast",
    sm_limit=50,
    sm_credentials={"api_key": youtube_api_key},
    categories=["Commentary & Reaction", "Interviews", "Humor & Parody", "Other"],
    api_key=api_key
)

# YouTube — classify videos using full transcripts
results = cat.classify(
    sm_source="youtube",
    sm_handle="@H3Podcast",
    sm_limit=20,
    sm_credentials={"api_key": youtube_api_key},
    sm_youtube_transcript=True,
    sm_youtube_transcript_max_chars=50_000,
    categories=["Commentary & Reaction", "Interviews", "Humor & Parody", "Other"],
    api_key=api_key
)

# YouTube — classify at the comment level (video stats as covariates)
results = cat.classify(
    sm_source="youtube",
    sm_handle="@H3Podcast",
    sm_limit=10,                    # 10 videos
    sm_youtube_content="comments",
    sm_comments_per_video=50,
    sm_credentials={"api_key": youtube_api_key},
    categories=["Supportive", "Critical", "Humorous", "Off-topic"],
    api_key=api_key
)

# Image classification (auto-detected from file paths)
results = cat.classify(
    input_data="/path/to/images/",
    categories=["Contains person", "Outdoor scene", "Has text"],
    description="Instagram post images",
    api_key=api_key
)

# Multi-model ensemble for higher accuracy
results = cat.classify(
    input_data=df['post_text'],
    categories=["Hate speech", "Harassment", "Safe content"],
    models=[
        ("gpt-5", "openai", "sk-..."),
        ("claude-sonnet-4-5-20250929", "anthropic", "sk-ant-..."),
        ("gemini-2.5-flash", "google", "AIza..."),
    ],
    consensus_threshold="majority",
)
```

**Multi-Model Ensemble:**

When you provide the `models` parameter, CatVader runs classification across multiple models in parallel and combines results using majority voting. The output includes:
- Individual model predictions (e.g., `category_1_gpt_4o`, `category_1_claude`)
- Consensus columns (e.g., `category_1_consensus`)
- Agreement scores showing how many models agreed

---

### `extract()`

Unified category extraction function for text, image, and PDF inputs. Automatically discovers categories in your data when you don't have a predefined scheme.

**Parameters:**
- `input_data`: The data to explore (text list, image paths, or PDF paths)
- `api_key` (str): API key for the LLM service
- `input_type` (str, default=`"text"`): Type of input — `"text"`, `"image"`, or `"pdf"`
- `description` (str): Description of the input data
- `platform` (str, optional): Social media platform context
- `handle` (str, optional): Author handle context
- `hashtags` (str or list, optional): Hashtag context
- `post_metadata` (dict, optional): Additional metadata context
- `max_categories` (int, default=12): Maximum number of categories to return
- `categories_per_chunk` (int, default=10): Categories to extract per chunk
- `divisions` (int, default=12): Number of chunks to divide data into
- `iterations` (int, default=8): Number of extraction passes over the data
- `user_model` (str, default=`"gpt-5"`): Model to use
- `specificity` (str, default=`"broad"`): `"broad"` or `"specific"` category granularity
- `research_question` (str, optional): Research context to guide extraction
- `focus` (str, optional): Focus instruction (e.g., `"emotional tone"`, `"political stance"`)
- `filename` (str, optional): Output filename for CSV

**Returns:**
- `dict` with keys:
  - `counts_df`: DataFrame of categories with counts
  - `top_categories`: List of top category names
  - `raw_top_text`: Raw model output

**Example:**

```python
import catvader as cat

# Discover categories in Reddit posts
results = cat.extract(
    input_data=df['comment_text'],
    description="r/technology comments about AI",
    platform="Reddit",
    api_key=api_key,
    max_categories=10,
    focus="concerns and criticisms"
)

print(results['top_categories'])
# ['Privacy concerns', 'Job displacement', 'Bias in AI', ...]
```

---

### `explore()`

Raw category extraction for frequency and saturation analysis. Unlike `extract()`, which normalizes and merges categories, `explore()` returns **every category string from every chunk across every iteration** — with duplicates intact.

**Parameters:**
- `input_data`: List of text responses or pandas Series
- `api_key` (str): API key for the LLM service
- `description` (str): Description of the data
- `platform` (str, optional): Social media platform context
- `handle` (str, optional): Author handle context
- `hashtags` (str or list, optional): Hashtag context
- `post_metadata` (dict, optional): Additional metadata context
- `categories_per_chunk` (int, default=10): Categories to extract per chunk
- `divisions` (int, default=12): Number of chunks to divide data into
- `user_model` (str, default=`"gpt-5"`): Model to use
- `creativity` (float, optional): Temperature setting
- `specificity` (str, default=`"broad"`): `"broad"` or `"specific"`
- `research_question` (str, optional): Research context
- `focus` (str, optional): Focus instruction
- `iterations` (int, default=8): Number of passes over the data
- `random_state` (int, optional): Random seed for reproducibility
- `filename` (str, optional): Output CSV filename

**Returns:**
- `list[str]`: Every category extracted from every chunk across every iteration.

**Example:**

```python
import catvader as cat

# Run many iterations for saturation analysis
raw_categories = cat.explore(
    input_data=df['post_text'],
    description="TikTok comments on a viral video",
    platform="TikTok",
    api_key=api_key,
    iterations=20,
    divisions=5,
    categories_per_chunk=10,
)

from collections import Counter
counts = Counter(raw_categories)
for category, freq in counts.most_common(15):
    print(f"{freq:3d}x  {category}")
```

---

### `image_features()`

Extracts specific features and attributes from images, returning exact answers to user-defined questions.

**Parameters:**
- `image_description` (str): A description of what the model should expect to see
- `image_input` (list): List of image file paths or folder path
- `features_to_extract` (list): Features to extract (e.g., `["number of people", "primary color"]`)
- `api_key` (str): API key for the LLM service
- `user_model` (str, default=`"gpt-5"`): Specific vision model to use
- `creativity` (float, default=0): Temperature setting
- `filename` (str, default=`"categorized_data.csv"`): Filename for CSV output
- `save_directory` (str, optional): Directory path to save the CSV file
- `model_source` (str, default=`"OpenAI"`): Model provider

**Returns:**
- `pandas.DataFrame`: DataFrame with image paths and extracted feature values

**Example:**

```python
import catvader as cat

features = cat.image_features(
    image_description='Social media post screenshots',
    features_to_extract=['number of hashtags', 'contains image', 'estimated likes'],
    image_input='/path/to/screenshots/',
    user_model="gpt-5",
    api_key=api_key
)
```

---

## Deprecated Functions

The following functions are deprecated and will be removed in a future version. Please use `classify()` instead.

| Deprecated Function | Replacement |
|---------------------|-------------|
| `multi_class()` | `classify(input_data=texts, ...)` |
| `image_multi_class()` | `classify(input_data=images, ...)` |
| `pdf_multi_class()` | `classify(input_data=pdfs, ...)` |
| `explore_corpus()` | `extract(input_data=texts, ...)` |
| `explore_common_categories()` | `extract(input_data=texts, ...)` |

---

## Contributing & Support

Contributions are welcome!

- **Report bugs or request features**: [Open a GitHub Issue](https://github.com/chrissoria/cat-vader/issues)
- **Ask questions or get help**: [GitHub Discussions](https://github.com/chrissoria/cat-vader/discussions)
- **Research collaboration**: Email [ChrisSoria@Berkeley.edu](mailto:ChrisSoria@Berkeley.edu)

## License

`cat-vader` is distributed under the terms of the [GNU GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.en.html) license.
