# cat-vader

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
  - [image_score_drawing()](#image_score_drawing)
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
- `input_data`: The data to classify (text list/Series, image paths, or PDF paths)
- `categories` (list): List of category names for classification
- `api_key` (str): API key for the LLM service (single-model mode)
- `description` (str): Description of the input data context
- `platform` (str, optional): Social media platform (e.g., `"Twitter/X"`, `"Reddit"`, `"Instagram"`, `"TikTok"`). Injected into the classification prompt.
- `handle` (str, optional): Author handle (e.g., `"@username"`, `"r/subreddit"`).
- `hashtags` (str or list, optional): Hashtags associated with the posts.
- `post_metadata` (dict, optional): Additional post metadata (e.g., `{"likes": 1200, "shares": 450}`).
- `user_model` (str, default=`"gpt-4o"`): Model to use
- `mode` (str, default=`"image"`): PDF processing mode — `"image"`, `"text"`, or `"both"`
- `creativity` (float, optional): Temperature setting (0.0–1.0)
- `chain_of_thought` (bool, default=`False`): Enable step-by-step reasoning
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
- `pandas.DataFrame`: Classification results with category columns

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

# With social media context injected into prompt
results = cat.classify(
    input_data=df['post_text'],
    categories=["Misinformation", "Opinion", "Factual", "Satire"],
    description="Posts about the 2024 election",
    platform="Twitter/X",
    hashtags=["#Election2024", "#Politics"],
    post_metadata={"avg_likes": 450, "avg_shares": 120},
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
        ("gpt-4o", "openai", "sk-..."),
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
- `user_model` (str, default=`"gpt-4o"`): Model to use
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
- `user_model` (str, default=`"gpt-4o"`): Model to use
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

### `image_score_drawing()`

Performs quality scoring of images against a reference description, returning structured results with optional CSV export.

**Parameters:**
- `reference_image_description` (str): A description of what the model should expect to see
- `image_input` (list): List of image file paths or folder path containing images
- `reference_image` (str): A file path to the reference image
- `api_key` (str): API key for the LLM service
- `user_model` (str, default=`"gpt-4o"`): Specific vision model to use
- `creativity` (float, default=0): Temperature setting
- `safety` (bool, default=False): Enable safety checks
- `filename` (str, default=`"image_scores.csv"`): Filename for CSV output
- `save_directory` (str, optional): Directory path to save the CSV file
- `model_source` (str, default=`"OpenAI"`): Model provider

**Returns:**
- `pandas.DataFrame`: DataFrame with image paths, quality scores, and analysis details

**Example:**

```python
import catvader as cat

image_scores = cat.image_score_drawing(
    reference_image_description='A hand-drawn circle',
    image_input=['image1.jpg', 'image2.jpg'],
    user_model="gpt-4o",
    api_key=api_key
)
```

---

### `image_features()`

Extracts specific features and attributes from images, returning exact answers to user-defined questions.

**Parameters:**
- `image_description` (str): A description of what the model should expect to see
- `image_input` (list): List of image file paths or folder path
- `features_to_extract` (list): Features to extract (e.g., `["number of people", "primary color"]`)
- `api_key` (str): API key for the LLM service
- `user_model` (str, default=`"gpt-4o"`): Specific vision model to use
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
    user_model="gpt-4o",
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
