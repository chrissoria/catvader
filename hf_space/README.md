---
title: CatVader - Social Media Classifier
emoji: 🐱
colorFrom: yellow
colorTo: yellow
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
license: mit
short_description: Classify social media posts using LLMs
---

# CatVader - Social Media Classifier

A web interface for the [cat-vader](https://github.com/chrissoria/cat-vader) Python package. Classify social media posts into custom categories using various LLM providers.

## How to Use

1. **Upload Your Data**: Upload a CSV or Excel file containing social media posts
2. **Select Column**: Choose the column containing the text to classify
3. **Define Categories**: Enter your classification categories (e.g., "Positive", "Negative", "Neutral")
4. **Choose a Model**: Select your preferred LLM (free models available!)
5. **Click Classify**: View and download results with category assignments

## Supported Models

| Provider | Models |
|----------|--------|
| **OpenAI** | gpt-4o, gpt-4o-mini |
| **Anthropic** | Claude Sonnet, Claude Haiku |
| **Google** | Gemini 2.5 Flash, Gemini 2.5 Pro |
| **Mistral** | Mistral Large, Mistral Small |
| **xAI** | Grok models |
| **HuggingFace** | Qwen, Llama 4, DeepSeek, and thousands more |

## Features

- **Multi-label classification**: Assign multiple categories per post
- **Social media context**: Inject platform, handle, and hashtag context into prompts
- **Image classification**: Classify posts with images or image URL columns
- **Multi-model ensemble**: Run multiple models and use majority voting
- **Batch processing**: Handle thousands of posts efficiently
- **CSV/Excel export**: Download results ready for analysis
