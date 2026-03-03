# Changelog

All notable changes to **cat-vader** are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

> cat-vader was forked from [cat-llm](https://github.com/chrissoria/cat-llm) at v2.5.0 and re-scoped for social media data analysis. See the cat-llm repo for pre-fork history.

---

## [Unreleased]

---

## [1.11.0] - 2026-03-02

### Added
- **Mastodon support** via `sm_source="mastodon"`.
  - No credentials required — uses the public REST API.
  - `sm_handle` accepts `"user@instance.social"`, `"@user@instance.social"`, or a full profile URL.
  - HTML tags stripped and entities decoded from post content.
- **YouTube support** via `sm_source="youtube"`.
  - Requires a YouTube Data API v3 key: `sm_credentials={"api_key": "..."}` or `YOUTUBE_KEY` / `YOUTUBE_API_KEY` env var.
  - `sm_handle` accepts `"@ChannelHandle"`, a channel ID (`"UCxxxxxx"`), or a full channel URL.
  - **Video mode** (`sm_youtube_content="video"`, default): one row per video; `text` = title + description.
  - **Transcript mode** (`sm_youtube_transcript=True`): replaces description with the auto-generated transcript (requires `pip install youtube-transcript-api`); falls back to description if unavailable.
  - **Comments mode** (`sm_youtube_content="comments"`): one row per comment; video-level stats travel as covariate columns (`video_id`, `video_title`, `video_likes`, `video_views`, `video_comment_count`, `video_duration_seconds`, `video_tags`).
  - `duration_seconds` and `tags` columns added to video-mode output.
  - HTML entities and tags decoded from YouTube comment text.
- New `classify()` parameters for YouTube:
  - `sm_youtube_content` (`"video"` | `"comments"`) — unit of analysis.
  - `sm_youtube_transcript` (bool) — use transcript as text in video mode.
  - `sm_youtube_transcript_max_chars` (int, default `10_000`) — max transcript characters; `None` = full transcript.
  - `sm_comments_per_video` (int, default `20`) — max comments per video in comments mode.
- `sm_handle` parameter on `classify()` and `fetch_social_media()` — top-level handle for platforms that require one (Mastodon, YouTube, Bluesky).

---

## [1.10.0] - 2026-03-04

### Added
- `sm_days` parameter on `classify()` — fetch posts from the last N days (e.g. `sm_days=1` for today's posts). Overrides `sm_months`. Currently Reddit only.
- Reddit: `upvote_ratio` column (raw from API), `upvotes_raw` and `downvotes_est` columns (estimated from score × ratio — Reddit fuzzes these slightly).
- Reddit: built-in rate limiting — 6s between paginated requests without OAuth (10 req/min), 1s with OAuth (60 req/min).
- Status message now shows fetch window (e.g. `last 1d`, `last 3mo`, `limit=50`) instead of just limit count.

---

## [1.9.0] - 2026-03-02

### Added
- Reddit support via `sm_source="reddit"`.
  - **Public subreddit** (no credentials required): `sm_credentials={"subreddit": "MachineLearning"}`
  - **Public user profile** (no credentials required): `sm_credentials={"username": "chrissoria"}`
  - **OAuth** (60 req/min vs 10 req/min): add `client_id` and `client_secret` from reddit.com/prefs/apps — no user password needed for public profiles
  - Falls back to env vars: `REDDIT_SUBREDDIT`, `REDDIT_USERNAME`, `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`
  - `text` = post title + body (selftext); link posts use title only
  - `likes` = net score, `replies` = num_comments, `reposts` = num_crossposts
  - Crossposts flagged as `REPOST_FACADE` — consistent with Bluesky so `is_repost=1` works automatically
  - `quotes`, `views`, `shares` = 0 (not exposed by Reddit's API)
  - Hard cap: Reddit limits any listing to 1,000 posts regardless of auth

---

## [1.8.0] - 2026-02-28

### Added
- Bluesky support via `sm_source="bluesky"`.
  - Credentials: `sm_credentials={"handle": "...", "app_password": "..."}` or env vars `BLUESKY_HANDLE` / `BLUESKY_APP_PASSWORD`
  - Fetches authenticated user's own posts via AT Protocol
  - `views` and `shares` = 0 (not exposed by Bluesky's API)

---

## [1.7.1] - 2026-02-26

### Fixed
- Logo not displaying on PyPI — switched README `<img>` tag to absolute GitHub raw URL.

---

## [1.7.0] - 2026-02-26

### Added
- `contains_image` binary column: 1 if the post has an attached image or video thumbnail.
- `contains_url` binary column: 1 if the post text contains a URL (`https?://`).

---

## [1.6.0] - 2026-02-26

### Added
- `post_length` column: character count of the post text.

---

## [1.5.0] - 2026-02-26

### Added
- `is_repost` binary column: 1 if the post is a repost/retweet/crosspost (detected from `media_type` containing `REPOST`).

---

## [1.4.0] - 2026-02-26

### Added
- `n_posts_that_day` column: number of posts in the fetched batch made on the same calendar date. Useful for normalizing engagement by posting volume.

---

## [1.3.0] - 2026-02-26

### Added
- `sm_timezone` parameter on `classify()`: converts timestamps before deriving day/month/hour columns. Accepts any pytz/IANA string (e.g. `"America/Los_Angeles"`). Default `"UTC"`.
- `hour` column: hour of post (0–23) after timezone conversion.

---

## [1.2.0] - 2026-02-26

### Added
- `day` column: day of week the post was made (e.g. `"Monday"`).
- `month` column: month name the post was made (e.g. `"March"`).

---

## [1.1.0] - 2026-02-25

### Changed
- Default model updated to `gpt-5` across all parameter defaults, docstrings, and examples.

---

## [1.0.0] - 2026-02-25

### Added (cat-vader launch)
- `sm_source` parameter on `classify()`: auto-fetches posts from a social media platform and appends engagement metrics to the output DataFrame. Initial support: Threads.
- `sm_limit`, `sm_months`, `sm_credentials` parameters for controlling feed fetches.
- Engagement columns appended to output when using `sm_source`: `post_id`, `timestamp`, `media_type`, `image_url`, `likes`, `replies`, `reposts`, `quotes`, `views`, `shares`.
- `platform`, `handle`, `hashtags`, `post_metadata` parameters: inject social media context into the LLM classification prompt.
- CatVader logo added to repo, README, and PyPI page.

### Changed
- Output column renamed from `survey_input` → `social_media_input`.
- Package renamed from `cat-llm` / `catllm` to `cat-vader` / `catvader`.

### Removed
- `summarize()` function (survey-specific, not applicable to social media use case).
- CERAD medical drawing scoring (`image_score_drawing()`).
