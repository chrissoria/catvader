# SPDX-FileCopyrightText: 2025-present Christopher Soria <chrissoria@berkeley.edu>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from .__about__ import (
    __version__,
    __author__,
    __description__,
    __title__,
    __url__,
    __license__,
)

# =============================================================================
# Public API - Main entry points (cat-vader wrappers with social media support)
# =============================================================================
from .classify import classify
from .extract import extract
from .explore import explore

# =============================================================================
# Provider utilities — re-exported from cat_stack for convenience
# =============================================================================
from cat_stack import (
    UnifiedLLMClient,
    detect_provider,
    PROVIDER_CONFIG,
    set_ollama_endpoint,
    check_ollama_running,
    list_ollama_models,
    check_ollama_model,
    pull_ollama_model,
    # Category analysis
    has_other_category,
    check_category_verbosity,
    # Batch exceptions
    BatchJobExpiredError,
    BatchJobFailedError,
    # JSON / validation utilities
    build_json_schema,
    extract_json,
    validate_classification_json,
    # Image utilities
    image_score_drawing,
    image_features,
)

# Define public API
__all__ = [
    # Main entry points
    "extract",
    "explore",
    "classify",
    # Category analysis
    "has_other_category",
    "check_category_verbosity",
    # Provider utilities
    "UnifiedLLMClient",
    "detect_provider",
    "PROVIDER_CONFIG",
    "set_ollama_endpoint",
    "check_ollama_running",
    "list_ollama_models",
    "check_ollama_model",
    "pull_ollama_model",
    # Batch exceptions
    "BatchJobExpiredError",
    "BatchJobFailedError",
    # Utilities
    "build_json_schema",
    "extract_json",
    "validate_classification_json",
    "image_score_drawing",
    "image_features",
]
