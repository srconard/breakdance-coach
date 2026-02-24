"""Configuration settings and API key management."""

import os
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Settings:
    """Configuration settings for the tutorial generator."""

    # Video preprocessing
    downscale: bool = True
    downscale_height: int = 480
    reduce_fps: bool = True
    reduced_fps: int = 15
    trim_intro: int = 0  # Seconds to trim from start
    trim_outro: int = 0  # Seconds to trim from end

    # GIF output settings
    gif_fps: int = 10
    gif_width: int = 480

    # LLM provider for descriptions
    description_model: Literal["google", "anthropic", "openai"] = "google"

    # Output settings
    output_dir: str = "output"


def get_api_key(provider: str) -> str:
    """Get API key for the specified provider from environment variables.

    Args:
        provider: One of 'google', 'anthropic', or 'openai'

    Returns:
        The API key string

    Raises:
        ValueError: If the API key is not set
    """
    key_names = {
        "google": "GOOGLE_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
    }

    if provider not in key_names:
        raise ValueError(f"Unknown provider: {provider}")

    key_name = key_names[provider]
    api_key = os.environ.get(key_name)

    if not api_key:
        raise ValueError(
            f"{key_name} environment variable not set. "
            f"Please set it with your {provider} API key."
        )

    return api_key


# Default settings instance
DEFAULT_SETTINGS = Settings()
