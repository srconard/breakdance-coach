"""Multi-provider LLM description generator for tutorial steps."""

import time
from abc import ABC, abstractmethod
from typing import Literal

import anthropic
import google.generativeai as genai
import openai

from config import get_api_key
from src.video_analyzer import TutorialStep


class DescriptionProvider(ABC):
    """Abstract base class for description providers."""

    @abstractmethod
    def generate_description(self, step: TutorialStep, video_title: str) -> str:
        """Generate a polished description for a tutorial step.

        Args:
            step: The tutorial step to describe
            video_title: Title of the video for context

        Returns:
            A polished, instructional description
        """
        pass

    def generate_descriptions(
        self, steps: list[TutorialStep], video_title: str, rate_limit_delay: float = 15.0
    ) -> list[str]:
        """Generate descriptions for all steps.

        Args:
            steps: List of tutorial steps
            video_title: Title of the video for context
            rate_limit_delay: Seconds to wait between API calls (default 15s for free tier)

        Returns:
            List of descriptions in the same order as steps
        """
        descriptions = []
        for i, step in enumerate(steps, start=1):
            print(f"Generating description {i}/{len(steps)}: {step.label}")
            desc = self.generate_description(step, video_title)
            descriptions.append(desc)
            # Rate limit: wait between calls to avoid quota errors
            if i < len(steps):
                print(f"  Waiting {rate_limit_delay}s for rate limit...")
                time.sleep(rate_limit_delay)
        return descriptions


class GoogleDescriptionProvider(DescriptionProvider):
    """Google Gemini description provider."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        api_key = get_api_key("google")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name=model_name)

    def generate_description(self, step: TutorialStep, video_title: str) -> str:
        prompt = self._build_prompt(step, video_title)
        response = self.model.generate_content(prompt)
        return response.text.strip()

    def _build_prompt(self, step: TutorialStep, video_title: str) -> str:
        return f"""You are writing instructional descriptions for a breakdancing tutorial called "{video_title}".

Write a clear, engaging description for this tutorial step:
- Step: {step.label}
- Duration: {step.end_seconds - step.start_seconds:.1f} seconds

Guidelines:
- Write 2-3 sentences that explain what the learner should do
- Include specific body positioning cues where relevant
- Use encouraging, instructional language
- Don't include timestamps in the description
- Keep it concise and actionable

Write only the description, no additional formatting or labels."""


class AnthropicDescriptionProvider(DescriptionProvider):
    """Anthropic Claude description provider."""

    def __init__(self, model_name: str = "claude-3-haiku-20240307"):
        api_key = get_api_key("anthropic")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name

    def generate_description(self, step: TutorialStep, video_title: str) -> str:
        prompt = self._build_prompt(step, video_title)
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    def _build_prompt(self, step: TutorialStep, video_title: str) -> str:
        return f"""You are writing instructional descriptions for a breakdancing tutorial called "{video_title}".

Write a clear, engaging description for this tutorial step:
- Step: {step.label}
- Duration: {step.end_seconds - step.start_seconds:.1f} seconds

Guidelines:
- Write 2-3 sentences that explain what the learner should do
- Include specific body positioning cues where relevant
- Use encouraging, instructional language
- Don't include timestamps in the description
- Keep it concise and actionable

Write only the description, no additional formatting or labels."""


class OpenAIDescriptionProvider(DescriptionProvider):
    """OpenAI GPT description provider."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        api_key = get_api_key("openai")
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate_description(self, step: TutorialStep, video_title: str) -> str:
        prompt = self._build_prompt(step, video_title)
        response = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    def _build_prompt(self, step: TutorialStep, video_title: str) -> str:
        return f"""You are writing instructional descriptions for a breakdancing tutorial called "{video_title}".

Write a clear, engaging description for this tutorial step:
- Step: {step.label}
- Duration: {step.end_seconds - step.start_seconds:.1f} seconds

Guidelines:
- Write 2-3 sentences that explain what the learner should do
- Include specific body positioning cues where relevant
- Use encouraging, instructional language
- Don't include timestamps in the description
- Keep it concise and actionable

Write only the description, no additional formatting or labels."""


def get_description_provider(
    provider: Literal["google", "anthropic", "openai"]
) -> DescriptionProvider:
    """Get a description provider by name.

    Args:
        provider: Provider name ('google', 'anthropic', or 'openai')

    Returns:
        DescriptionProvider instance
    """
    providers = {
        "google": GoogleDescriptionProvider,
        "anthropic": AnthropicDescriptionProvider,
        "openai": OpenAIDescriptionProvider,
    }

    if provider not in providers:
        raise ValueError(
            f"Unknown provider: {provider}. Choose from: {list(providers.keys())}"
        )

    return providers[provider]()


if __name__ == "__main__":
    # Test the providers
    from src.video_analyzer import TutorialStep

    test_step = TutorialStep(
        step_number=1,
        start_time="00:15",
        end_time="00:45",
        label="Basic Toprock Step",
    )

    print("Testing Google provider...")
    try:
        provider = get_description_provider("google")
        desc = provider.generate_description(test_step, "Beginner Toprock Tutorial")
        print(f"Google: {desc}\n")
    except Exception as e:
        print(f"Google error: {e}\n")

    print("Testing Anthropic provider...")
    try:
        provider = get_description_provider("anthropic")
        desc = provider.generate_description(test_step, "Beginner Toprock Tutorial")
        print(f"Anthropic: {desc}\n")
    except Exception as e:
        print(f"Anthropic error: {e}\n")

    print("Testing OpenAI provider...")
    try:
        provider = get_description_provider("openai")
        desc = provider.generate_description(test_step, "Beginner Toprock Tutorial")
        print(f"OpenAI: {desc}\n")
    except Exception as e:
        print(f"OpenAI error: {e}\n")
