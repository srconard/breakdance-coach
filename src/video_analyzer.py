"""Gemini API video analyzer for identifying tutorial steps."""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import google.generativeai as genai

from config import get_api_key


@dataclass
class TutorialStep:
    """A single step in the tutorial."""

    step_number: int
    start_time: str  # MM:SS format
    end_time: str  # MM:SS format
    label: str  # Short label for the step
    start_seconds: float = 0.0
    end_seconds: float = 0.0

    def __post_init__(self):
        """Convert time strings to seconds."""
        self.start_seconds = self._time_to_seconds(self.start_time)
        self.end_seconds = self._time_to_seconds(self.end_time)

    @staticmethod
    def _time_to_seconds(time_str: str) -> float:
        """Convert MM:SS or HH:MM:SS to seconds."""
        parts = time_str.split(':')
        if len(parts) == 2:
            minutes, seconds = map(float, parts)
            return minutes * 60 + seconds
        elif len(parts) == 3:
            hours, minutes, seconds = map(float, parts)
            return hours * 3600 + minutes * 60 + seconds
        else:
            raise ValueError(f"Invalid time format: {time_str}")


def analyze_video(video_path: Path, model_name: str = "gemini-2.5-flash") -> list[TutorialStep]:
    """Analyze a video to identify tutorial steps and timestamps.

    Args:
        video_path: Path to the video file
        model_name: Gemini model to use (default: gemini-2.5-flash)

    Returns:
        List of TutorialStep objects
    """
    # Configure Gemini
    api_key = get_api_key("google")
    genai.configure(api_key=api_key)

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    print(f"Uploading video to Gemini: {video_path.name}")

    # Upload the video file
    video_file = genai.upload_file(path=str(video_path))

    # Wait for processing
    print("Waiting for Gemini to process video", end="")
    while video_file.state.name == "PROCESSING":
        print(".", end="", flush=True)
        time.sleep(2)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise RuntimeError(f"Video processing failed: {video_file.state.name}")

    print(" Done!")

    # Create the model
    model = genai.GenerativeModel(model_name=model_name)

    # Prompt for step identification
    prompt = """You are analyzing a breakdancing tutorial video. Your task is to identify the distinct steps or moves being taught.

For each step, provide:
1. "start": The exact timestamp when the step begins (format: MM:SS)
2. "end": The exact timestamp when the step ends (format: MM:SS)
3. "label": A short, descriptive label for the step (2-5 words)

Guidelines:
- Focus on distinct teaching moments or moves
- Include setup positions and transitions if they are clearly explained
- Make sure timestamps don't overlap
- Labels should be clear and action-oriented (e.g., "Basic Toprock Step", "Drop to Six-Step", "Freeze Position")

Return ONLY a valid JSON array with no additional text or markdown formatting.

Example output:
[
  {"start": "00:15", "end": "00:45", "label": "Basic Stance Setup"},
  {"start": "00:45", "end": "01:30", "label": "First Toprock Move"},
  {"start": "01:30", "end": "02:15", "label": "Transition to Floor"}
]"""

    print("Analyzing video for tutorial steps...")

    # Generate content with the video
    response = model.generate_content([video_file, prompt])

    # Parse the response
    response_text = response.text.strip()

    # Clean up any markdown formatting
    if response_text.startswith("```"):
        # Remove markdown code block
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1])

    try:
        steps_data = json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"Failed to parse Gemini response: {response_text}")
        raise ValueError(f"Invalid JSON response from Gemini: {e}")

    # Convert to TutorialStep objects
    steps = []
    for i, step_data in enumerate(steps_data, start=1):
        step = TutorialStep(
            step_number=i,
            start_time=step_data["start"],
            end_time=step_data["end"],
            label=step_data["label"],
        )
        steps.append(step)

    print(f"Found {len(steps)} tutorial steps")

    # Clean up uploaded file
    try:
        genai.delete_file(video_file.name)
    except Exception:
        pass  # Ignore cleanup errors

    return steps


def print_steps(steps: list[TutorialStep]) -> None:
    """Pretty print the tutorial steps."""
    print("\n" + "=" * 60)
    print("TUTORIAL STEPS")
    print("=" * 60)

    for step in steps:
        print(f"\nStep {step.step_number}: {step.label}")
        print(f"  Time: {step.start_time} - {step.end_time}")
        print(f"  Duration: {step.end_seconds - step.start_seconds:.1f}s")


if __name__ == "__main__":
    # Test the analyzer
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.video_analyzer <video_path>")
        sys.exit(1)

    video = Path(sys.argv[1])
    steps = analyze_video(video)
    print_steps(steps)
