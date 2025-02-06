import os
from dotenv import load_dotenv
import json
from pathlib import Path
from loguru import logger
import typer
from generator.llm_processor import LLMProcessor
from generator.subtitle import SubtitleGenerator
from generator.video_processor import VideoProcessor

load_dotenv()  # Load environment variables

app = typer.Typer()


@app.command()
def process_video(
    video_path: Path = typer.Argument(..., help="Path to input video file"),
    output_path: Path = typer.Option(
        "outputs/highlights.mp4", "-o", "--output", help="Path to output video file"
    ),
    llm_model: str = typer.Option(
        None, "--llm-model", help="Ollama LLM model for processing"
    ),
    model_size: str = typer.Option(
        "base", "--model-size", help="Model size for subtitle generation"
    ),
    max_duration: float = typer.Option(
        60.0, "--max-duration", help="Maximum duration of highlights in seconds"
    ),
):
    """AI Video Processor for creating highlights with subtitles."""
    logger.debug("Initializing app")
    video_processor = VideoProcessor()
    logger.info(f"Processing video: {video_path}")

    # Extract audio and generate transcription
    audio_path = video_processor.extract_audio(video_path)
    sentences = SubtitleGenerator(model_size).process(audio_path)

    # Save sentences to JSON
    json_sentences = [s.model_dump_json() for s in sentences]
    with open("sentences.json", "w") as f:
        json.dump(json_sentences, f, indent=2)

    # Find interesting moments
    moments = LLMProcessor(llm_model).process(sentences, max_duration)

    # Create highlights with subtitles
    video_processor.create_highlights(video_path, moments, sentences, output_path)

    logger.info("\nHighlights created:")
    for m in moments:
        logger.info(f"{m.start_time:.2f}-{m.end_time:.2f} ({m.duration}s)")

    if audio_path.exists():
        audio_path.unlink()


if __name__ == "__main__":
    app()
