from pathlib import Path
import typer
from loguru import logger

app = typer.Typer()


@app.command()
def process_video(
    video_path: Path = typer.Argument(..., help="Path to input video file"),
    output_path: Path = typer.Option(
        "outputs/highlights.mp4", "-o", "--output", help="Path to output video file"
    ),
    llm_model: str = typer.Option(
        None, "-m", "--llm-model", help="Ollama LLM model for processing"
    ),
    model_size: str = typer.Option(
        "base", "-s", "--model-size", help="Model size for subtitle generation"
    ),
    max_duration: float = typer.Option(
        60.0, "--max-duration", help="Maximum duration of highlights in seconds"
    ),
):
    """AI Video Processor for creating highlights with subtitles."""
    # Only import heavy modules when the command is actually run
    from dotenv import load_dotenv
    from generator.llm_processor import LLMProcessor
    from generator.subtitle import SubtitleGenerator
    from generator.video_processor import VideoProcessor

    load_dotenv()

    if not video_path.exists():
        logger.error(f"Input video not found: {video_path}")
        raise typer.Exit(1)

    output_path.parent.mkdir(exist_ok=True)

    # Initialize processors only when needed
    video_processor = VideoProcessor()
    logger.info(f"Processing video: {video_path}")

    # Extract audio and generate transcription
    audio_path = video_processor.extract_audio(video_path)

    logger.info("Initializing subtitle generator...")
    subtitle_gen = SubtitleGenerator(model_size)
    sentences = subtitle_gen.process(audio_path)

    logger.info("Initializing LLM processor...")
    llm_proc = LLMProcessor(llm_model)
    moments = llm_proc.process(sentences, max_duration)

    video_processor.create_highlights(video_path, moments, sentences, output_path)

    logger.info("\nHighlights created:")
    for m in moments:
        logger.info(f"{m.start_time:.2f}-{m.end_time:.2f} ({m.duration}s)")

    if audio_path and audio_path.exists():
        audio_path.unlink()


def main():
    app()


if __name__ == "__main__":
    main()
