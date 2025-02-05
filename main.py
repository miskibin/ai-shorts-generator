from pathlib import Path
from loguru import logger
from generator.llm_processor import LLMProcessor
from generator.subtitle import SubtitleGenerator
from generator.video_processor import VideoProcessor


def process_video(
    video_path: Path,
    output_path: Path,
    model_size: str = "base",
    llm_model: str = "gemma2",
    max_duration: float = 30.0,
) -> None:
    audio_path = None
    video_processor = VideoProcessor()
    logger.info(f"Processing video: {video_path}")

    # Extract audio and generate transcription
    audio_path = video_processor.extract_audio(video_path)
    sentences = SubtitleGenerator(model_size).process(audio_path)

    # Find interesting moments
    moments = LLMProcessor(llm_model).process(sentences, max_duration)

    # Create highlights with subtitles
    video_processor.create_highlights(video_path, moments, sentences, output_path)

    logger.info("\nHighlights created:")
    for m in moments:
        logger.info(f"{m.start_time:.2f}-{m.end_time:.2f} ({m.duration}s): {m.reason}")

    if audio_path.exists():
        audio_path.unlink()


if __name__ == "__main__":
    input_video = Path(__file__).parent / "inputs" / "test.mkv"
    output_video = Path(__file__).parent / "outputs" / "highlights.mp4"
    output_video.parent.mkdir(exist_ok=True)

    process_video(
        video_path=input_video,
        output_path=output_video,
        model_size="base",
        llm_model="gemma2",
        max_duration=40.0,
    )
