import os
from dotenv import load_dotenv
import argparse

load_dotenv()  # Add this at the very top of the file

import json
from pathlib import Path
from loguru import logger
from generator.llm_processor import LLMProcessor
from generator.subtitle import SubtitleGenerator
from generator.video_processor import VideoProcessor


def process_video(
    video_path: Path,
    output_path: Path,
    model_size: str = "base",
    llm_model: str = "phi4",
    max_duration: float = 30.0,
) -> None:
    audio_path = None
    video_processor = VideoProcessor()
    logger.info(f"Processing video: {video_path}")

    # Extract audio and generate transcription
    audio_path = video_processor.extract_audio(video_path)
    sentences = SubtitleGenerator(model_size).process(audio_path)
    # save sentences to jsonn
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Video Processor for creating highlights with subtitles"
    )
    parser.add_argument("input", type=str, help="Path to input video file")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to output video file (default: ./outputs/highlights.mp4)",
        default="outputs/highlights.mp4",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=[
            "large-v2",
            "large-v1",
            "medium",
            "medium.en",
            "small",
            "small.en",
            "base",
            "base.en",
            "tiny",
            "tiny.en",
        ],
        default="base",
        help="Model size for whisperx subtitle generation available models: https://huggingface.co/collections/guillaumekln/faster-whisper-64f9c349b3115b4f51434976",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="phi4",
        help=" Ollama. LLM model for processing (default: phi4) Must be installed in ollama",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=60.0,
        help="Maximum duration of highlights in seconds (default: 60.0)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    input_video = Path(args.input)
    output_video = Path(args.output)
    output_video.parent.mkdir(exist_ok=True)

    if not input_video.exists():
        logger.error(f"Input video not found: {input_video}")
        exit(1)

    process_video(
        video_path=input_video,
        output_path=output_video,
        model_size=args.model_size,
        llm_model=args.llm_model,
        max_duration=args.max_duration,
    )
