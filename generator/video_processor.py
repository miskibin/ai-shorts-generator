from pathlib import Path
import ffmpeg
import cv2
import numpy as np
from loguru import logger
import tempfile
from typing import List, Optional
from generator.face_detector import FaceTracker
from generator.llm_processor import TimeSegment
from generator.subtitle import Sentence
from generator.utils import (
    AspectRatio,
    draw_multiline_text,
)
import importlib.resources


class VideoProcessor:
    def __init__(self):
        # Get the font path using importlib.resources
        try:
            with importlib.resources.path(
                "generator.assets.fonts", "Roboto-Bold.ttf"
            ) as font_path:
                self.font_path = Path(font_path)
        except Exception as e:
            logger.error(f"Failed to locate font: {e}")
            raise FileNotFoundError(f"Font not found in package resources")

        if not self.font_path.exists():
            raise FileNotFoundError(f"Font not found: {self.font_path}")

        self.face_tracker = FaceTracker(smoothing_window=30)

    def extract_audio(self, video_path: Path) -> Path:
        """Extract audio using direct ffmpeg"""
        temp_audio = Path(tempfile.gettempdir()) / "temp_audio.wav"
        logger.info(f"Extracting audio to {temp_audio}")
        # length in seconds

        (
            ffmpeg.input(str(video_path))
            .output(
                str(temp_audio),
                acodec="pcm_s16le",
                ar="16k",
            )
            .overwrite_output()
            .run(quiet=True)
        )

        return temp_audio

    def create_highlights(
        self,
        video_path: Path,
        segments: List[TimeSegment],
        sentences: List[Sentence],
        output_path: Path,
    ):
        logger.info(f"Creating highlights from {len(segments)} segments")
        temp_segments = []
        concat_file = Path(tempfile.gettempdir()) / "concat.txt"

        for i, seg in enumerate(segments):
            subtitled = Path(tempfile.gettempdir()) / f"segment_{i}.mp4"
            temp_segments.append(subtitled)

            logger.debug(
                f"Processing segment {i+1}/{len(segments)}: {seg.start_time:.2f}-{seg.end_time:.2f}"
            )
            self._create_segment_with_subtitles(
                video_path, subtitled, sentences, seg.start_time, seg.duration
            )

        # Create concat file
        concat_file.write_text(
            "\n".join(f"file '{f.absolute()}'" for f in temp_segments), encoding="utf-8"
        )

        # Create final video
        (
            ffmpeg.input(str(concat_file), f="concat", safe=0)
            .output(str(output_path), c="copy")
            .overwrite_output()
            .run(quiet=True)
        )

        logger.success(f"Created highlights video: {output_path}")

        # Cleanup
        for f in temp_segments:
            f.unlink(missing_ok=True)
        concat_file.unlink(missing_ok=True)

    def _create_segment_with_subtitles(
        self,
        video_path: Path,
        output_path: Path,
        sentences: List[Sentence],
        start_time: float,
        duration: float,
    ):
        """Create video segment with subtitles in one pass"""
        # Get video info for dimensions
        probe = ffmpeg.probe(str(video_path))
        video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
        width = int(video_info["width"])
        height = int(video_info["height"])

        # Extract segment
        temp_frames = output_path.with_suffix(".frames.mp4")

        (
            ffmpeg.input(str(video_path), ss=start_time, t=duration)
            .output(str(temp_frames), vf=f"scale={width}:{height}", acodec="aac")
            .overwrite_output()
            .run(quiet=True)
        )

        # Process frames with face tracking
        cap = cv2.VideoCapture(str(temp_frames))
        fps = cap.get(cv2.CAP_PROP_FPS)

        temp_subtitled = output_path.with_suffix(".subtitled.mp4")
        out = cv2.VideoWriter(
            str(temp_subtitled),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (AspectRatio.STORY.value[0], AspectRatio.STORY.value[1]),
        )

        # Get words for this segment with exact timing
        words = []
        for sentence in sentences:
            if sentence.start <= start_time + duration and sentence.end >= start_time:
                for word in sentence.words:
                    word_start = float(word["start"])
                    word_end = float(word["end"])
                    if word_start >= start_time and word_end <= start_time + duration:
                        words.append(word)

        # Sort words by start time
        words.sort(key=lambda w: float(w["start"]))

        frame_idx = 0
        first_frame = True
        force_center = False  # Track if we should use center framing

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Initialize or update tracking
            if first_frame:
                box = self.face_tracker.detect_face(frame)
                if box is None:
                    # Multiple faces detected, use center framing
                    force_center = True
                    logger.info("Multiple faces detected, using center framing")
                else:
                    self.face_tracker.update_tracking(frame)
                first_frame = False

            # Get face center if available
            face_center_x = self.face_tracker.update_tracking(frame)

            # Get crop box (will be centered if no face detected)
            x1, y1, x2, y2 = self.face_tracker.get_crop_box(
                frame.shape[1],
                frame.shape[0],
                AspectRatio.STORY.value[0],
                AspectRatio.STORY.value[1],
                face_center_x,
            )

            # Crop and resize
            cropped = frame[y1:y2, x1:x2]
            resized = cv2.resize(cropped, AspectRatio.STORY.value)

            # Add current word
            current_time = start_time + (frame_idx / fps)
            current_word = self._get_current_word(words, current_time)
            if current_word:
                resized = draw_multiline_text(
                    resized,
                    current_word,
                    pos=(
                        AspectRatio.STORY.value[0] // 2,
                        int(AspectRatio.STORY.value[1] * 0.8),
                    ),
                    font_path=self.font_path,
                    font_size=80,  # Increased font size
                    color=(255, 255, 255),
                    stroke_width=3,
                )

            out.write(resized)
            frame_idx += 1

        cap.release()
        out.release()

        # Combine subtitled video with original audio
        video = ffmpeg.input(str(temp_subtitled))
        audio = ffmpeg.input(str(temp_frames))
        (
            ffmpeg.output(
                video,
                audio,
                str(output_path),
                vcodec="libx264",
                acodec="aac",
                map="0:v:0",
            )
            .overwrite_output()
            .run(quiet=True)
        )

        # Cleanup temporary files
        temp_frames.unlink(missing_ok=True)
        temp_subtitled.unlink(missing_ok=True)

    def _get_current_word(self, words: List[dict], time: float) -> Optional[str]:
        """Get word visible at current time"""
        for word in words:
            if float(word["start"]) - 0.05 <= time <= float(word["end"]) + 0.05:
                return word["word"]
        return None
