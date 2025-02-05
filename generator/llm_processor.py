from dataclasses import dataclass
import ollama
import json
from pathlib import Path
from loguru import logger

from generator.subtitle import Sentence


@dataclass
class TimeSegment:
    start_time: float
    end_time: float
    reason: str
    duration: float = 0.0

    def __post_init__(self):
        self.duration = round(self.end_time - self.start_time, 2)


class LLMProcessor:
    PROMPT_TEMPLATE = """Analyze this political speech and select up to 3 most interesting fragments (total max {max_duration}s).
Each fragment can be a single sentence or multiple consecutive sentences.

Speech segments:
{segments}

Return JSON in this exact format:
{{
    "selections": [
        {{
            "start_time": float,
            "end_time": float,
            "reason": "why this part is interesting"
        }}
        // up to 3 selections allowed
    ]
}}

Rules:
1. Pick up to 3 most interesting parts
2. Each part must be continuous (no gaps within a selection)
3. Total duration of all selections must not exceed {max_duration} seconds
4. Fragments don't need to be consecutive (can be from different parts of speech)

Consider: announcements, emotional statements, key messages, controversies."""

    def __init__(self, model_name: str = "gemma"):
        self.model = model_name
        logger.info(f"Initialized LLM processor with {model_name}")

    def process(
        self, sentences: list[Sentence], max_duration: float = 30.0
    ) -> list[TimeSegment]:
        try:
            segments_text = "\n".join(
                f"[{i}] [{s.start:.2f}-{s.end:.2f}] {s.text}"
                for i, s in enumerate(sentences)
            )

            prompt = self.PROMPT_TEMPLATE.format(
                segments=segments_text, max_duration=max_duration
            )
            logger.debug(f"Sending prompt to Ollama:\n{prompt}")

            response = ollama.chat(
                model=self.model, messages=[{"role": "user", "content": prompt}]
            )
            content = self._extract_json(response["message"]["content"].strip())
            result = json.loads(content)

            # Convert all selections to TimeSegments
            segments = [
                TimeSegment(
                    start_time=float(sel["start_time"]),
                    end_time=float(sel["end_time"]),
                    reason=str(sel["reason"]),
                )
                for sel in result["selections"]
            ]

            total_duration = sum(seg.duration for seg in segments)
            logger.info(
                f"Selected {len(segments)} segments, total duration: {total_duration:.2f}s"
            )

            if total_duration > max_duration:
                logger.warning(
                    f"Total duration {total_duration:.2f}s exceeds limit {max_duration}s"
                )

            return segments

        except Exception as e:
            logger.error(f"Failed to process: {e}")
            logger.debug(f"Last response: {response['message']['content']}")
            raise

    def _extract_json(self, content: str) -> str:
        if "```" in content:
            content = content.split("```")[1].strip()
        return content[content.find("{") : content.rfind("}") + 1]
