from dataclasses import dataclass
import ollama
import json
from pathlib import Path
from loguru import logger

from generator.subtitle import Sentence
from pydantic import BaseModel


class TimeSegment(BaseModel):
    start_time: float
    end_time: float
    duration: float = 0.0

    def __init__(self, **data):
        super().__init__(**data)
        self.duration = round(self.end_time - self.start_time, 2)


class LLMProcessor:
    PROMPT_TEMPLATE = """Analyze this political speech (given as a single text chunked into sentences) and select up to 3 continuous fragments that together form a coherent and logical segment. If one long segment makes the most sense, you can select just that one.
    Requirements:
    1. The selection must include the segment where the topic of the statement is introduced.  
    2. The total duration must not exceed {max_duration}s; aim to use almost all of it.  
    3. Each fragment must be a continuous block of sentences.  
    4. The final selection should read as a naturally connected piece, maintaining logical flow.  

    Speech segments:
    {segments}

    Return JSON in this exact format:
    {{
        "selections": [
            {{
                "start_time": float,
                "end_time": float,
            }}
            // up to 3 selections allowed
        ]
    }}
    """

    def __init__(self, model_name: str | None):
        self.model = model_name
        logger.info(f"Initialized LLM processor with {model_name}")

    def process(
        self, sentences: list[Sentence], max_duration: float
    ) -> list[TimeSegment]:
        if self.model is None:
            logger.warning(
                "Model is None, selecting beginning sentences up to max_duration"
            )
            segments = []
            current_duration = 0.0

            for sentence in sentences:
                if current_duration + (sentence.end - sentence.start) > max_duration:
                    break
                segment = TimeSegment(
                    start_time=sentence.start,
                    end_time=sentence.end,
                )
                segments.append(segment)
                current_duration += segment.duration

            logger.info(
                f"Selected {len(segments)} segments, total duration: {current_duration:.2f}s"
            )
            return segments

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
        logger.debug(f"Received response: {result}")
        # Convert all selections to TimeSegments
        segments = [
            TimeSegment(
                start_time=float(sel["start_time"]),
                end_time=float(sel["end_time"]),
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

    def _extract_json(self, content: str) -> str:
        if "```" in content:
            content = content.split("```")[1].strip()
        return content[content.find("{") : content.rfind("}") + 1]
