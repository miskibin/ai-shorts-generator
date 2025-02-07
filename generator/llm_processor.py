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
    PROMPT_TEMPLATE = """You are an expert speech analyst. Your task is to analyze this political speech and select the most important and coherent segments that capture the main message.

    First, identify the main topic and key points of the speech. Then, select continuous segments that best represent these points while maintaining natural flow and coherence.

    Requirements:
    1. First identify and include the segment that introduces the main topic.
    2. Select the most impactful and important segments that support or develop the main topic.
    3. Make as few cuts as possible - prefer longer continuous segments over multiple short ones.
    4. The selected segments must flow naturally when played together - ensure logical connections between segments.
    5. Total duration MUST be between 80-100% of {max_duration}s, never exceeding it.
    6. You may select up to 3 continuous segments if needed, but prefer fewer if possible.
    
    Speech segments:
    {segments}

    Return JSON in this exact format:
    {{
         "analysis": {{
            "main_topic": "Brief description of the main topic",
            "selection_reasoning": "Brief explanation of why these segments were chosen"
        }},
        "selections": [
            {{
                "start_time": float,
                "end_time": float,
            }}
            // up to 3 selections allowed
        ],
       
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
