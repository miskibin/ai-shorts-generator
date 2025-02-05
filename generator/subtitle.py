from pydantic import BaseModel
import whisperx
import torch
import json
from pathlib import Path
from loguru import logger


class Sentence(BaseModel):
    text: str
    start: float
    end: float
    words: list[dict]


class SubtitleGenerator:
    def __init__(self, model_size="base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            f"Initializing SubtitleGenerator with {model_size} model on {self.device}"
        )
        self.model = whisperx.load_model(
            model_size, device=self.device, compute_type="float32"
        )

    def process(self, audio_path: Path) -> list[Sentence]:
        try:
            # Transcribe and align
            result = self.model.transcribe(str(audio_path), language="pl")
            logger.debug(f"Transcribed {len(result['segments'])} segments")

            model_a, metadata = whisperx.load_align_model(
                language_code="pl", device=self.device
            )
            aligned = whisperx.align(
                result["segments"], model_a, metadata, str(audio_path), self.device
            )
            logger.debug("Alignment completed")

            return self._merge_into_sentences(
                self._normalize_word_segments(aligned.get("segments", []))
            )

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise

    def _normalize_word_segments(self, segments: list[dict]) -> list[dict]:
        """Normalize word segments to ensure consistent structure"""
        normalized = []
        for segment in segments:
            if "words" in segment:
                # Handle word-level alignments
                for word in segment["words"]:
                    if isinstance(word, dict) and "word" in word:
                        normalized.append(
                            {
                                "word": word["word"],
                                "start": word.get("start", segment["start"]),
                                "end": word.get("end", segment["end"]),
                            }
                        )
            else:
                normalized.append(
                    {
                        "word": segment.get("text", "").strip(),
                        "start": segment.get("start", 0),
                        "end": segment.get("end", 0),
                    }
                )
        return normalized

    def _merge_into_sentences(self, words: list[dict]) -> list[Sentence]:
        if not words:
            return []

        sentences: list[Sentence] = []
        current_words = []
        current_text = []

        def is_sentence_boundary(word: dict, next_word: dict = None) -> bool:
            # Polish sentence boundary detection
            if word["word"].rstrip().endswith((".", "!", "?", "...")):
                return True

            if next_word is None:
                return True

            # Safer time gap check
            try:
                time_gap = next_word.get("start", 0) - word.get("end", 0)
                if time_gap > 0.7:
                    return True
            except (TypeError, KeyError):
                return False

            # Polish conjunctions
            conjunctions = {"i", "oraz", "ale", "lecz", "czy", "albo", "lub"}
            if next_word["word"].lower().strip() in conjunctions:
                return False

            return False

        for i, word in enumerate(words):
            current_words.append(word)
            current_text.append(word["word"])

            next_word = words[i + 1] if i < len(words) - 1 else None

            if is_sentence_boundary(word, next_word):
                if current_words:
                    sentences.append(
                        Sentence(
                            text=" ".join(current_text).strip(),
                            start=current_words[0]["start"],
                            end=current_words[-1]["end"],
                            words=current_words.copy(),
                        )
                    )
                    current_words.clear()
                    current_text.clear()

        # Handle any remaining words
        if current_words:
            sentences.append(
                Sentence(
                    text=" ".join(current_text).strip(),
                    start=current_words[0]["start"],
                    end=current_words[-1]["end"],
                    words=current_words,
                )
            )

        return sentences


if __name__ == "__main__":
    generator = SubtitleGenerator()
    result = generator.process(
        Path(__file__).parent.parent / "inputs" / "temp_audio.wav"
    )
    output_path = "output.json"
    output = {
        "language": "pl",
        "sentences": [
            {
                "text": sentence.text,
                "start": round(sentence.start, 3),
                "end": round(sentence.end, 3),
                "words": sentence.words,
            }
            for sentence in result
        ],
    }
    Path(output_path).write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
