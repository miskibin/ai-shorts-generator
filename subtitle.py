import whisperx
import torch
import json
from pathlib import Path


class SubtitleGenerator:
    def __init__(self, model_size="base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = whisperx.load_model(
            model_size, device=self.device, compute_type="float32"
        )

    def process(self, audio_path: str, output_path: str = None):
        try:
            # Transcribe and align
            result = self.model.transcribe(audio_path, batch_size=16)
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"], device=self.device
            )
            aligned = whisperx.align(
                result["segments"], model_a, metadata, audio_path, self.device
            )

            # Simple output format with safe data extraction
            output = {"language": result["language"], "words": []}

            for seg in aligned.get("word_segments", []):
                try:
                    word_data = {
                        "text": seg.get("word", ""),
                        "start": round(float(seg.get("start", 0)), 3),
                        "end": round(float(seg.get("end", 0)), 3),
                    }
                    output["words"].append(word_data)
                except (KeyError, ValueError, TypeError) as e:
                    print(f"Warning: Skipped malformed segment: {seg}, Error: {e}")
                    continue

            if output_path:
                Path(output_path).write_text(
                    json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
                )

            return output

        except Exception as e:
            raise RuntimeError(f"Processing failed: {str(e)}") from e


if __name__ == "__main__":
    generator = SubtitleGenerator()
    result = generator.process("temp_audio.wav", "output.json")
