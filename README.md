# 🎥 AI Shorts Generator

AI Shorts Generator is an intelligent video processing tool that automatically creates highlight reels with subtitles from your videos. Perfect for political speeches, interviews, or any dialogue-driven content, it extracts key segments, generates precise subtitles, and produces a polished highlights video—all with minimal manual intervention.

---

## ✨ Overview

- **Automatic Highlight Extraction:**  
  The tool analyzes your video, extracts audio, transcribes speech, and leverages a powerful LLM to identify continuous, coherent fragments for your highlight reel.

- **Dynamic Subtitle Generation:**  
  Advanced transcription and alignment techniques produce precise subtitles that sync perfectly with your video content.

- **Smart Video Processing:**  
  With built-in face tracking and dynamic cropping, every highlight is beautifully framed, ensuring a professional, cinematic result.

- **Flexible Customization:**  
  Adjust parameters such as maximum highlight duration and LLM model usage to suit your specific needs.

---

## 🚀 Installation

Install AI Shorts Generator easily from the wheel file provided on the [latest release page](https://github.com/miskibin/ai-shorts-generator/releases/latest).

Simply download the `.whl` file from the releases page and install it with pip:

```bash
pip install path/to/ai_shorts_generator-<version>-py3-none-any.whl
```

*All necessary dependencies will be automatically installed.*

---

## 🛠️ Usage

Once installed, you can process your videos directly from the command line. Here’s a typical command:

```bash
ai-shorts-generator process_video path/to/your_video.mp4 --output path/to/output_highlights.mp4 --max-duration 60.0 --llm-model <model_name> --model-size base
```

### Command Options

- **`video_path`**  
  *Path to the input video file.*

- **`--output`**  
  *Destination path for the generated highlights video.*

- **`--llm-model`**  
  *(Optional)* Specify an LLM model for advanced segment selection.  
  **Note:** To use the LLM functionality, you must install [Ollama](https://ollama.ai/) and the appropriate models.

- **`--model-size`**  
  *Choose the model size for subtitle generation (e.g., `base`).*

- **`--max-duration`**  
  *Maximum duration (in seconds) for the highlights video.*

The tool will:

1. **Extract Audio:**  
   Pull the audio track from your video.

2. **Generate Transcription & Subtitles:**  
   Transcribe and align speech to create precise subtitles.

3. **Process Video:**  
   Utilize smart cropping and face tracking to produce a well-framed highlights reel.

---

## 📌 Getting Started

1. **Install the Package:**  
   Download and install the latest wheel file from the [release page](https://github.com/miskibin/ai-shorts-generator/releases/latest).

2. **(Optional) Configure LLM Support:**  
   If you want to enable advanced highlight selection using an LLM, make sure to install [Ollama](https://ollama.ai/) and download the necessary models.

3. **Run the Command:**  
   Execute the CLI command as shown above to process your video file.

4. **Review Your Highlights:**  
   Find your generated highlights video at the specified output path and enjoy your content reimagined!s!
