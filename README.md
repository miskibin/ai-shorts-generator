# AI Image Generator

A Python-based tool for generating and manipulating images with AI-powered text overlays and effects.

## Features

- Multiple aspect ratio support (Square 1:1, Portrait 3:4, Story 9:16)
- Professional text rendering with:
  - Customizable font sizes and styles
  - Drop shadows
  - Stroke/outline effects
  - Semi-transparent backgrounds
  - Centered text alignment
- Image processing utilities including:
  - Moving average smoothing
  - Color space conversion
  - Multi-line text support

## Requirements

- Python 3.8+
- OpenCV (cv2)
- Pillow
- NumPy
- Loguru

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-generator-2.git
cd ai-generator-2

# Install dependencies
pip install opencv-python pillow numpy loguru
```

## Usage

```python
from generator.utils import draw_multiline_text, AspectRatio
from pathlib import Path

# Load your image
image = cv2.imread('your_image.jpg')

# Add text overlay
font_path = Path('path/to/your/font.ttf')
result = draw_multiline_text(
    img_array=image,
    text="Your\nMultiline\nText",
    pos=(360, 500),  # Center position
    font_path=font_path,
    font_size=48,
    stroke_width=3
)

# Save the result
cv2.imwrite('output.jpg', result)
```

## Aspect Ratios

The project supports multiple aspect ratios through the `AspectRatio` enum:
- `AspectRatio.SQUARE`: 720x720 (1:1)
- `AspectRatio.PORTRAIT`: 720x960 (3:4)
- `AspectRatio.STORY`: 1080x1920 (9:16)

## License

[Your chosen license]

## Contributing

Feel free to open issues or submit pull requests with improvements.
