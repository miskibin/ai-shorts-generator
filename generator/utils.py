from enum import Enum
from pathlib import Path
import cv2
from loguru import logger
import numpy as np
import torch

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import Tuple


class AspectRatio(Enum):
    SQUARE = (720, 720)  # 1:1
    PORTRAIT = (720, 960)  # 3:4
    STORY = (1080, 1920)  # 9:16


def moving_average(values, window_size):
    weights = np.exp(np.linspace(-1.0, 0.0, window_size))
    weights /= weights.sum()
    return np.convolve(values, weights, mode="valid")


def has_cuda() -> bool:
    """Check if CUDA is available and working"""
    try:
        if torch.cuda.is_available():
            # Try to initialize CUDA to check if it really works
            torch.cuda.init()
            return True
        return False
    except Exception as e:
        logger.warning(f"CUDA initialization failed: {e}")
        return False


def draw_multiline_text(
    img_array: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font_path: Path,
    font_size: int = 48,  # Increased default font size
    color: Tuple[int, int, int] = (255, 255, 255),
    stroke_width: int = 3,  # Increased stroke width
) -> np.ndarray:
    img_array_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_array_rgb)
    draw = ImageDraw.Draw(pil_img, "RGBA")  # Use RGBA for transparency support
    font = ImageFont.truetype(str(font_path.resolve()), font_size)
    lines = text.split("\n")

    # Calculate text dimensions for centering
    max_width = 0
    total_height = 0
    line_heights = []

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        max_width = max(max_width, width)
        line_heights.append(height)
        total_height += height

    x, y = pos
    x = x - max_width // 2  # Center horizontally by adjusting x position
    bg_padding = 40  # Increased padding

    # Draw single solid background without gradient
    bg_x1, bg_y1 = x - bg_padding, y - total_height - bg_padding
    bg_x2, bg_y2 = x + max_width + bg_padding, y + bg_padding
    bg_color = (0, 0, 0, 200)  # Single solid background color
    draw.rounded_rectangle([bg_x1, bg_y1, bg_x2, bg_y2], radius=20, fill=bg_color)

    current_y = y - total_height
    shadow_offset = 3  # Increased shadow offset
    shadow_color = (0, 0, 0, 180)  # Darker shadow

    for i, line in enumerate(lines):
        line_bbox = draw.textbbox((0, 0), line, font=font)
        line_width = line_bbox[2] - line_bbox[0]
        line_x = x + (max_width - line_width) // 2  # Center each line individually

        # Draw shadow
        draw.text(
            (line_x + shadow_offset, current_y + shadow_offset),
            line,
            font=font,
            fill=shadow_color,
        )

        # Thicker outline for better contrast
        if stroke_width > 0:
            draw.text(
                (line_x, current_y),
                line,
                font=font,
                fill=(0, 0, 0),  # Black outline
                stroke_width=stroke_width + 1,
            )

        # Main text (brighter white)
        draw.text((line_x, current_y), line, font=font, fill=(255, 255, 255))

        current_y += line_heights[i]

    # Convert back to numpy array and BGR color space
    result_array = np.array(pil_img)
    result_array = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)

    return result_array
