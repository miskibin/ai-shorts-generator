from enum import Enum
from pathlib import Path
import cv2
from loguru import logger
import numpy as np


class AspectRatio(Enum):
    SQUARE = (720, 720)  # 1:1
    PORTRAIT = (720, 960)  # 3:4
    STORY = (1080, 1920)  # 9:16


def moving_average(values, window_size):
    weights = np.exp(np.linspace(-1.0, 0.0, window_size))
    weights /= weights.sum()
    return np.convolve(values, weights, mode="valid")


from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import Tuple


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
    bg_padding = 30  # Update background padding for larger text
    text_padding = 10

    # Draw semi-transparent background with rounded corners
    bg_x1, bg_y1 = x - bg_padding, y - total_height - bg_padding
    bg_x2, bg_y2 = x + max_width + bg_padding, y + bg_padding
    bg_color = (0, 0, 0, 160)  # More subtle opacity for better aesthetics
    draw.rounded_rectangle([bg_x1, bg_y1, bg_x2, bg_y2], radius=15, fill=bg_color)

    # Draw text with shadow for better readability
    current_y = y - total_height
    shadow_offset = 2
    shadow_color = (0, 0, 0, 150)  # Semi-transparent black for soft shadow

    for i, line in enumerate(lines):
        # Draw text shadow
        draw.text(
            (x + shadow_offset, current_y + shadow_offset),
            line,
            font=font,
            fill=shadow_color,
        )

        # Draw text stroke/outline (if stroke_width > 0)
        if stroke_width > 0:
            draw.text(
                (x, current_y),
                line,
                font=font,
                fill=(0, 0, 0),  # Black outline
                stroke_width=stroke_width,
            )

        # Draw main text
        draw.text((x, current_y), line, font=font, fill=color)

        current_y += line_heights[i]

    # Convert back to numpy array and BGR color space
    result_array = np.array(pil_img)
    result_array = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)

    return result_array


def calculate_crop_dimensions(
    frame_width: int, frame_height: int, target_ratio: float, margin: float = 0.6
) -> tuple[int, int]:
    """Calculate crop dimensions with much wider view"""
    crop_height = frame_height
    crop_width = int(crop_height * target_ratio)

    # Use larger margin for much wider view
    effective_width = int(crop_width * (1 - margin))
    effective_height = int(crop_height * (1 - margin))

    return effective_width, effective_height


def adjust_crop_box(
    center_x: int,
    center_y: int,
    frame_width: int,
    frame_height: int,
    target_width: int,
    target_height: int,
) -> tuple[int, int, int, int]:
    """Calculate crop coordinates with better face positioning"""
    # Center point should be higher in the frame for better composition
    center_y = int(center_y * 0.8)  # Move center point up by 20%

    # Calculate initial crop area
    x_start = max(0, center_x - target_width // 2)
    x_end = min(frame_width, x_start + target_width)
    y_start = max(0, center_y - target_height // 3)  # Use 1/3 for better head room
    y_end = min(frame_height, y_start + target_height)

    # Adjust if we hit frame boundaries
    if x_start == 0:
        x_end = target_width
    elif x_end == frame_width:
        x_start = frame_width - target_width

    if y_start == 0:
        y_end = target_height
    elif y_end == frame_height:
        y_start = frame_height - target_height

    return x_start, y_start, x_end, y_end
