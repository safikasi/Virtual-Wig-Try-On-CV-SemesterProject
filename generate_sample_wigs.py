"""
Sample Wig Generator
Creates sample wig PNG images for testing the Virtual Wig Try-On app.
Run this script once to generate sample wigs in the assets/wigs folder.
"""

import cv2
import numpy as np
from pathlib import Path


def create_wig_base(width: int, height: int) -> np.ndarray:
    """Create a transparent base image for a wig."""
    return np.zeros((height, width, 4), dtype=np.uint8)


def draw_hair_strand(image: np.ndarray, start: tuple, end: tuple,
                     color: tuple, thickness: int = 2):
    """Draw a single hair strand with some curve."""
    cv2.line(image, start, end, color, thickness, cv2.LINE_AA)


def generate_short_hair(color: tuple, name: str) -> np.ndarray:
    """Generate a short hairstyle."""
    width, height = 400, 250
    img = create_wig_base(width, height)

    # Draw hair as overlapping ellipses and lines
    center_x = width // 2
    base_y = height - 30

    # Main hair volume
    for i in range(50):
        offset_x = np.random.randint(-150, 150)
        offset_y = np.random.randint(20, 150)
        size = np.random.randint(30, 80)

        cv2.ellipse(img, (center_x + offset_x, base_y - offset_y),
                    (size, size // 2), np.random.randint(-30, 30),
                    0, 360, (*color, 255), -1, cv2.LINE_AA)

    # Add some texture with lines
    for i in range(100):
        start_x = np.random.randint(50, width - 50)
        start_y = np.random.randint(50, height - 50)
        length = np.random.randint(10, 40)
        angle = np.random.uniform(-0.5, 0.5)

        end_x = int(start_x + length * np.sin(angle))
        end_y = int(start_y + length)

        darker = tuple(max(0, c - 30) for c in color)
        cv2.line(img, (start_x, start_y), (end_x, end_y),
                 (*darker, 200), 1, cv2.LINE_AA)

    return img


def generate_long_hair(color: tuple, name: str) -> np.ndarray:
    """Generate a long hairstyle."""
    width, height = 500, 400
    img = create_wig_base(width, height)

    center_x = width // 2

    # Draw long flowing strands
    for i in range(200):
        start_x = np.random.randint(50, width - 50)
        start_y = np.random.randint(10, 60)

        # Create flowing curve
        wave_amplitude = np.random.randint(5, 20)
        strand_length = np.random.randint(250, 350)

        points = []
        for y in range(start_y, min(start_y + strand_length, height - 10), 5):
            wave = int(wave_amplitude * np.sin((y - start_y) * 0.05))
            x = start_x + wave + int((y - start_y) * 0.1 * (1 if start_x > center_x else -1))
            points.append([x, y])

        if len(points) > 2:
            points = np.array(points, dtype=np.int32)
            shade = np.random.randint(-20, 20)
            strand_color = tuple(max(0, min(255, c + shade)) for c in color)
            cv2.polylines(img, [points], False, (*strand_color, 230),
                          np.random.randint(1, 3), cv2.LINE_AA)

    # Add volume at top
    for i in range(30):
        offset_x = np.random.randint(-180, 180)
        cv2.ellipse(img, (center_x + offset_x, 50),
                    (60, 40), np.random.randint(-20, 20),
                    0, 360, (*color, 255), -1, cv2.LINE_AA)

    return img


def generate_curly_hair(color: tuple, name: str) -> np.ndarray:
    """Generate a curly hairstyle."""
    width, height = 450, 350
    img = create_wig_base(width, height)

    center_x = width // 2

    # Draw curls as small circles and spirals
    for i in range(150):
        x = np.random.randint(40, width - 40)
        y = np.random.randint(30, height - 50)
        radius = np.random.randint(10, 30)

        shade = np.random.randint(-30, 30)
        curl_color = tuple(max(0, min(255, c + shade)) for c in color)

        # Draw curl as overlapping circles
        for j in range(3):
            offset = np.random.randint(-5, 5)
            cv2.circle(img, (x + offset, y + offset), radius,
                       (*curl_color, 200), -1, cv2.LINE_AA)

    # Add volume at crown
    for i in range(20):
        offset_x = np.random.randint(-150, 150)
        cv2.ellipse(img, (center_x + offset_x, 60),
                    (50, 40), 0, 0, 360, (*color, 255), -1, cv2.LINE_AA)

    return img


def generate_spiky_hair(color: tuple, name: str) -> np.ndarray:
    """Generate a spiky hairstyle."""
    width, height = 400, 300
    img = create_wig_base(width, height)

    center_x = width // 2
    base_y = height - 20

    # Draw spiky strands pointing upward
    for i in range(80):
        base_x = np.random.randint(50, width - 50)
        spike_height = np.random.randint(100, 250)
        angle = np.random.uniform(-0.8, 0.8)

        tip_x = int(base_x + spike_height * np.sin(angle) * 0.5)
        tip_y = int(base_y - spike_height)

        # Draw triangular spike
        pts = np.array([
            [base_x - 8, base_y],
            [base_x + 8, base_y],
            [tip_x, tip_y]
        ], np.int32)

        shade = np.random.randint(-20, 20)
        spike_color = tuple(max(0, min(255, c + shade)) for c in color)
        cv2.fillPoly(img, [pts], (*spike_color, 230), cv2.LINE_AA)

    # Base volume
    cv2.ellipse(img, (center_x, base_y), (150, 40), 0, 180, 360,
                (*color, 255), -1, cv2.LINE_AA)

    return img


def generate_afro(color: tuple, name: str) -> np.ndarray:
    """Generate an afro hairstyle."""
    width, height = 500, 400
    img = create_wig_base(width, height)

    center_x, center_y = width // 2, height // 2 + 30

    # Create large rounded afro shape with many small circles
    for i in range(500):
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(0, 180)

        # Bias toward oval shape (wider than tall)
        x = int(center_x + distance * 1.1 * np.cos(angle))
        y = int(center_y - 50 + distance * 0.9 * np.sin(angle))

        radius = np.random.randint(8, 20)

        shade = np.random.randint(-25, 25)
        circle_color = tuple(max(0, min(255, c + shade)) for c in color)

        if 0 <= x < width and 0 <= y < height:
            cv2.circle(img, (x, y), radius, (*circle_color, 220), -1, cv2.LINE_AA)

    return img


def main():
    """Generate all sample wigs."""
    output_dir = Path("assets/wigs")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating sample wig images...")

    # Define hair colors (BGR format)
    colors = {
        "black": (20, 20, 20),
        "brown": (40, 70, 120),
        "blonde": (100, 180, 220),
        "red": (60, 60, 180),
        "gray": (130, 130, 140),
    }

    # Generate different styles
    styles = [
        ("short", generate_short_hair),
        ("long", generate_long_hair),
        ("curly", generate_curly_hair),
        ("spiky", generate_spiky_hair),
        ("afro", generate_afro),
    ]

    count = 0
    for style_name, generator in styles:
        for color_name, color in colors.items():
            name = f"{style_name}_{color_name}"
            wig_img = generator(color, name)

            output_path = output_dir / f"{name}.png"
            cv2.imwrite(str(output_path), wig_img)
            print(f"  Created: {output_path}")
            count += 1

    print(f"\nGenerated {count} sample wigs in '{output_dir}'")
    print("Run 'python main.py' to start the Virtual Wig Try-On app!")


if __name__ == "__main__":
    main()
