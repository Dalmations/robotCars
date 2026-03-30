#!/usr/bin/env python3
"""
Display text on a 0.96" SSD1306 I2C OLED from the command line.

Wiring (SSD1306 I2C → Raspberry Pi GPIO):
  VCC → 3.3V (Pin 1)
  GND → GND  (Pin 6)
  SCL → SCL  (Pin 5, GPIO 3)
  SDA → SDA  (Pin 3, GPIO 2)

Install dependencies:
  pip install luma.oled pillow

Enable I2C on your Pi:
  sudo raspi-config → Interface Options → I2C → Enable
"""

import argparse
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306
from luma.core.render import canvas
from PIL import ImageFont

# ── Config ────────────────────────────────────────────────────────────────────
I2C_PORT    = 1       # I2C bus (1 = /dev/i2c-1 on modern Pis)
I2C_ADDRESS = 0x3C    # Most SSD1306 boards use 0x3C; try 0x3D if it doesn't work

WIDTH     = 128
HEIGHT    = 64
FONT_SIZE = 12        # Increase for larger text (at the cost of fewer lines)


def get_device():
    serial = i2c(port=I2C_PORT, address=I2C_ADDRESS)
    return ssd1306(serial, width=WIDTH, height=HEIGHT)


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Word-wrap text to fit within max_width pixels."""
    words = text.split()
    lines, current = [], ""
    for word in words:
        test = f"{current} {word}".strip()
        w = font.getlength(test)
        if w <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def display_text(text: str, font_size: int = FONT_SIZE):
    device = get_device()

    # Try to load a nicer font; fall back to PIL default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    lines = wrap_text(text, font, WIDTH - 4)  # 2px padding each side
    line_height = font_size + 2

    with canvas(device) as draw:
        for i, line in enumerate(lines):
            y = i * line_height
            if y + line_height > HEIGHT:
                break  # Stop if we've run out of screen space
            draw.text((2, y), line, font=font, fill="white")


def main():
    parser = argparse.ArgumentParser(description="Display text on SSD1306 I2C OLED")
    parser.add_argument("text", nargs="?", help="Text to display (quoted string)")
    parser.add_argument("--font-size", type=int, default=FONT_SIZE, help=f"Font size (default: {FONT_SIZE})")
    args = parser.parse_args()

    # If no argument given, prompt interactively
    text = args.text or input("Enter text to display: ")
    display_text(text, args.font_size)
    print("✓ Displayed on OLED.")


if __name__ == "__main__":
    main()
