#!/usr/bin/env python3
"""Generate SSA CHATBOT raster assets from the approved SSA logo.

The source logo is never overwritten.  This script makes deterministic UI
exports so the organisation mark is not redrawn by an image model.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "docs" / "lg (1).png"
DEFAULT_BACKGROUND = ROOT / "public" / "brand" / "generated" / "ssa-chatbot-background.png"
DEFAULT_OUTPUT = ROOT / "public"

GREEN = "#006A00"
COPPER = "#C27D38"
IVORY = "#F8F3E9"
CHARCOAL = "#24302B"
WHITE = "#FFFFFF"
FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    if not path.exists():
        raise FileNotFoundError(f"Required font is missing: {path}")
    return ImageFont.truetype(str(path), size=size)


def contain(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    return copy


def cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    scale = max(size[0] / image.width, size[1] / image.height)
    resized = image.resize(
        (round(image.width * scale), round(image.height * scale)),
        Image.Resampling.LANCZOS,
    )
    left = (resized.width - size[0]) // 2
    top = (resized.height - size[1]) // 2
    return resized.crop((left, top, left + size[0], top + size[1]))


def centered_paste(canvas: Image.Image, image: Image.Image, center: tuple[int, int]) -> None:
    canvas.alpha_composite(image, (center[0] - image.width // 2, center[1] - image.height // 2))


def extract_symbol(source: Image.Image) -> Image.Image:
    # The approved mark occupies the left section of the supplied lockup.
    # Stop before x=135: the first copper separator starts at that column.
    region = source.crop((0, 0, 131, source.height))
    bbox = region.getchannel("A").getbbox()
    if bbox is None:
        raise ValueError("The source logo has no visible symbol pixels")
    symbol = region.crop(bbox)
    side = max(symbol.size) + 24
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    centered_paste(square, symbol, (side // 2, side // 2))
    return square


def make_icon(symbol: Image.Image, size: int, *, maskable: bool = False) -> Image.Image:
    canvas = Image.new("RGBA", (size, size), IVORY)
    draw = ImageDraw.Draw(canvas)
    inset = round(size * (0.075 if not maskable else 0.03))
    draw.rounded_rectangle(
        (inset, inset, size - inset, size - inset),
        radius=round(size * 0.18),
        fill="#FFFFFF",
        outline=COPPER,
        width=max(1, round(size * 0.018)),
    )
    # Maskable artwork needs a larger quiet area because launchers crop it.
    mark_side = round(size * (0.58 if maskable else 0.72))
    mark = contain(symbol, (mark_side, mark_side))
    centered_paste(canvas, mark, (size // 2, size // 2))
    return canvas


def make_lockup(symbol: Image.Image, *, dark: bool) -> Image.Image:
    canvas = Image.new("RGBA", (1200, 300), (0, 0, 0, 0))
    mark = contain(symbol, (230, 230))
    centered_paste(canvas, mark, (130, 150))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle((274, 35, 282, 265), fill=COPPER)
    draw.rectangle((292, 35, 296, 265), fill=COPPER)

    title_font = font(FONT_BOLD, 104)
    subtitle_font = font(FONT_BOLD, 31)
    x, y = 340, 65
    draw.text((x, y), "SSA", font=title_font, fill=COPPER)
    ssa_width = draw.textlength("SSA", font=title_font)
    draw.text((x + ssa_width + 28, y), "CHATBOT", font=title_font, fill=GREEN)
    draw.text(
        (344, 196),
        "GCF PROPOSAL KNOWLEDGE ASSISTANT",
        font=subtitle_font,
        fill=WHITE if dark else CHARCOAL,
    )
    return canvas


def export_assets(source_path: Path, background_path: Path, output: Path) -> None:
    brand = output / "brand"
    icons = brand / "icons"
    splash = brand / "splash"
    avatars = output / "avatars"
    for directory in (brand, icons, splash, avatars):
        directory.mkdir(parents=True, exist_ok=True)

    source = Image.open(source_path).convert("RGBA")
    symbol = extract_symbol(source)
    background = Image.open(background_path).convert("RGB")

    shutil.copyfile(source_path, brand / "ssa-corporate-logo.png")

    mark = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    centered_paste(mark, contain(symbol, (460, 460)), (256, 256))
    mark.save(brand / "ssa-mark-transparent-512.png", optimize=True)

    for size in (192, 512):
        make_icon(symbol, size).save(icons / f"icon-{size}.png", optimize=True)
    make_icon(symbol, 512, maskable=True).save(icons / "icon-maskable-512.png", optimize=True)
    make_icon(symbol, 180).convert("RGB").save(icons / "apple-touch-icon-180.png", optimize=True)
    make_icon(symbol, 512).save(avatars / "ssa_chatbot.png", optimize=True)

    favicon = make_icon(symbol, 64).convert("RGBA")
    favicon.save(output / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48), (64, 64)])

    for theme in ("light", "dark"):
        logo = make_lockup(symbol, dark=theme == "dark")
        logo.save(brand / f"ssa-chatbot-logo-{theme}.png", optimize=True)
        logo.save(output / f"logo_{theme}.png", optimize=True)

    login = cover(background, (1672, 941))
    login.save(brand / "ssa-chatbot-login-background.jpg", quality=90, optimize=True, progressive=True)

    social = cover(background, (1200, 630)).convert("RGBA")
    panel = Image.new("RGBA", social.size, (0, 0, 0, 0))
    ImageDraw.Draw(panel).rounded_rectangle((55, 105, 790, 525), radius=38, fill=(248, 243, 233, 232))
    social = Image.alpha_composite(social, panel)
    lockup = contain(make_lockup(symbol, dark=False), (690, 172))
    social.alpha_composite(lockup, (78, 175))
    ImageDraw.Draw(social).text(
        (108, 389),
        "Evidence-backed answers from indexed GCF proposals",
        font=font(FONT_REGULAR, 27),
        fill=CHARCOAL,
    )
    social.convert("RGB").save(brand / "ssa-chatbot-social-1200x630.jpg", quality=91, optimize=True)

    launch = cover(background, (2048, 2048)).convert("RGBA")
    veil = Image.new("RGBA", launch.size, (248, 243, 233, 118))
    launch = Image.alpha_composite(launch, veil)
    lockup = contain(make_lockup(symbol, dark=False), (1320, 330))
    plate = Image.new("RGBA", launch.size, (0, 0, 0, 0))
    ImageDraw.Draw(plate).rounded_rectangle((280, 725, 1768, 1323), radius=68, fill=(248, 243, 233, 238))
    launch = Image.alpha_composite(launch, plate)
    centered_paste(launch, lockup, (1024, 1000))
    ImageDraw.Draw(launch).text(
        (1024, 1200),
        "Evidence-backed GCF proposal knowledge",
        font=font(FONT_REGULAR, 39),
        fill=CHARCOAL,
        anchor="mm",
    )
    launch.convert("RGB").save(splash / "ssa-chatbot-splash-2048.png", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--background", type=Path, default=DEFAULT_BACKGROUND)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export_assets(args.source.resolve(), args.background.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
