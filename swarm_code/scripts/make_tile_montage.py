#!/usr/bin/env python3
"""Create labelled montages covering every non-empty 1.2 km review tile.

  .venv/bin/python code/make_tile_montage.py data/derived/0072

Montages are for visual review only; native per-tile PNGs remain 3 m/pixel.
"""
from pathlib import Path
import argparse
import json
from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("derived", type=Path)
    p.add_argument("--columns", type=int, default=4)
    p.add_argument("--thumb", type=int, default=250)
    args = p.parse_args()
    tiles = json.loads((args.derived / "tiles.json").read_text())["tiles"]
    for layer in ("relief15", "hillshade"):
        present = [tile for tile in tiles
                   if (args.derived / "tiles" / f"{tile['id']}_{layer}.png").exists()]
        if not present:
            raise ValueError(f"no {layer} tiles in {args.derived}")
        w, h, label = args.thumb, args.thumb, 22
        rows = (len(present) + args.columns - 1) // args.columns
        sheet = Image.new("RGB", (args.columns * w, rows * (h + label)), "white")
        draw = ImageDraw.Draw(sheet)
        for index, tile in enumerate(present):
            image = Image.open(args.derived / "tiles" / f"{tile['id']}_{layer}.png").convert("RGB")
            image = image.resize((w, h), Image.Resampling.NEAREST)
            x, y = (index % args.columns) * w, (index // args.columns) * (h + label)
            sheet.paste(image, (x, y + label))
            draw.text((x + 4, y + 4), f"{tile['id']} | {layer} | 1.2km", fill="black")
        destination = args.derived / f"tile_montage_{layer}.jpg"
        sheet.save(destination, quality=90)
        (args.derived / f"tile_montage_{layer}.json").write_text(json.dumps({
            "tile_ids": [tile["id"] for tile in present], "tile_count": len(present),
            "source_pixel_m": 3, "montage": "resized thumbnails; inspect native tile for measurement"},
            indent=2) + "\n")
        print(destination, len(present))


if __name__ == "__main__":
    main()
