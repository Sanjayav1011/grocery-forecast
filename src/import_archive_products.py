"""Flatten every product workbook in the local archive into one CSV dataset."""

import io
import zipfile
import pandas as pd


from config import ARCHIVE_PATH, ARCHIVE_CSV
OUTPUT_PATH = ARCHIVE_CSV


def build():
    if not ARCHIVE_PATH.exists():
        if ARCHIVE_CSV.exists():
            print(f"archive.zip not found; using existing {ARCHIVE_CSV.name}")
            return
        raise SystemExit(f"Put archive.zip in {ARCHIVE_PATH.parent} or set ARCHIVE_PATH")
    frames = []
    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        for name in archive.namelist():
            if not name.lower().endswith("product_data.xlsx"):
                continue
            frame = pd.read_excel(io.BytesIO(archive.read(name)))
            frame.insert(0, "Source File", name)
            frames.append(frame)

    products = pd.concat(frames, ignore_index=True)
    products.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(products):,} archive products from {len(frames)} Excel files to {OUTPUT_PATH}")
    return products


if __name__ == "__main__":
    build()