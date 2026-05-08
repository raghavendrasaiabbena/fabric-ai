"""
Downloads AI super-resolution model weights.
Run once before starting the server.
"""

import os
import sys
import requests

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODELS = {
    "EDSR_x4.pb": {
        "url": "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb",
        "size_mb": 150,
        "desc": "EDSR 4x Super Resolution (OpenCV DNN)"
    },
    "LapSRN_x4.pb": {
        "url": "https://github.com/fannymonori/TF-LapSRN/raw/master/export/LapSRN_x4.pb",
        "size_mb": 2,
        "desc": "LapSRN 4x Super Resolution (fast fallback)"
    },
}


def download_file(url: str, dest: str, desc: str):
    print(f"\nDownloading {desc}...")
    print(f"  URL: {url}")
    print(f"  Dest: {dest}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    chunk_size = 1024 * 1024  # 1 MB

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    bar = "#" * int(pct / 5)
                    print(f"\r  [{bar:<20}] {pct:.1f}%", end="", flush=True)

    print(f"\n  Done: {dest}")


def main():
    print("=" * 60)
    print("Fabric AI — Model Downloader")
    print("=" * 60)

    for filename, info in MODELS.items():
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.exists(dest):
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"\n[SKIP] {filename} already exists ({size_mb:.1f} MB)")
            continue
        try:
            download_file(info["url"], dest, info["desc"])
        except Exception as e:
            print(f"\n[ERROR] Failed to download {filename}: {e}")
            print("        Falling back to Pillow LANCZOS upscaling.")

    print("\n" + "=" * 60)
    print("Model download complete. Run: python main.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
