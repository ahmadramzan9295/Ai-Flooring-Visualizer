"""
Download Segment Anything Model (SAM) Checkpoint

This script downloads the SAM ViT-H checkpoint (~2.4GB) to the project directory.
"""

import urllib.request
import os
import sys

MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
MODEL_FILENAME = "sam_vit_h_4b8939.pth"


def download_model(target_dir: str = ".") -> str:
    """
    Download the SAM model checkpoint.

    Args:
        target_dir: Directory to save the model. Defaults to current directory.

    Returns:
        Path to the downloaded model file.
    """
    os.makedirs(target_dir, exist_ok=True)
    model_path = os.path.join(target_dir, MODEL_FILENAME)

    if os.path.exists(model_path):
        print(f"Model already exists at: {model_path}")
        return model_path

    print(f"Downloading SAM model to: {model_path}")
    print("This may take a while (~2.4GB)...")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        bar_len = 40
        filled = int(bar_len * percent / 100)
        bar = "=" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\r[{bar}] {percent:.1f}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(MODEL_URL, model_path, reporthook=progress_hook)
    print("\nDownload complete!")
    return model_path


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    download_model(target)
