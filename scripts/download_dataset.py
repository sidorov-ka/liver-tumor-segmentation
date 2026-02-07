"""
Download LITS17 (LiTS train/val) from Kaggle and save path for prepare_dataset.py.
Requires KAGGLE_API_TOKEN in environment or in .env (or ~/.kaggle/kaggle.json).
"""
from pathlib import Path

import kagglehub
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
DOWNLOADED_PATH_FILE = PROJECT_ROOT / "data" / "downloaded_path.txt"
DATASET = "javariatahir/litstrain-val"


def main():
    print(f"Downloading {DATASET} (this may take a while, ~53 GB)...")
    path = kagglehub.dataset_download(DATASET)
    path = Path(path).resolve()
    for candidate in [path, path.parent]:
        if (candidate / "train_CT").exists() and (candidate / "train_mask").exists():
            path = candidate
            break
    DOWNLOADED_PATH_FILE.parent.mkdir(parents=True, exist_ok=True)
    DOWNLOADED_PATH_FILE.write_text(str(path), encoding="utf-8")
    print("Path to dataset files:", path)
    print("Saved to", DOWNLOADED_PATH_FILE)


if __name__ == "__main__":
    main()
