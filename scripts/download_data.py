"""Fetch the Stack Overflow Python Q&A dataset from Kaggle.

Needs Kaggle credentials. Either:
  - put kaggle.json at ~/.kaggle/kaggle.json, or
  - export KAGGLE_USERNAME and KAGGLE_KEY.

Get a token at https://www.kaggle.com/settings -> "Create New API Token".
"""

import shutil
from pathlib import Path

import kagglehub

DATASET = "stackoverflow/pythonquestions"
WANTED = ("Questions.csv", "Answers.csv")
RAW_DIR = Path("data/raw")


def main() -> None:
    src = Path(kagglehub.dataset_download(DATASET))
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for name in WANTED:
        found = next(src.rglob(name), None)
        if found is None:
            raise FileNotFoundError(f"{name} not found under {src}")
        dest = RAW_DIR / name
        shutil.copy2(found, dest)
        print(f"{name}: {dest} ({dest.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
