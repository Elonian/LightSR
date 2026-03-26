import argparse
from pathlib import Path

import kagglehub


def default_output_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "datasets"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download DF2K to a local datasets directory.")
    parser.add_argument(
        "--output-dir",
        default=str(default_output_dir()),
        help="Directory where KaggleHub should place the downloaded dataset.",
    )
    args = parser.parse_args()

    path = kagglehub.dataset_download(
        "anvu1204/df2kdata",
        force_download=True,
        output_dir=args.output_dir,
    )
    print("Path to dataset files:", path)
