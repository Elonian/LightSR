import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def default_output_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "datasets" / "SEN2NAIPv2"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the SEN2NAIPv2 dataset snapshot.")
    parser.add_argument(
        "--output-dir",
        default=str(default_output_dir()),
        help="Local directory where the Hugging Face dataset snapshot will be stored.",
    )
    args = parser.parse_args()

    local_dir = args.output_dir
    snapshot_download(
        repo_id="tacofoundation/SEN2NAIPv2",
        repo_type="dataset",
        local_dir=local_dir,
    )
    print("Downloaded dataset to:", local_dir)
