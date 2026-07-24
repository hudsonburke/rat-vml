"""Download sourcedata from HuggingFace dataset.

Bypasses git LFS — downloads C3D archives directly via the HF Hub API.
Use this when git lfs pull doesn't work (e.g., permission issues, missing LFS server).

Usage::

    uv run python scripts/download_sourcedata.py \
        --subjects BAA01 BAA02 BAA03 \
        --sessions Baseline \
        -o data/raw/sourcedata

    # Download everything (large!)
    uv run python scripts/download_sourcedata.py --all -o data/raw/sourcedata
"""

import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_ID = "hudsonburke/rat-hindlimb-mocap"


def download_sourcedata(
    output_dir: str,
    subjects: list[str] | None = None,
    sessions: list[str] | None = None,
    download_all: bool = False,
) -> None:
    """Download C3D archives from HuggingFace dataset via the Hub API.

    Parameters
    ----------
    output_dir : str
        Local directory to download to.
    subjects : list[str] or None
        Subject IDs to download (e.g. ["BAA01", "BAA02"]).
        If None and download_all=False, downloads nothing.
    sessions : list[str] or None
        Session names to filter (e.g. ["Baseline", "Week24"]).
        If None, downloads all sessions for each subject.
    download_all : bool
        If True, downloads all sourcedata files.
    """
    try:
        from huggingface_hub import HfApi, login, hf_hub_download
    except ImportError:
        raise ImportError("huggingface-hub is required. Install: uv pip install huggingface-hub")

    token = os.environ.get("HF_TOKEN")
    if token:
        login(token)

    api = HfApi()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # List all sourcedata files
    try:
        all_files = api.list_repo_files(REPO_ID, repo_type="dataset")
    except Exception as e:
        logger.error(f"Failed to list repo files: {e}")
        return

    sourcedata_files = [f for f in all_files if f.startswith("sourcedata/")]

    if download_all:
        target_files = sourcedata_files
    elif subjects:
        target_files = []
        for f in sourcedata_files:
            # sourcedata/{subject}/{file}
            parts = f.split("/")
            if len(parts) >= 2 and parts[1] in subjects:
                if sessions is None:
                    target_files.append(f)
                else:
                    # Check if filename matches a session
                    filename = parts[-1] if len(parts) > 2 else ""
                    if any(s.lower() in filename.lower() for s in sessions):
                        target_files.append(f)
                    elif filename.endswith(".enf"):  # Always include .enf files
                        target_files.append(f)
    else:
        logger.info("No subjects specified. Use --subjects or --all")
        return

    logger.info(f"Downloading {len(target_files)} files from {REPO_ID}")

    downloaded = 0
    for f in target_files:
        local_path = output_path / f
        if local_path.exists():
            logger.info(f"  Skipping {f} (already exists)")
            continue

        logger.info(f"  Downloading {f}")
        try:
            hf_hub_download(
                REPO_ID,
                f,
                repo_type="dataset",
                local_dir=str(output_path),
                local_dir_use_symlinks=False,
            )
            downloaded += 1
        except Exception as e:
            logger.warning(f"  Failed to download {f}: {e}")

    logger.info(f"Downloaded {downloaded} files to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download sourcedata from HuggingFace dataset (bypasses git LFS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Download specific subjects\n"
            "  python scripts/download_sourcedata.py --subjects BAA01 BAA02 -o data/raw\n"
            "\n"
            "  # Download specific sessions for a subject\n"
            "  python scripts/download_sourcedata.py --subjects BAA01 --sessions Baseline Week24 -o data/raw\n"
            "\n"
            "  # Download everything\n"
            "  python scripts/download_sourcedata.py --all -o data/raw\n"
        ),
    )
    parser.add_argument("--subjects", nargs="+", help="Subject IDs to download")
    parser.add_argument("--sessions", nargs="+", help="Session names to filter")
    parser.add_argument("--all", action="store_true", help="Download all sourcedata")
    parser.add_argument("-o", "--output", default="data/raw/sourcedata", help="Output directory")

    args = parser.parse_args()
    download_sourcedata(args.output, args.subjects, args.sessions, args.all)


if __name__ == "__main__":
    main()
