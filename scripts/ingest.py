"""One-time ingestion helpers for the HF dataset.

This script handles downloading and uploading .rrd files.
C3D → .rrd conversion lives in the HF dataset repo (scripts/convert.py).

Usage::

    # Download .rrd files from HuggingFace (CI or fresh clone)
    uv run python scripts/ingest.py pull -o data/rrd/

    # Push .rrd files to HuggingFace (after running convert.py)
    uv run python scripts/ingest.py push --rrd-dir data/rrd/
"""

import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_ID = "hudsonburke/rat-hindlimb-mocap"


def push_rrd_to_hf(rrd_dir: str, repo_id: str = REPO_ID) -> None:
    """Push .rrd files to a HuggingFace dataset."""
    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        raise ImportError("huggingface-hub is required. Install: uv pip install huggingface-hub")

    token = os.environ.get("HF_TOKEN")
    if token:
        login(token)

    api = HfApi()
    rrd_path = Path(rrd_dir)
    rrd_files = sorted(rrd_path.glob("*.rrd"))

    if not rrd_files:
        logger.error(f"No .rrd files found in {rrd_dir}")
        return

    logger.info(f"Uploading {len(rrd_files)} .rrd files to {repo_id}/rrd/")

    for rrd_file in rrd_files:
        dest = f"rrd/{rrd_file.name}"
        logger.info(f"  Uploading {rrd_file.name}")
        api.upload_file(
            path_or_fileobj=str(rrd_file),
            path_in_repo=dest,
            repo_id=repo_id,
            repo_type="dataset",
        )

    logger.info(f"Push complete: {len(rrd_files)} files uploaded to {repo_id}")


def pull_rrd_from_hf(output_dir: str, repo_id: str = REPO_ID) -> None:
    """Download .rrd files from HuggingFace dataset."""
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

    try:
        files = api.list_repo_files(repo_id, repo_type="dataset")
        rrd_files = [f for f in files if f.startswith("rrd/") and f.endswith(".rrd")]
    except Exception as e:
        logger.error(f"Failed to list repo files: {e}")
        return

    if not rrd_files:
        logger.warning(f"No .rrd files found in {repo_id}/rrd/")
        return

    logger.info(f"Downloading {len(rrd_files)} .rrd files from {repo_id}")

    for rrd_file in rrd_files:
        filename = Path(rrd_file).name
        dest = output_path / filename
        if dest.exists():
            logger.info(f"  Skipping {filename} (already exists)")
            continue

        logger.info(f"  Downloading {filename}")
        hf_hub_download(
            repo_id,
            rrd_file,
            repo_type="dataset",
            local_dir=str(output_path),
            local_dir_use_symlinks=False,
        )

    logger.info(f"Download complete: {len(rrd_files)} .rrd files in {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Move .rrd files to/from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Conversion (C3D → .rrd) lives in the HF dataset repo:\n"
            "  python scripts/convert.py --c3d-dir sourcedata/ -o rrd/ -j 8\n"
            "\nExamples:\n"
            "  python scripts/ingest.py pull -o data/rrd/\n"
            "  python scripts/ingest.py push --rrd-dir data/rrd/\n"
        ),
    )
    sub = parser.add_subparsers(dest="command")

    push_p = sub.add_parser("push", help="Push .rrd files to HuggingFace")
    push_p.add_argument("--rrd-dir", default="data/rrd")
    push_p.add_argument("--repo", default=REPO_ID)

    pull_p = sub.add_parser("pull", help="Download .rrd files from HuggingFace")
    pull_p.add_argument("-o", "--output", default="data/rrd")
    pull_p.add_argument("--repo", default=REPO_ID)

    args = parser.parse_args()

    if args.command == "push":
        push_rrd_to_hf(args.rrd_dir, args.repo)
    elif args.command == "pull":
        pull_rrd_from_hf(args.output, args.repo)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
