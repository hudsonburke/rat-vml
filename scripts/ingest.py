"""One-time C3D → .rrd ingestion for the workstation.

Run this on an x86_64 machine with ezc3d and the Rerun C3D importer installed.
It converts all C3D files to .rrd and optionally pushes them to HuggingFace
so CI can download the .rrd files directly (no C3D conversion needed).

Usage::

    # Convert C3D to .rrd (run once on workstation)
    uv run --extra ingest python scripts/ingest.py convert \\
        --c3d-dir /path/to/sourcedata \\
        --output data/rrd/

    # Push .rrd files to HuggingFace dataset
    uv run --extra ingest python scripts/ingest.py push \\
        --rrd-dir data/rrd/ \\
        --repo hudsonburke/rat-hindlimb-mocap

    # Download .rrd files from HuggingFace (used by CI)
    uv run python scripts/ingest.py pull \\
        --output data/rrd/
"""

import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_c3d_to_rrd(c3d_dir: str, output_dir: str, group_map_path: str | None = None) -> None:
    """Convert C3D files to .rrd using rerun-importer-c3d batch mode.

    This runs on the workstation and requires ezc3d + rerun-importer-c3d.
    """
    try:
        from rerun_importer_c3d.batch import batch_import, load_group_map
    except ImportError:
        raise ImportError(
            "rerun-importer-c3d is required for C3D ingestion. "
            "Install with: uv sync --extra ingest"
        )

    try:
        from rat_vml.analysis.subject_groups import SUBJECT_TO_GROUP
    except ImportError:
        SUBJECT_TO_GROUP = {}

    # Load or auto-generate group map
    group_map = load_group_map(group_map_path)
    if not group_map and SUBJECT_TO_GROUP:
        group_map = SUBJECT_TO_GROUP
        logger.info(f"Auto-generated group map with {len(group_map)} subjects")

    logger.info(f"Converting C3D files from {c3d_dir} → {output_dir}")
    results = batch_import(c3d_dir, output_dir, group_map=group_map)
    logger.info(f"Converted {len(results)} subjects to .rrd")


def push_rrd_to_hf(rrd_dir: str, repo_id: str, commit_message: str | None = None) -> None:
    """Push .rrd files to a HuggingFace dataset.

    Uploads all .rrd files in the directory to the repo's `rrd/` subdirectory.
    """
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

    msg = commit_message or f"Update .rrd catalog ({len(rrd_files)} subjects)"
    logger.info(f"Push complete: {len(rrd_files)} files uploaded to {repo_id}")


def pull_rrd_from_hf(output_dir: str, repo_id: str = "hudsonburke/rat-hindlimb-mocap") -> None:
    """Download .rrd files from HuggingFace dataset.

    Used by CI to get pre-built .rrd files without needing C3D conversion.
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

    # List .rrd files in the repo
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
        description="C3D → .rrd ingestion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Convert C3D to .rrd on workstation\n"
            "  python scripts/ingest.py convert --c3d-dir sourcedata/ -o data/rrd/\n"
            "\n"
            "  # Push .rrd files to HuggingFace\n"
            "  python scripts/ingest.py push --rrd-dir data/rrd/\n"
            "\n"
            "  # Download .rrd files from HuggingFace (CI or fresh clone)\n"
            "  python scripts/ingest.py pull -o data/rrd/\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    # convert
    convert_parser = subparsers.add_parser("convert", help="Convert C3D to .rrd (workstation only)")
    convert_parser.add_argument("--c3d-dir", required=True, help="C3D source directory")
    convert_parser.add_argument("-o", "--output", default="data/rrd", help="Output .rrd directory")
    convert_parser.add_argument("--group-map", default=None, help="JSON group map file")

    # push
    push_parser = subparsers.add_parser("push", help="Push .rrd files to HuggingFace")
    push_parser.add_argument("--rrd-dir", default="data/rrd", help=".rrd directory to upload")
    push_parser.add_argument("--repo", default="hudsonburke/rat-hindlimb-mocap", help="HF repo ID")
    push_parser.add_argument("--message", default=None, help="Commit message")

    # pull
    pull_parser = subparsers.add_parser("pull", help="Download .rrd files from HuggingFace")
    pull_parser.add_argument("-o", "--output", default="data/rrd", help="Output directory")
    pull_parser.add_argument("--repo", default="hudsonburke/rat-hindlimb-mocap", help="HF repo ID")

    args = parser.parse_args()

    if args.command == "convert":
        convert_c3d_to_rrd(args.c3d_dir, args.output, args.group_map)
    elif args.command == "push":
        push_rrd_to_hf(args.rrd_dir, args.repo, args.message)
    elif args.command == "pull":
        pull_rrd_from_hf(args.output, args.repo)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
