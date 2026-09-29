"""CLI entry point for running a single experiment."""

from __future__ import annotations

import argparse
import subprocess
import sys

from dotenv import load_dotenv

from scripts.config import load_config
from scripts.runner import run_experiment

load_dotenv()


def _check_clean_worktree() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
        )
        return not result.stdout.strip()
    except Exception:
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bibliographic parsing experiment.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow running with uncommitted changes (not for official experiments).",
    )
    args = parser.parse_args()

    if not args.allow_dirty and not _check_clean_worktree():
        print(
            "error: working tree is not clean. Commit or stash changes before running.\n"
            "  Use --allow-dirty for development/testing only.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    config = load_config(args.config)
    success = run_experiment(config)
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
