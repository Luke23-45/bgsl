"""
bgsl/data/upload_cli.py
-----------------------
``bgsl-upload`` console script entry point.

The producer side of the round-trip. Validates the local build, computes
a manifest, renders a dataset card, then ``HfApi.upload_folder`` to the
configured HF repo.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional, Sequence

from bgsl.data.acquisition import HfConfig, upload_to_hub


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bgsl-upload",
        description=(
            "Upload a local BGSL PhysioNet2019 build to a HuggingFace Hub "
            "repo (the producer counterpart to `bgsl-acquire`)."
        ),
    )
    p.add_argument(
        "--data-dir", required=True, type=str,
        help="Local BGSL data root (the one passed to bgsl-acquire).",
    )
    p.add_argument(
        "--repo-id", required=True, type=str,
        help="HuggingFace repo id, e.g. your-org/bgsl-physionet2019.",
    )
    p.add_argument(
        "--repo-type", default="dataset", choices=["dataset", "model"],
        help="HF repo type (default: dataset).",
    )
    p.add_argument(
        "--revision", default="main",
        help="HF revision (branch, tag, or commit SHA). Default: main.",
    )
    p.add_argument(
        "--private", action="store_true",
        help="Create the repo as private (idempotent).",
    )
    p.add_argument(
        "--message", "-m", default="Update BGSL PhysioNet2019 LMDB",
        help="Commit message for the upload.",
    )
    p.add_argument(
        "--token-env", default="HF_TOKEN",
        help="Environment variable carrying the HF token. Default: HF_TOKEN.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Compute manifest + card locally; do not touch the network.",
    )
    p.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase logging verbosity (-v INFO, -vv DEBUG).",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    level = logging.WARNING
    if args.verbose >= 2:
        level = logging.DEBUG
    elif args.verbose == 1:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    hf = HfConfig(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        token_env=args.token_env,
    )

    try:
        info = upload_to_hub(
            data_dir=args.data_dir,
            hf=hf,
            private=args.private,
            commit_message=args.message,
            dry_run=args.dry_run,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ModuleNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"dry-run ok: {info['file_count']} files, "
              f"{info['total_bytes']:,} bytes; repo={info['repo_id']} @ {info['revision']}")
    else:
        print(f"uploaded: {info['url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
