"""
bgsl/data/acquisition_cli.py
----------------------------
``bgsl-acquire`` console script entry point.

A thin argparse wrapper over :func:`bgsl.data.acquisition.ensure_dataset`.
Ships zero Lightning dependencies so it works in any environment that can
import ``bgsl`` (no GPU, no torch, etc.).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional, Sequence

from bgsl.data.acquisition import (
    AcquisitionConfig,
    AcquisitionResult,
    plan,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bgsl-acquire",
        description=(
            "Acquire the BGSL PhysioNet2019 LMDB via a 3-tier chain "
            "(local -> HuggingFace -> build-from-raw)."
        ),
    )
    p.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to a YAML acquisition config (see experiments/configs/data_acquisition.yaml).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-acquire even if a valid marker is present (also: env BGSL_FORCE_REACQUIRE=1).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the planned tier chain and exit. Never touches disk or network.",
    )
    p.add_argument(
        "--tier", choices=["1", "2", "3", "all"], default="all",
        help="Restrict the chain (debug aid). Default: all.",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Emit a single-line JSON result on stdout (machine-friendly).",
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

    try:
        cfg = AcquisitionConfig.from_yaml(args.config)
    except FileNotFoundError:
        print(f"error: config file not found: {args.config}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"error: failed to parse {args.config}: {exc}", file=sys.stderr)
        return 2

    if args.force:
        cfg.force = True

    # ---- Dry-run: just print the plan --------------------------------
    if args.dry_run:
        for step in plan(cfg):
            print(json.dumps(step) if args.json else
                  f"[tier {step['tier']}] {step['name']}: {step['action']}")
        return 0

    # ---- Optional tier gating (debug) --------------------------------
    if args.tier != "all":
        tier = int(args.tier)
        if tier == 1:
            cfg.hf = None
            cfg.build = None
        elif tier == 2:
            cfg.build = None
        elif tier == 3:
            cfg.hf = None

    result: AcquisitionResult = ensure_dataset_with_logging(cfg)

    if args.json:
        print(json.dumps(_result_to_dict(result), sort_keys=True))
    else:
        _print_result_human(result)

    return 0 if result.ok else 1


# ---------------------------------------------------------------------------
# Helpers (kept here so acquisition.py stays CLI-free)
# ---------------------------------------------------------------------------


def ensure_dataset_with_logging(
    cfg: AcquisitionConfig,
) -> AcquisitionResult:
    """Run :func:`ensure_dataset` and forward progress to the ``bgsl.acquisition`` logger."""
    from bgsl.data.acquisition import ensure_dataset  # local import keeps CLI snappy
    return ensure_dataset(cfg)


def _result_to_dict(result: AcquisitionResult) -> dict:
    return {
        "ok": result.ok,
        "source": result.source,
        "data_dir": str(result.data_dir),
        "elapsed_s": round(result.elapsed_s, 3),
        "bytes_acquired": result.bytes_acquired,
        "tier_attempts": result.tier_attempts,
        "error": result.error,
        "marker": result.marker,
    }


def _print_result_human(result: AcquisitionResult) -> None:
    if result.ok:
        print(f"ok: data ready at {result.data_dir} (source={result.source}, "
              f"{result.bytes_acquired:,} bytes, {result.elapsed_s:.2f}s)")
    else:
        print(f"failed: could not acquire data (tiers tried: "
              f"{', '.join(result.tier_attempts)}). error={result.error}",
              file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
