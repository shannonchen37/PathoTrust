"""Command-line interface for PathoTrust."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pathodataforge.core.pipeline import run_pipeline
from pathodataforge.utils.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the PathoTrust pathology processing pipeline.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to a YAML config file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(Path(args.config))
        summary = run_pipeline(config)
    except Exception as exc:
        print(f"PathoTrust failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
