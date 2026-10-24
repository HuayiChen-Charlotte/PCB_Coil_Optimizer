"""Command-line interface for the math-structured coil optimizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import load_config, run
from .models import default_config


DEFAULT_CONFIG = Path("optimizer_config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="JSON config path")
    parser.add_argument(
        "--write-example-config",
        action="store_true",
        help="Write an example config to --config and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_example_config:
        args.config.write_text(json.dumps(default_config(), indent=2), encoding="utf-8")
        print(f"Wrote example config to {args.config}")
        return
    config = load_config(args.config)
    summary = run(config)
    print(
        json.dumps(
            {
                "case_name": summary["case_name"],
                "valid_loaded_candidate_count": summary["counts"]["valid_loaded_candidate_count"],
                "top_loaded_efficiency": summary["top_loaded_efficiency_candidate"]["loaded_efficiency"],
                "top_eta_max": summary["top_eta_max_candidate"]["eta_max"],
                "output_dir": str(Path(summary["paths"]["summary_json"]).parent),
                "checks": summary["checks"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
