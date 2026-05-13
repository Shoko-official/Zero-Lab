"""Command line entry point for Zero Lab."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from zero_lab import __version__
from zero_lab.core.config import RuntimeConfig, load_runtime_config
from zero_lab.core.logging import configure_logging
from zero_lab.core.random import seed_python


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zero-lab",
        description="Zero Lab research engine command line interface.",
    )
    parser.add_argument("--version", action="version", version=f"zero-lab {__version__}")

    subcommands = parser.add_subparsers(dest="command")

    smoke = subcommands.add_parser("smoke-test", help="Validate the local runtime foundation.")
    add_runtime_options(smoke)
    smoke.set_defaults(handler=run_smoke_test)

    show_config = subcommands.add_parser("show-config", help="Print the effective runtime config.")
    add_runtime_options(show_config)
    show_config.set_defaults(handler=run_show_config)

    return parser


def add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="Optional JSON or TOML runtime config path.")
    parser.add_argument("--run-dir", type=Path, help="Directory for runtime artifacts.")
    parser.add_argument("--seed", type=int, help="Runtime seed.")
    parser.add_argument("--log-level", help="Python logging level.")


def resolve_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    config = load_runtime_config(args.config)
    return config.with_overrides(
        run_dir=args.run_dir,
        seed=args.seed,
        log_level=args.log_level,
    )


def run_show_config(args: argparse.Namespace) -> int:
    config = resolve_runtime_config(args)
    print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
    return 0


def run_smoke_test(args: argparse.Namespace) -> int:
    config = resolve_runtime_config(args)
    logger = configure_logging(config)
    seed_report = seed_python(config.seed)

    payload = {
        "project": config.project_name,
        "run_dir": str(config.run_dir),
        "seed": config.seed,
        "seeded_modules": list(seed_report.seeded_modules),
        "status": "ok",
        "version": __version__,
    }

    logger.info("Smoke test completed for %s", config.project_name)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)

    if handler is None:
        parser.print_help()
        return 0

    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
