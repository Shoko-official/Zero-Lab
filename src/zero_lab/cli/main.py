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
from zero_lab.games import ChessGame, ConnectFourGame, GameRules, TicTacToeGame
from zero_lab.games.toy import TicTacToeState
from zero_lab.replay import append_episode, summarize_replay
from zero_lab.search import AlphaZeroSearch, MCTSSearchConfig
from zero_lab.search.alpha_zero import UniformEvaluator
from zero_lab.self_play import AlphaZeroSelfPlay, SelfPlayConfig


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

    list_games = subcommands.add_parser("list-games", help="List built-in game adapters.")
    list_games.set_defaults(handler=run_list_games)

    search_demo = subcommands.add_parser(
        "search-demo",
        help="Run a deterministic AlphaZero search smoke scenario.",
    )
    search_demo.add_argument("--simulations", type=int, default=32)
    search_demo.set_defaults(handler=run_search_demo)

    self_play_demo = subcommands.add_parser(
        "self-play-demo",
        help="Generate a deterministic Tic Tac Toe self-play episode.",
    )
    self_play_demo.add_argument("--simulations", type=int, default=16)
    self_play_demo.add_argument("--seed", type=int, default=0)
    self_play_demo.add_argument("--output", type=Path)
    self_play_demo.set_defaults(handler=run_self_play_demo)

    replay_summary = subcommands.add_parser("replay-summary", help="Summarize a JSONL replay file.")
    replay_summary.add_argument("input", type=Path)
    replay_summary.set_defaults(handler=run_replay_summary)

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


def run_list_games(_args: argparse.Namespace) -> int:
    games = list(builtin_games().values())
    payload = [
        {
            "action_size": game.action_size,
            "name": game.name,
        }
        for game in games
    ]
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run_search_demo(args: argparse.Namespace) -> int:
    state = TicTacToeState()
    for action in (0, 3, 1, 4):
        state = state.apply(action)

    result = AlphaZeroSearch(
        UniformEvaluator(),
        MCTSSearchConfig(simulations=args.simulations),
    ).run(state)
    payload = {
        "best_action": result.best_action,
        "game": "tic_tac_toe",
        "policy": {str(action): probability for action, probability in result.policy.items()},
        "simulations": args.simulations,
        "visit_counts": {str(action): visits for action, visits in result.visit_counts.items()},
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run_self_play_demo(args: argparse.Namespace) -> int:
    runner = AlphaZeroSelfPlay(
        UniformEvaluator(),
        MCTSSearchConfig(simulations=args.simulations),
        SelfPlayConfig(max_moves=9, temperature=0.0),
    )
    episode = runner.play(TicTacToeGame(), seed=args.seed)
    if args.output is not None:
        append_episode(args.output, episode)

    payload = {
        "game": episode.game,
        "length": episode.length,
        "outcome": episode.outcome,
        "output": None if args.output is None else str(args.output),
        "schema_version": episode.schema_version,
        "seed": args.seed,
        "simulations": args.simulations,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run_replay_summary(args: argparse.Namespace) -> int:
    print(json.dumps(summarize_replay(args.input).to_dict(), indent=2, sort_keys=True))
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
