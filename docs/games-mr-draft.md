# Merge Request: Add Base Game Adapters (Chess, Tic-Tac-Toe, Connect Four)

## Summary
This merge request implements the core game adapter architecture and provides initial implementations for Chess and two toy games (Tic-Tac-Toe, Connect Four).

It establishes the `BaseGame` abstract interface which ensures that all games provide consistent observations, legal move generation, and state transitions required for AlphaZero and MuZero training.

## Changes
- **Core Architecture**: Added `src/zero_lab/games/base.py` defining the `BaseGame` interface.
- **Chess Adapter**: Added `src/zero_lab/games/chess.py` using `python-chess` for robust move validation and SAN/UCI support.
- **Toy Games**: Added `src/zero_lab/games/toy/` with implementations for Tic-Tac-Toe and Connect Four for fast debugging and smoke testing.
- **CLI Integration**: Updated the main CLI to include `list-games` and improved smoke-test capabilities.
- **Testing**: Added comprehensive unit tests in `tests/` for all game logic and state transitions.

## Motivation
A standardized game interface is the prerequisite for the Monte Carlo Tree Search (MCTS) implementation. By supporting Chess as a primary environment from the start, we ensure the engine is built for complex action spaces while toy games allow for rapid iteration.

## Verification
- Ran `python -m pytest`: All 39 tests passed.
    - `tests/test_chess.py` (Pass)
    - `tests/test_tic_tac_toe.py` (Pass)
    - `tests/test_connect_four.py` (Pass)
- Verified CLI output: `zero-lab list-games` correctly identifies all adapters.

## Follow-up
- Add AlphaZero model contracts (separate MR).
- Implement the MCTS core.
