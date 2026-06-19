# Development Setup

Zero Lab uses Python 3.12 for local development and GitHub Actions CI.

## Environment

On Windows, prefer the Python launcher so commands do not accidentally use another installed
interpreter:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Linux or macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The editable install provides the `zero-lab` CLI and installs development tools used by CI.

## Review Gates

Run these commands before opening a pull request:

```bash
python -m pytest
python -m ruff check .
python -m mypy src tests
```

If the default `python` command points to an unsupported interpreter, use the explicit launcher:

```powershell
py -3.12 -m pytest
py -3.12 -m ruff check .
py -3.12 -m mypy src tests
```

## Generated Files

Do not commit virtual environments, caches, build artifacts, run directories, or local replay output.
The repository `.gitignore` already excludes common Python-generated files.
