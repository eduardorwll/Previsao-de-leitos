"""Permite `python -m previsao_leitos`, usado pelo executor paralelo por alvo."""

from __future__ import annotations

import sys

from previsao_leitos.cli.main import cli_main

if __name__ == "__main__":
    sys.exit(cli_main())
