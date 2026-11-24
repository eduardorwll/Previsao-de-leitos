"""Módulo CLI reprodutível e logging estruturado em JSON (Feature F07)."""

from previsao_leitos.cli.logging_json import emit_json_log, format_json_log_record, resolve_git_sha
from previsao_leitos.cli.main import build_parser, cli_main

__all__ = [
    "build_parser",
    "cli_main",
    "emit_json_log",
    "format_json_log_record",
    "resolve_git_sha",
]
