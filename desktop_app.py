#!/usr/bin/env python3
"""Ponto de entrada direto para o Aplicativo Desktop Local (SCADA Tkinter)."""

from __future__ import annotations

import sys

from app.cli import main


def launch_desktop(argv: list[str] | None = None) -> int:
    """Inicia diretamente o aplicativo desktop nativo passando '--app' por padrão."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not any(
        flag in args for flag in ("--app", "--local-app", "--desktop-gui", "--benchmark-frames")
    ):
        args.insert(0, "--app")
    return main(args)


if __name__ == "__main__":
    raise SystemExit(launch_desktop())
