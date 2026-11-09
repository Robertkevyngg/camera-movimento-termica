#!/usr/bin/env python3
"""Ponto de entrada para execução local do Sistema de Visão Industrial."""

from __future__ import annotations

import sys

from app.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
