"""Persistência local da configuração e do histórico de eventos.

A configuração é salva ao fechar e restaurada na próxima abertura, para que
ajustes como o Modo Leve não precisem ser refeitos. Os eventos vão para um
arquivo JSON Lines com tamanho limitado.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from pydantic import ValidationError

from app.config import EventRecord, SystemConfig

logger = logging.getLogger(__name__)

APP_DIR_NAME = "sistema-visao-industrial"
DATA_DIR_ENV = "VISAO_DATA_DIR"
CONFIG_FILE_NAME = "config.json"
EVENTS_FILE_NAME = "eventos.jsonl"
RECORDINGS_DIR_NAME = "gravacoes"
MAX_EVENT_LOG_BYTES = 1_000_000
MAX_CONFIG_FILE_BYTES = 100_000
PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600


def _write_private_atomic(path: Path, data: bytes) -> None:
    """Grava em arquivo temporário legível só pelo dono e o troca de uma vez."""
    path.parent.mkdir(mode=PRIVATE_DIR_MODE, parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, PRIVATE_FILE_MODE)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    os.replace(tmp_path, path)


def get_data_dir() -> Path:
    """Pasta de dados do usuário (ajustável pela variável VISAO_DATA_DIR)."""
    override = os.environ.get(DATA_DIR_ENV)
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / APP_DIR_NAME


def load_config(path: Path) -> SystemConfig | None:
    """Lê a configuração salva; devolve None se não existir ou estiver inválida."""
    try:
        if not path.is_file() or path.stat().st_size > MAX_CONFIG_FILE_BYTES:
            return None
        return SystemConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, ValidationError) as exc:
        # Arquivo corrompido ou de versão antiga: segue com os padrões.
        logger.warning("Configuração salva ignorada (%s): %s", path, exc)
        return None


def save_config(config: SystemConfig, path: Path) -> bool:
    """Grava a configuração de forma atômica; devolve False se o disco recusar."""
    try:
        _write_private_atomic(path, config.model_dump_json(indent=2).encode("utf-8"))
    except OSError as exc:
        logger.warning("Não foi possível salvar a configuração em %s: %s", path, exc)
        return False
    return True


class EventLog:
    """Acrescenta eventos a um arquivo JSON Lines, descartando a metade antiga ao encher."""

    def __init__(self, path: Path, max_bytes: int = MAX_EVENT_LOG_BYTES) -> None:
        self._path = path
        self._max_bytes = max_bytes
        self._failed = False

    def append(self, event: EventRecord) -> None:
        """Grava um evento; após a primeira falha de disco desiste em silêncio."""
        if self._failed:
            return
        try:
            self._path.parent.mkdir(mode=PRIVATE_DIR_MODE, parents=True, exist_ok=True)
            if self._path.exists() and self._path.stat().st_size > self._max_bytes:
                self._drop_older_half()
            fd = os.open(
                self._path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, PRIVATE_FILE_MODE
            )
            with os.fdopen(fd, "wb") as handle:
                handle.write(event.model_dump_json().encode("utf-8") + b"\n")
        except OSError as exc:
            self._failed = True
            logger.warning("Histórico de eventos desativado (%s): %s", self._path, exc)

    def _drop_older_half(self) -> None:
        # Divide só em "\n": o JSON escapa quebras de linha, mas não U+2028 e afins.
        lines = self._path.read_bytes().split(b"\n")
        kept = [line for line in lines[len(lines) // 2 :] if line]
        _write_private_atomic(self._path, b"\n".join(kept) + b"\n")
