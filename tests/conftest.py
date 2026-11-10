"""Configuração compartilhada dos testes."""

import pytest


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Impede que os testes leiam ou gravem a configuração real do usuário."""
    data_dir = tmp_path / "dados-app"
    monkeypatch.setenv("VISAO_DATA_DIR", str(data_dir))
    return data_dir
