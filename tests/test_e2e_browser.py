"""Testes E2E reais de navegador (Headless Chrome via CDP) para o painel Web SCADA."""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import websockets

CHROME_BIN = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_BIN = str(PROJECT_ROOT / ".venv" / "bin" / "python")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _cdp_eval(ws, msg_id: int, expression: str):
    await ws.send(
        json.dumps(
            {
                "id": msg_id,
                "method": "Runtime.evaluate",
                "params": {
                    "expression": expression,
                    "returnByValue": True,
                    "awaitPromise": True,
                },
            }
        )
    )
    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
        msg = json.loads(raw)
        if msg.get("id") == msg_id:
            return msg["result"]["result"].get("value")


@pytest.mark.skipif(CHROME_BIN is None, reason="Navegador Chrome/Chromium não disponível")
def test_e2e_scada_dashboard_in_headless_chrome():
    app_port = _free_port()
    cdp_port = _free_port()

    server = subprocess.Popen(
        [PYTHON_BIN, "main.py", "--host", "127.0.0.1", "--port", str(app_port)],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    chrome = None
    try:
        for _ in range(40):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{app_port}/api/health", timeout=0.5)
                break
            except (urllib.error.URLError, OSError):
                time.sleep(0.1)

        assert CHROME_BIN is not None
        chrome = subprocess.Popen(
            [
                CHROME_BIN,
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                f"--remote-debugging-port={cdp_port}",
                f"http://127.0.0.1:{app_port}/",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        ws_url = None
        for _ in range(50):
            with contextlib.suppress(urllib.error.URLError, OSError, ValueError):
                pages = json.loads(
                    urllib.request.urlopen(f"http://127.0.0.1:{cdp_port}/json", timeout=0.5).read()
                )
                for page in pages:
                    if page.get("type") == "page" and f":{app_port}" in page.get("url", ""):
                        ws_url = page["webSocketDebuggerUrl"]
                        break
                if ws_url:
                    break
            time.sleep(0.1)

        assert ws_url is not None, "Não foi possível conectar ao WebSocket CDP do Chrome"

        async def run_browser_interactions():
            async with websockets.connect(ws_url) as ws:
                await ws.send(json.dumps({"id": 1, "method": "Runtime.enable"}))
                await asyncio.sleep(0.8)

                # 1. Validar carregamento inicial do stream MJPEG e KPIs
                initial = await _cdp_eval(
                    ws,
                    10,
                    """
                    (() => {
                        const img = document.getElementById("video-stream");
                        return {
                            naturalWidth: img.naturalWidth,
                            naturalHeight: img.naturalHeight,
                            kpiRes: document.getElementById("kpi-resolution").textContent,
                        };
                    })()
                    """,
                )
                assert initial["naturalWidth"] == 640
                assert initial["naturalHeight"] == 480
                assert initial["kpiRes"] == "640 x 480"

                # 2. Mudar resolução para 1280x720, FPS para 30 e modo para câmera térmica
                updated = await _cdp_eval(
                    ws,
                    11,
                    """
                    (async () => {
                        const sel = document.getElementById("ctrl-res-preset");
                        sel.value = "1280x720";
                        sel.dispatchEvent(new Event("change"));

                        document.querySelector('.chip[data-fps="30"]').click();
                        document.querySelector('.mode-tab[data-mode="thermal"]').click();

                        await new Promise(r => setTimeout(r, 450));
                        const img = document.getElementById("video-stream");
                        return {
                            kpiRes: document.getElementById("kpi-resolution").textContent,
                            lblFps: document.getElementById("lbl-fps").textContent,
                            thermalActive: document.querySelector('.mode-tab[data-mode="thermal"]').classList.contains("active"),
                            naturalWidth: img.naturalWidth,
                            naturalHeight: img.naturalHeight,
                        };
                    })()
                    """,
                )
                assert updated["kpiRes"] == "1280 x 720"
                assert updated["lblFps"] == "30 FPS"
                assert updated["thermalActive"] is True
                assert updated["naturalWidth"] == 1280
                assert updated["naturalHeight"] == 720

                # 3. Ajustar escala térmica (reduzir Temp. Máx para 70°C abaixo do alarme padrão 85°C)
                temp_scale = await _cdp_eval(
                    ws,
                    12,
                    """
                    (async () => {
                        const tmax = document.getElementById("ctrl-temp-max");
                        tmax.value = "70";
                        tmax.dispatchEvent(new Event("change"));
                        await new Promise(r => setTimeout(r, 150));
                        const cfg = await fetch("/api/config").then(r => r.json());
                        return {
                            domMax: tmax.value,
                            domAlarm: document.getElementById("ctrl-temp-alarm").value,
                            srvMax: cfg.data.thermal.max_temp_c,
                            srvAlarm: cfg.data.thermal.overheat_threshold_c,
                        };
                    })()
                    """,
                )
                assert temp_scale["domMax"] == "70"
                assert temp_scale["domAlarm"] == "70"
                assert temp_scale["srvMax"] == 70
                assert temp_scale["srvAlarm"] == 70

                # 4. Injetar anomalia de pico térmico e verificar alarme crítico na UI
                anom_state = await _cdp_eval(
                    ws,
                    13,
                    """
                    (async () => {
                        document.querySelector('[data-anomaly="thermal_spike"]').click();
                        await new Promise(r => setTimeout(r, 200));
                        return {
                            badgeAlert: document.getElementById("badge-alert").textContent,
                            eventsHtml: document.getElementById("events-list").innerHTML,
                        };
                    })()
                    """,
                )
                assert "CRITICO" in anom_state["badgeAlert"]
                assert "ANOMALY" in anom_state["eventsHtml"]

        asyncio.run(run_browser_interactions())
    finally:
        if chrome is not None:
            chrome.terminate()
            try:
                chrome.wait(timeout=3)
            except subprocess.TimeoutExpired:
                chrome.kill()
        server.terminate()
        try:
            server.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server.kill()
