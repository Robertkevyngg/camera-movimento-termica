"""Servidor FastAPI para controle local, telemetria e stream MJPEG do sistema."""

from __future__ import annotations

import hmac
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlsplit

from fastapi import Body, FastAPI, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from app.camera_manager import CameraManager
from app.config import FPS_PRESETS, RESOLUTION_PRESETS, THERMAL_PROFILE_PRESETS
from app.industrial_sim import AnomalyType
from app.thermal_simulator import PALETTE_MAP

STATIC_DIR = Path(__file__).resolve().parent / "static"
API_TOKEN_HEADER = "x-api-token"
API_TOKEN_QUERY = "token"
API_TOKEN_COOKIE = "visao_token"
# "testserver" é o Host usado pelo cliente de testes do Starlette.
LOCAL_HOSTNAMES = frozenset({"127.0.0.1", "localhost", "::1", "testserver"})


class AnomalyRequest(BaseModel):
    """Payload validado para injeção de anomalia industrial."""

    kind: AnomalyType = "thermal_spike"
    duration_frames: int = Field(default=45, ge=1, le=600)


def _ok_envelope(data: Any) -> dict[str, Any]:
    return {"success": True, "data": data, "error": None}


def _err_response(message: str, status_code: int = 422) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "data": None, "error": message},
    )


def _token_matches(candidate: str | None, expected: bytes) -> bool:
    """Comparação em tempo constante, para não vazar o token por temporização."""
    return bool(candidate) and hmac.compare_digest(
        candidate.encode("utf-8", "surrogateescape"), expected
    )


def _hostname_of(netloc: str) -> str:
    """Extrai o nome do host de 'host:porta' ou '[ipv6]:porta', em minúsculas."""
    netloc = netloc.strip().lower()
    if netloc.startswith("["):
        return netloc[1:].split("]", 1)[0]
    return netloc.rsplit(":", 1)[0] if ":" in netloc else netloc


def _is_cross_site(request: Request) -> bool:
    """True se o navegador indica que a requisição partiu de outro site."""
    if request.headers.get("sec-fetch-site", "").lower() == "cross-site":
        return True
    origin = request.headers.get("origin")
    if not origin:
        return False
    origin_netloc = urlsplit(origin).netloc.lower()
    return origin_netloc != request.headers.get("host", "").strip().lower()


def _install_request_guard(app: FastAPI, api_token: str | None) -> None:
    """Protege todas as rotas contra acesso indevido.

    - Requisições vindas de outro site são recusadas (CSRF): sem isso, qualquer
      página aberta no navegador poderia acionar o painel em 127.0.0.1.
    - Sem token (uso local), o cabeçalho Host precisa ser de loopback, o que
      barra DNS rebinding (domínio externo apontando para 127.0.0.1).
    - Com token, ele é exigido por cabeçalho, cookie ou ?token= na primeira visita.
    """
    expected = api_token.encode("utf-8", "surrogateescape") if api_token else None

    @app.middleware("http")
    async def guard_request(request: Request, call_next: Any) -> Response:
        if _is_cross_site(request):
            return _err_response("Requisição de outra origem recusada", status_code=403)
        if expected is None:
            if _hostname_of(request.headers.get("host", "")) not in LOCAL_HOSTNAMES:
                return _err_response("Host não permitido", status_code=403)
            return await call_next(request)

        query_ok = _token_matches(request.query_params.get(API_TOKEN_QUERY), expected)
        authorized = (
            query_ok
            or _token_matches(request.headers.get(API_TOKEN_HEADER), expected)
            or _token_matches(request.cookies.get(API_TOKEN_COOKIE), expected)
        )
        if not authorized:
            return _err_response("Token de acesso ausente ou inválido", status_code=401)
        response = await call_next(request)
        if query_ok:
            # O cookie dispensa o token na URL nas próximas requisições da página.
            response.set_cookie(API_TOKEN_COOKIE, api_token, httponly=True, samesite="strict")
        return response


def create_app(manager: CameraManager | None = None, api_token: str | None = None) -> FastAPI:
    """Cria e configura a aplicação FastAPI com todas as rotas SCADA.

    Com `api_token` definido, toda rota exige o token; sem ele só são aceitas
    requisições locais (Host de loopback) e nenhuma vinda de outro site.
    """
    cam_manager = manager or CameraManager()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            # Avisa os streams abertos antes de liberar a câmera, para não reabri-la.
            _app.state.shutting_down = True
            cam_manager.close()

    app = FastAPI(
        title="Sistema de Visão Industrial — Movimento & Termografia Simulada",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.camera_manager = cam_manager
    app.state.shutting_down = False
    _install_request_guard(app, api_token)
    # As rotas abaixo que tocam a câmera são síncronas de propósito: o FastAPI as
    # executa no pool de threads, e o processamento de vídeo não trava o laço de eventos.

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_model=None)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        cfg = cam_manager.get_config()
        return _ok_envelope(
            {
                "status": "online",
                "source": cfg.source,
                "camera_device": cam_manager.get_camera_device_label(cfg.camera_index),
                "resolution": f"{cfg.width}x{cfg.height}",
                "target_fps": cfg.target_fps,
            }
        )

    @app.get("/api/presets")
    def get_presets() -> dict[str, Any]:
        return _ok_envelope(
            {
                "cameras": cam_manager.list_available_cameras(),
                "resolutions": {
                    k: {"width": v[0], "height": v[1]}
                    for k, v in RESOLUTION_PRESETS.items()
                },
                "fps_presets": list(FPS_PRESETS),
                "thermal_palettes": list(PALETTE_MAP.keys()),
                "thermal_profiles": THERMAL_PROFILE_PRESETS,
                "vision_modes": ["normal", "thermal", "hybrid", "motion_mask"],
                "motion_algorithms": ["frame_diff", "mog2", "knn"],
                "scenarios": ["conveyor", "machinery", "perimeter", "boiler"],
            }
        )

    @app.get("/api/config")
    def get_config() -> dict[str, Any]:
        return _ok_envelope(cam_manager.get_config().model_dump())

    @app.patch("/api/config", response_model=None)
    def patch_config(
        payload: Annotated[dict[str, Any] | None, Body()] = None,
    ) -> dict[str, Any] | JSONResponse:
        try:
            updated = cam_manager.update_config(payload or {})
            return _ok_envelope(updated.model_dump())
        except (ValidationError, ValueError) as exc:
            return _err_response(str(exc), status_code=422)

    @app.get("/api/telemetry")
    def get_telemetry(step: bool = Query(default=False)) -> dict[str, Any]:
        if step:
            _, tel = cam_manager.step_frame()
        else:
            tel = cam_manager.get_latest_telemetry()
        return _ok_envelope(tel.model_dump())

    @app.post("/api/anomaly", response_model=None)
    def post_anomaly(
        payload: Annotated[dict[str, Any] | None, Body()] = None,
    ) -> dict[str, Any] | JSONResponse:
        try:
            req = AnomalyRequest.model_validate(payload or {})
            cam_manager.trigger_anomaly(req.kind, duration_frames=req.duration_frames)
            return _ok_envelope(
                {"kind": req.kind, "duration_frames": req.duration_frames}
            )
        except (ValidationError, ValueError) as exc:
            return _err_response(str(exc), status_code=422)

    @app.post("/api/reset")
    def post_reset() -> dict[str, Any]:
        cam_manager.reset_detectors()
        return _ok_envelope({"status": "recalibrated"})

    @app.post("/api/camera/reconnect")
    def post_camera_reconnect() -> dict[str, Any]:
        ok = cam_manager.reconnect_camera()
        cfg = cam_manager.get_config()
        return _ok_envelope(
            {
                "connected": ok,
                "camera_index": cfg.camera_index,
                "camera_device": cam_manager.get_camera_device_label(cfg.camera_index),
            }
        )

    @app.get("/api/events")
    def get_events(
        limit: int = Query(default=50, ge=1, le=200),
    ) -> dict[str, Any]:
        items = [ev.model_dump() for ev in cam_manager.get_events(limit=limit)]
        return _ok_envelope({"items": items, "total": len(items), "limit": limit})

    @app.delete("/api/events")
    def delete_events() -> dict[str, Any]:
        cam_manager.clear_events()
        return _ok_envelope({"cleared": True})

    @app.get("/api/snapshot")
    def get_snapshot() -> Response:
        jpeg_bytes = cam_manager.get_snapshot_jpeg(quality=90)
        return Response(
            content=jpeg_bytes,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": 'inline; filename="snapshot-industrial.jpg"',
                "Cache-Control": "no-store",
            },
        )

    @app.get("/api/stream")
    def video_stream(
        max_frames: int | None = Query(default=None, ge=1, le=10000),
    ) -> StreamingResponse:
        def frame_generator() -> Iterator[bytes]:
            sent = 0
            while not app.state.shutting_down:
                t0 = time.perf_counter()
                jpeg_bytes = cam_manager.get_stream_jpeg(quality=80)
                header = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg_bytes)}\r\n\r\n".encode("ascii")
                )
                yield header + jpeg_bytes + b"\r\n"
                sent += 1
                if max_frames is not None and sent >= max_frames:
                    break
                elapsed = time.perf_counter() - t0
                time.sleep(max(0.002, cam_manager.get_frame_interval_sec() - elapsed))

        return StreamingResponse(
            frame_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )

    return app


app = create_app()
