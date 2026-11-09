"""Servidor FastAPI para controle local, telemetria e stream MJPEG do sistema."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, FastAPI, Query, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from app.camera_manager import CameraManager
from app.config import FPS_PRESETS, RESOLUTION_PRESETS, THERMAL_PROFILE_PRESETS
from app.industrial_sim import AnomalyType
from app.thermal_simulator import PALETTE_MAP

STATIC_DIR = Path(__file__).resolve().parent / "static"


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


def create_app(manager: CameraManager | None = None) -> FastAPI:
    """Cria e configura a aplicação FastAPI com todas as rotas SCADA."""
    cam_manager = manager or CameraManager()

    app = FastAPI(
        title="Sistema de Visão Industrial — Movimento & Termografia Simulada",
        version="1.0.0",
    )
    app.state.camera_manager = cam_manager

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_model=None)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
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
    async def get_presets() -> dict[str, Any]:
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
    async def get_config() -> dict[str, Any]:
        return _ok_envelope(cam_manager.get_config().model_dump())

    @app.patch("/api/config", response_model=None)
    async def patch_config(
        payload: Annotated[dict[str, Any] | None, Body()] = None,
    ) -> dict[str, Any] | JSONResponse:
        try:
            updated = cam_manager.update_config(payload or {})
            return _ok_envelope(updated.model_dump())
        except (ValidationError, ValueError) as exc:
            return _err_response(str(exc), status_code=422)

    @app.get("/api/telemetry")
    async def get_telemetry(step: bool = Query(default=False)) -> dict[str, Any]:
        if step:
            _, tel = cam_manager.step_frame()
        else:
            tel = cam_manager.get_latest_telemetry()
        return _ok_envelope(tel.model_dump())

    @app.post("/api/anomaly", response_model=None)
    async def post_anomaly(
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
    async def post_reset() -> dict[str, Any]:
        cam_manager.reset_detectors()
        return _ok_envelope({"status": "recalibrated"})

    @app.post("/api/camera/reconnect")
    async def post_camera_reconnect() -> dict[str, Any]:
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
    async def get_events(
        limit: int = Query(default=50, ge=1, le=200),
    ) -> dict[str, Any]:
        items = [ev.model_dump() for ev in cam_manager.get_events(limit=limit)]
        return _ok_envelope({"items": items, "total": len(items), "limit": limit})

    @app.delete("/api/events")
    async def delete_events() -> dict[str, Any]:
        cam_manager.clear_events()
        return _ok_envelope({"cleared": True})

    @app.get("/api/snapshot")
    async def get_snapshot() -> Response:
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
    async def video_stream(
        max_frames: int | None = Query(default=None, ge=1, le=10000),
    ) -> StreamingResponse:
        async def frame_generator():
            sent = 0
            while True:
                t0 = asyncio.get_running_loop().time()
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
                elapsed = asyncio.get_running_loop().time() - t0
                wait_sec = max(0.002, cam_manager.get_frame_interval_sec() - elapsed)
                await asyncio.sleep(wait_sec)

        return StreamingResponse(
            frame_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )

    return app


app = create_app()
