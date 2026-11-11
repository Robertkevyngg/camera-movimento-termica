"""Modelos de configuração imutáveis e estruturas de telemetria do sistema."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MIN_WIDTH = 160
MAX_WIDTH = 3840
MIN_HEIGHT = 120
MAX_HEIGHT = 2160
MIN_FPS = 1
MAX_FPS = 60

RESOLUTION_PRESETS: dict[str, tuple[int, int]] = {
    "320x240": (320, 240),
    "640x480": (640, 480),
    "800x600": (800, 600),
    "960x540": (960, 540),
    "1280x720": (1280, 720),
    "1920x1080": (1920, 1080),
}

FPS_PRESETS: tuple[int, ...] = (5, 10, 15, 24, 30, 60)

THERMAL_PROFILE_PRESETS: dict[str, dict[str, float]] = {
    "human": {
        "min_temp_c": 20.0,
        "max_temp_c": 42.0,
        "overheat_threshold_c": 37.5,
        "kinetic_heat_gain": 0.55,
        "emissivity_contrast": 1.35,
    },
    "ambient": {
        "min_temp_c": 18.0,
        "max_temp_c": 65.0,
        "overheat_threshold_c": 48.0,
        "kinetic_heat_gain": 0.50,
        "emissivity_contrast": 1.25,
    },
    "industrial": {
        "min_temp_c": 22.0,
        "max_temp_c": 120.0,
        "overheat_threshold_c": 85.0,
        "kinetic_heat_gain": 0.45,
        "emissivity_contrast": 1.20,
    },
}

# Perfil para computadores modestos: menos pixels e menos quadros por segundo.
LITE_MODE_PATCH: dict[str, Any] = {
    "resolution_preset": "320x240",
    "target_fps": 15,
    "motion": {"show_trails": False},
}

VisionMode = Literal["normal", "thermal", "hybrid", "motion_mask"]
MotionAlgorithm = Literal["frame_diff", "mog2", "knn"]
ThermalPalette = Literal["inferno", "jet", "hot", "turbo", "plasma", "bone", "magma"]
ThermalProfile = Literal["human", "ambient", "industrial"]
SimulatorScenario = Literal["conveyor", "machinery", "perimeter", "boiler"]
CameraSource = Literal["simulator", "camera"]
AlertLevel = Literal["NORMAL", "ATENCAO", "CRITICO"]


class ROIConfig(BaseModel):
    """Região de Interesse (ROI) normalizada [0.0 .. 1.0]."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    x: float = Field(default=0.1, ge=0.0, lt=0.95)
    y: float = Field(default=0.1, ge=0.0, lt=0.95)
    w: float = Field(default=0.8, ge=0.05, le=1.0)
    h: float = Field(default=0.8, ge=0.05, le=1.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> ROIConfig:
        if self.x + self.w > 1.0001:
            raise ValueError("ROI horizontal (x + w) não pode ultrapassar 1.0")
        if self.y + self.h > 1.0001:
            raise ValueError("ROI vertical (y + h) não pode ultrapassar 1.0")
        return self

    def to_pixel_rect(self, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
        """Converte a ROI normalizada para coordenadas em pixels (px_x, px_y, px_w, px_h)."""
        px_x = max(0, min(frame_width - 1, round(self.x * frame_width)))
        px_y = max(0, min(frame_height - 1, round(self.y * frame_height)))
        px_w = max(1, min(frame_width - px_x, round(self.w * frame_width)))
        px_h = max(1, min(frame_height - px_y, round(self.h * frame_height)))
        return (px_x, px_y, px_w, px_h)


class MotionConfig(BaseModel):
    """Parâmetros imutáveis para detecção de movimento industrial."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    algorithm: MotionAlgorithm = "frame_diff"
    sensitivity: int = Field(default=65, ge=1, le=100)
    min_area: int = Field(default=250, ge=10, le=100000)
    blur_size: int = Field(default=5, ge=3, le=31)
    alarm_motion_pct: float = Field(default=3.5, ge=0.1, le=100.0)
    show_boxes: bool = True
    show_trails: bool = True

    @field_validator("blur_size")
    @classmethod
    def ensure_odd_blur(cls, value: int) -> int:
        return value if value % 2 == 1 else min(31, value + 1)


class ThermalConfig(BaseModel):
    """Parâmetros imutáveis para simulação de câmera térmica infravermelha."""

    model_config = ConfigDict(frozen=True)

    palette: ThermalPalette = "inferno"
    min_temp_c: float = Field(default=22.0, ge=-50.0, le=500.0)
    max_temp_c: float = Field(default=120.0, ge=-40.0, le=1200.0)
    overheat_threshold_c: float = Field(default=85.0, ge=-40.0, le=1200.0)
    kinetic_heat_gain: float = Field(default=0.45, ge=0.0, le=1.0)
    thermal_decay: float = Field(default=0.90, ge=0.50, le=0.99)
    emissivity_contrast: float = Field(default=1.20, ge=0.50, le=3.00)
    show_hotspots: bool = True
    show_isotherms: bool = True
    show_scale_bar: bool = True

    @model_validator(mode="after")
    def validate_temperature_range(self) -> ThermalConfig:
        if self.max_temp_c <= self.min_temp_c:
            raise ValueError("max_temp_c deve ser estritamente maior que min_temp_c")
        if not (self.min_temp_c <= self.overheat_threshold_c <= self.max_temp_c):
            raise ValueError(
                "overheat_threshold_c deve estar entre min_temp_c e max_temp_c"
            )
        return self


class SimulatorConfig(BaseModel):
    """Parâmetros do gerador sintético de cenários industriais."""

    model_config = ConfigDict(frozen=True)

    scenario: SimulatorScenario = "conveyor"
    speed_multiplier: float = Field(default=1.0, ge=0.1, le=5.0)
    noise_level: float = Field(default=2.0, ge=0.0, le=25.0)
    ambient_heat_factor: float = Field(default=0.30, ge=0.0, le=1.0)


class SystemConfig(BaseModel):
    """Configuração global imutável de captura da câmera pessoal, movimento e simulação térmica."""

    model_config = ConfigDict(frozen=True)

    source: CameraSource = "camera"
    camera_index: int = Field(default=0, ge=0, le=10)
    mirror: bool = True
    width: int = Field(default=640, ge=MIN_WIDTH, le=MAX_WIDTH)
    height: int = Field(default=480, ge=MIN_HEIGHT, le=MAX_HEIGHT)
    target_fps: int = Field(default=24, ge=MIN_FPS, le=MAX_FPS)
    vision_mode: VisionMode = "normal"
    hud_enabled: bool = True
    roi: ROIConfig = Field(default_factory=ROIConfig)
    motion: MotionConfig = Field(default_factory=MotionConfig)
    thermal: ThermalConfig = Field(default_factory=ThermalConfig)
    simulator: SimulatorConfig = Field(default_factory=SimulatorConfig)
    # Gravação em vídeo dos eventos de movimento (câmera física): começa
    # `record_pre_sec` antes do movimento e termina `record_post_sec` depois dele.
    record_events: bool = False
    record_pre_sec: float = Field(default=2.0, ge=0.0, le=10.0)
    record_post_sec: float = Field(default=3.0, ge=0.5, le=60.0)

    def with_resolution(self, width: int, height: int) -> SystemConfig:
        """Retorna uma nova instância de SystemConfig com a resolução informada."""
        return SystemConfig.model_validate(
            {**self.model_dump(), "width": width, "height": height}
        )

    def with_preset_resolution(self, preset: str) -> SystemConfig:
        """Retorna uma nova instância usando um preset ('640x480', '1280x720', etc.)."""
        if preset not in RESOLUTION_PRESETS:
            valid = ", ".join(RESOLUTION_PRESETS.keys())
            raise ValueError(f"Preset de resolução inválido: '{preset}'. Válidos: {valid}")
        width, height = RESOLUTION_PRESETS[preset]
        return self.with_resolution(width, height)

    def with_fps(self, fps: int) -> SystemConfig:
        """Retorna uma nova instância com o FPS alvo atualizado."""
        return SystemConfig.model_validate({**self.model_dump(), "target_fps": fps})

    def apply_patch(self, patch: dict[str, Any]) -> SystemConfig:
        """Aplica um dicionário parcial de alterações retornando uma nova configuração validada."""
        current = self.model_dump()
        merged: dict[str, Any] = {**current}

        for key, value in patch.items():
            if key in ("roi", "motion", "thermal", "simulator") and isinstance(value, dict):
                base_sub = merged[key] if isinstance(merged.get(key), dict) else current[key]
                sub_merged = {**base_sub, **value}
                if (
                    key == "thermal"
                    and "overheat_threshold_c" not in value
                    and ("min_temp_c" in value or "max_temp_c" in value)
                ):
                    min_c = sub_merged.get("min_temp_c")
                    max_c = sub_merged.get("max_temp_c")
                    alarm_c = sub_merged.get("overheat_threshold_c")
                    if (
                        isinstance(min_c, (int, float))
                        and isinstance(max_c, (int, float))
                        and isinstance(alarm_c, (int, float))
                        and min_c < max_c
                    ):
                        sub_merged["overheat_threshold_c"] = min(
                            float(max_c), max(float(min_c), float(alarm_c))
                        )
                merged[key] = sub_merged
            elif key == "thermal_profile":
                if not isinstance(value, str) or value not in THERMAL_PROFILE_PRESETS:
                    raise ValueError(f"Perfil térmico inválido: '{value}'")
                base_thermal = merged["thermal"] if isinstance(merged.get("thermal"), dict) else current["thermal"]
                merged["thermal"] = {**base_thermal, **THERMAL_PROFILE_PRESETS[value]}
            elif key == "resolution_preset":
                if not isinstance(value, str) or value not in RESOLUTION_PRESETS:
                    raise ValueError(f"Preset de resolução inválido: '{value}'")
                w, h = RESOLUTION_PRESETS[value]
                merged["width"] = w
                merged["height"] = h
            else:
                merged[key] = value

        return SystemConfig.model_validate(merged)


class BoundingBox(BaseModel):
    """Caixa delimitadora imutável de um objeto em movimento."""

    model_config = ConfigDict(frozen=True)

    x: int
    y: int
    w: int
    h: int
    area: float
    center_x: int
    center_y: int


class FrameTelemetry(BaseModel):
    """Métricas imutáveis extraídas a cada quadro processado."""

    model_config = ConfigDict(frozen=True)

    frame_index: int
    timestamp_iso: str
    source_active: CameraSource
    hardware_fallback: bool
    width: int
    height: int
    target_fps: int
    actual_fps: float
    processing_ms: float
    vision_mode: VisionMode
    motion_detected: bool
    motion_alarm: bool
    motion_percentage: float
    object_count: int
    bounding_boxes: tuple[BoundingBox, ...]
    min_temp_c: float
    avg_temp_c: float
    max_temp_c: float
    hotspot_x: int
    hotspot_y: int
    coldspot_x: int
    coldspot_y: int
    overheat_alarm: bool
    alert_level: AlertLevel
    camera_fps: float = 0.0  # quadros por segundo entregues pela câmera física
    recording: bool = False  # gravando vídeo de um evento agora
    recording_dropped: int = 0  # quadros que a gravação não conseguiu acompanhar


class EventRecord(BaseModel):
    """Registro imutável de eventos e alarmes industriais."""

    model_config = ConfigDict(frozen=True)

    id: int
    timestamp_iso: str
    category: Literal["MOTION", "THERMAL", "SYSTEM", "ANOMALY"]
    severity: AlertLevel
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
