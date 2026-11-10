# Arquitetura do Sistema — Visão Industrial Local

## 1. Visão Geral
O sistema segue uma arquitetura modular em camadas com estado de configuração **imutável** (`Pydantic frozen models` / `frozen dataclasses`) e pipeline de processamento de quadros desacoplado da camada de transporte (Web API / CLI Desktop).

```
┌─────────────────────────────────────────────────────────────────┐
│                Interface Local (SCADA Web / CLI GUI)            │
│   Stream MJPEG  │  Controles REST  │  Telemetria & Snapshots    │
└────────┬──────────────────┬───────────────────────┬─────────────┘
         │                  ▼                       ▼
┌────────┴────────────────────────────────────────────────────────┐
│                   CameraManager (Orquestrador)                  │
│  - Gerencia fonte: Webcam Real (cv2.VideoCapture) ou Simulador  │
│  - Aplica Resolução (Hardware + Resize) e Pacing de FPS         │
│  - Mantém Telemetria e Log de Eventos Thread-Safe               │
└────────┬──────────────────┬───────────────────────┬─────────────┘
         ▼                  ▼                       ▼
┌─────────────────┐ ┌────────────────────┐ ┌──────────────────────┐
│ MotionDetector  │ │  ThermalSimulator  │ │    HudRenderer       │
│ - FrameDiff     │ │ - Emissividade IR  │ │ - Bounding Boxes     │
│ - MOG2 / KNN    │ │ - Calor Cinético   │ │ - Crosshair Hotspot  │
│ - Filtro ROI    │ │ - Paletas Térmicas │ │ - Barra Termográfica │
│ - BoundingBoxes │ │ - Hotspot/Coldspot │ │ - Telemetria SCADA   │
└─────────────────┘ └────────────────────┘ └──────────────────────┘
```

## 2. Módulos Principais
- `app/config.py`: Modelos imutáveis (`SystemConfig`, `MotionConfig`, `ThermalConfig`, `SimulatorConfig`, `ROI`, `FrameTelemetry`, `EventRecord`) e validação rigorosa de limites.
- `app/industrial_sim.py`: Gerador determinístico/animado de cenas industriais sintéticas (`conveyor`, `machinery`, `perimeter`, `boiler`) com suporte a injeção de anomalias térmicas e cinéticas.
- `app/motion_detector.py`: Pipeline de detecção de movimento com suporte a ROI, múltiplos algoritmos OpenCV (`frame_diff`, `mog2`, `knn`), filtragem morfológica e extração de contornos.
- `app/thermal_simulator.py`: Simulador termográfico que combina luminância calibrada, difusão térmica, acúmulo de calor por movimento (fricção/energia cinética), mapeamento linear para graus Celsius (`°C`), detecção de extremos térmicos (`hotspot` / `coldspot`) e aplicação de colormaps industriais.
- `app/hud_renderer.py`: Renderizador de sobreposição visual (HUD SCADA) com caixas delimitadoras, zona ROI, mira no ponto mais quente, barra de escala térmica em `°C` e indicadores de alarme.
- `app/camera_manager.py`: Controlador de captura e processamento com troca dinâmica de resolução/FPS, fallback automático de câmera física para simulador, histórico de eventos e captura de snapshots.
- `app/camera_capture.py`: Thread dedicada que abre a câmera e lê quadros continuamente; o `CameraManager` só consulta o quadro mais recente, sem bloquear a interface em leituras lentas ou em câmera ausente.
- `app/frame_utils.py`: Redução do quadro para a resolução de análise (largura máxima de 320 px) e cálculo de percentis por histograma, usados por movimento e termografia.
- `app/frame_pump.py`: Thread que executa `CameraManager.step_frame` no ritmo do FPS alvo; o app desktop apenas exibe o quadro pronto mais recente.
- `app/persistence.py`: Leitura/gravação atômica da configuração (`config.json`) e histórico de eventos em JSON Lines com tamanho limitado.
- `app/desktop_tabs.py` / `app/desktop_widgets.py`: Abas de controle, tema e geometria do vídeo, separados de `app/desktop_app.py`.
- `app/api.py`: Servidor FastAPI fornecendo endpoints RESTful, stream de vídeo MJPEG (`/api/stream`) e arquivos estáticos da interface SCADA.
