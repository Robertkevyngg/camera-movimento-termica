# Task List — Implementação TDD

- [x] **Fase 1**: Planejamento (`PRD.md`, `ARCHITECTURE.md`, `TASK_LIST.md`) e configuração do ambiente Python 3.12 (`uv`).
- [x] **Fase 2**: Testes e implementação de `app/config.py` (Modelos imutáveis, presets de resolução/FPS, validação de limites).
- [x] **Fase 3**: Testes e implementação de `app/industrial_sim.py` (Gerador de cenas industriais sintéticas e injeção de falhas).
- [x] **Fase 4**: Testes e implementação de `app/motion_detector.py` (Detecção de movimento, algoritmos `frame_diff`, `mog2`, `knn`, filtro ROI).
- [x] **Fase 5**: Testes e implementação de `app/thermal_simulator.py` (Câmera térmica simulada, paletas IR, calor cinético, telemetria em `°C`, hotspot/coldspot).
- [x] **Fase 6**: Testes e implementação de `app/hud_renderer.py` (HUD industrial, escala termográfica, mira de hotspot, ROI, alertas).
- [x] **Fase 7**: Testes e implementação de `app/camera_manager.py` (Gerenciamento de câmera real + simulador, mudança dinâmica de resolução e FPS, eventos e snapshots).
- [x] **Fase 8**: Testes e implementação de `app/api.py` + Interface SCADA (`app/static/*`) + CLI (`app/cli.py`, `main.py`).
- [x] **Fase 9**: Execução da suíte completa de testes (`pytest --cov`), verificação de casos de borda e validação end-to-end do servidor local.
- [x] **Fase 10**: Implementação do Aplicativo Desktop Nativo completo (`app/desktop_app.py`, `desktop_app.py`, `run_app.sh`, `SistemaVisaoIndustrial.desktop`) com interface gráfica Tkinter SCADA, seleção de ROI via mouse, controles completos de resolução/FPS/termografia/movimento e suíte de testes (`tests/test_desktop_app.py`).
- [x] **Fase 11**: Tempo real em 1080p a 60 FPS: câmera pedida a 60 FPS, quadros JPEG direto da câmera e relógio do driver (Linux), FPS fixo e gravação em vídeo dos eventos de movimento sem perda de quadros (`app/event_recorder.py`, `tests/test_event_recorder.py`).
