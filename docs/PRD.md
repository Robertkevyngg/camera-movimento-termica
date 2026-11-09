# PRD — Sistema de Visão Industrial: Detecção de Movimento e Câmera Térmica Simulada

## 1. Objetivo do Produto
Desenvolver um sistema local de visão computacional voltado para **testes e simulações industriais**, capaz de:
1. Capturar vídeo em tempo real de uma câmera física (USB/V4L2 via OpenCV) ou de um **Simulador Industrial Sintético** integrado.
2. Permitir alteração dinâmica de **Resolução** e **FPS ( quadros por segundo )** em tempo real.
3. Detectar movimento com múltiplos algoritmos de subtração de fundo, filtro de área mínima, sensibilidade ajustável e Região de Interesse (ROI / Zona de Segurança).
4. Operar uma **Câmera Térmica Simulada (Termografia Infravermelha)** com múltiplas paletas térmicas (Ironbow/Inferno, Jet, Hot, Turbo, Plasma, White-Hot), acúmulo de calor cinético em regiões de movimento, mapeamento de temperatura em `°C`, detecção automática de *Hotspot* (ponto mais quente) e alarme de superaquecimento.

## 2. Público-Alvo e Casos de Uso
- **Engenheiros de Automação e Testes Industriais**: validação de algoritmos de monitoramento de linhas de produção, subestações, caldeiras e zonas restritas (EPI / intrusão).
- **Simulação de Planta Industrial**: execução 100% local (sem dependência de nuvem), funcionando tanto com webcam real quanto com cenários industriais sintéticos e injeção de anomalias sob demanda.

## 3. Requisitos Funcionais
- **RF01 — Controle de Resolução e FPS**:
  - Presets de resolução (`320x240`, `640x480`, `800x600`, `960x540`, `1280x720`, `1920x1080`) e resolução customizada.
  - Controle de FPS alvo (`1` a `60` FPS) com medição de FPS real e latência de processamento em `ms`.
- **RF02 — Detecção de Movimento Industrial**:
  - Algoritmos selecionáveis: `frame_diff` (Diferença Adaptativa), `mog2` (Mistura Gaussiana) e `knn`.
  - Sensibilidade (`1-100`), área mínima de contorno (`px²`), limiar de alarme (`%` de área ativa) e Região de Interesse (ROI).
  - Rastreamento de bounding boxes, centroides, contagem de objetos e percentual de movimentação.
- **RF03 — Câmera Térmica Simulada**:
  - Modos de visão: `normal` (RGB), `thermal` (Térmica Simulada), `hybrid` (Térmica + Contornos de Movimento + Isotermas) e `motion_mask` (Máscara Binária/Heatmap de Movimento).
  - Paletas térmicas: `inferno`, `jet`, `hot`, `turbo`, `plasma`, `bone`, `magma`.
  - Simulação de calor cinético/fricção (objetos em movimento geram assinatura térmica superior).
  - Escala de temperatura configurável (`°C`), localização de *Hotspot* (`Máx °C`) e *Coldspot* (`Mín °C`), barra de legenda térmica no HUD e alarme de superaquecimento.
- **RF04 — Simulador de Cenários Industriais e Injeção de Falhas**:
  - Cenários: `conveyor` (Esteira de Produção), `machinery` (Motores Rotativos), `perimeter` (Zona de Segurança/Intrusão) e `boiler` (Caldeira e Tubulação).
  - Injeção manual de anomalias: Pico Térmico (Superaquecimento) e Intrusão/Movimento Brusco.
- **RF05 — Interface SCADA Local & Modo Janela Desktop**:
  - Painel Web SCADA/HMI em tempo real (`FastAPI` + stream MJPEG + telemetria + log de eventos + captura de snapshot).
  - Modo de execução CLI/Desktop OpenCV opcional.
