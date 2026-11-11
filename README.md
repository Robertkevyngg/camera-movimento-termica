# 📷 Câmera Pessoal — Análise de Movimento & Câmera Térmica Simulada

Sistema de visão computacional em Python projetado para operar **localmente** utilizando a sua **câmera pessoal (webcam USB/integrada como `/dev/video0`)** para captar e analisar movimentos em tempo real, permitir a troca dinâmica de resolução e taxa de quadros (FPS), e simular o fluxo da sua própria câmera como se fosse uma **câmera térmica infravermelha profissional (FLIR/Termografia)**.

---

## 🚀 Principais Funcionalidades

1. **Captura Direta da Câmera Pessoal (`source="camera"`)**:
   - Inicia instantaneamente com a sua webcam local (padrão `/dev/video0`), com detecção automática de câmeras e reconexão amigável.
   - Opção de **espelhamento horizontal (`mirror=True`)** para visualização natural no estilo espelho.
   - Fallback de segurança elegante caso nenhuma câmera física esteja conectada.

2. **Detecção e Rastreamento de Movimentos em Tempo Real**:
   - Algoritmos selecionáveis:
     - `frame_diff`: Diferença adaptativa de quadros com compensação de iluminação ambiente.
     - `mog2`: Mistura Gaussiana de fundo com adaptação dinâmica.
     - `knn`: K-Nearest Neighbors para separação de primeiro plano e fundo.
   - Ajuste em tempo real de **sensibilidade (1% a 100%)**, **área mínima do objeto (px²)** e **limiar de alarme**.
   - Caixas delimitadoras (*Bounding Boxes*), vetor e rastro de trajetória do movimento.
   - **Região de Interesse (ROI)** interativa: desenhe a área monitorada diretamente na tela clicando e arrastando o mouse sobre o vídeo.

3. **Simulador de Câmera Térmica Infravermelha (IR)**:
   - Converte o vídeo real da sua webcam em termografia com **7 paletas termográficas**:
     - `inferno` (FLIR Ironbow / padrão industrial)
     - `jet` (escala arco-íris clássica)
     - `turbo` (arco-íris aprimorado pelo Google)
     - `hot` (gradiente preto-vermelho-amarelo-branco)
     - `plasma`
     - `bone` (White-Hot estilo visão noturna militar)
     - `magma`
   - **Perfis Rápidos de Temperatura**:
     - *Corporal / Pessoal* (`20°C a 42°C`): otimizado para temperatura da pele humana.
     - *Ambiente* (`18°C a 65°C`): para objetos e movimentação em salas e escritórios.
     - *Alta Temperatura* (`22°C a 120°C`): simulações com superfícies quentes.
   - **Simulação de Calor Cinético por Movimento**: áreas que se movimentam na frente da câmera geram calor cinético residual simulando fricção ou atividade física com taxa de decaimento térmico ajustável.
   - Rastreamento automático de **Hotspot** (ponto mais quente) e **Coldspot** (ponto mais frio) com coordenadas e temperatura em °C.
   - Isoterma crítica e barra de escala termográfica graduada.

4. **Controle Dinâmico de Resolução e FPS**:
   - Alterne instantaneamente entre resoluções pré-definidas (`320x240`, `640x480`, `800x600`, `960x540`, `1280x720`, `1920x1080`) ou defina uma resolução personalizada.
   - Controle dinâmico de taxa de quadros de **1 a 60 FPS** com sincronização temporal precisa.
   - Acima de 30 FPS a câmera é pedida a **60 FPS de verdade** (ex.: Logitech StreamCam/Brio em 1080p60).

6. **Gravação em Vídeo dos Movimentos (tempo real, até 1080p a 60 FPS)**:
   - Ligue com `--record`, ou na opção *Gravar Vídeo dos Movimentos* do app desktop e do painel web.
   - Cada evento de movimento vira um vídeo que começa **2 s antes** do movimento e termina **3 s depois**
     (`record_pre_sec` / `record_post_sec`). A thread de captura entrega **todos** os quadros à gravação,
     mesmo os que a análise não chegou a usar: o vídeo mostra o movimento real, quadro a quadro.
   - **No Linux**, os quadros JPEG que a câmera já envia (MJPG) são gravados direto num `.avi`, **sem
     recomprimir**: custo de CPU quase zero e a qualidade original da câmera. Nos demais casos, `.mp4`.
   - Junto de cada vídeo vai `<vídeo>_tempos.csv` com o horário de cada quadro em milissegundos; no Linux
     o horário vem do próprio driver da câmera, verdadeiro mesmo se o computador atrasar.
   - No Linux a câmera recebe **FPS fixo** (não cai para 30 FPS com pouca luz); o ajuste é desfeito ao sair.
   - Os vídeos ficam na subpasta `gravacoes` da pasta de dados (ou em `--recordings-dir`). A telemetria
     mostra o FPS real da câmera, `● REC` durante a gravação e quadros perdidos, se houver.

7. **4 Modos de Visão**:
   - `normal`: Vídeo RGB original com caixas e rastreamento de movimento.
   - `thermal`: Visão termográfica simulada completa.
   - `hybrid`: Fusão termo-óptica (imagem real translúcida sobreposta ao mapa térmico com contornos destacados).
   - `motion_mask`: Máscara binária de primeiro plano isolando o que está em movimento.

---

## 📦 Instalação e Requisitos

### Pré-requisitos
- **Python 3.10 ou superior**
- Sistema Linux, macOS ou Windows
- Webcam USB ou câmera integrada
- Pacote Tkinter (no Ubuntu/Debian: `sudo apt install python3-tk`)

### Clonando o Repositório
```bash
git clone https://github.com/Robertkevyngg/camera-movimento-termica.git
cd camera-movimento-termica
```

### Criando o Ambiente Virtual e Instalando Dependências
Com `uv` (recomendado):
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

Ou com `pip` convencional:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🖥️ Como Executar

### 1. Aplicativo Desktop Nativo (Recomendado)
Para abrir diretamente o aplicativo com interface gráfica nativa:

```bash
./run_app.sh
```

Ou diretamente pelo Python:
```bash
python desktop_app.py
```

Opções de inicialização personalizada:
```bash
# Iniciar em HD 720p a 30 FPS no modo Câmera Térmica:
python desktop_app.py --resolution 1280x720 --fps 30 --mode thermal

# Iniciar no modo Híbrido com paleta Turbo:
python desktop_app.py --mode hybrid --palette turbo

# Computador modesto: modo leve (320x240 @ 15 FPS, sem trilhas de movimento)
python desktop_app.py --lite
```

#### Desempenho em computadores modestos
- **Modo Leve**: use `--lite` ou o botão **Modo Leve (PC fraco)** abaixo do vídeo.
- A câmera é lida em uma thread separada, então a janela não congela se a webcam
  estiver lenta, em pouca luz ou desconectada (aparece "Conectando a camera..." e,
  após 2 s sem imagem, o simulador de fallback).
- Movimento e termografia são analisados em no máximo 320 px de largura; o vídeo
  exibido continua na resolução escolhida.
- Os painéis de telemetria atualizam 4x por segundo e a lista de eventos 2x por
  segundo, em vez de a cada quadro.
- Captura, análise e composição do vídeo rodam em uma thread própria; a janela só
  exibe o quadro pronto e continua respondendo mesmo se um quadro demorar.
- No Modo Leve (alvo de até 15 FPS) a webcam também é aberta a 15 FPS.
- Com **Pillow** instalado o vídeo é desenhado cerca de 4x mais rápido; sem ele o
  app funciona pelo caminho PPM, mais lento.

#### Configuração e histórico salvos
Ao fechar, a configuração atual (resolução, FPS, modo, paleta, ROI...) é salva e
restaurada na próxima abertura; opções digitadas na linha de comando têm prioridade.
Os eventos são gravados em `eventos.jsonl` (limitado a ~1 MB).

- Linux: `~/.config/sistema-visao-industrial/`
- Windows: `%APPDATA%\sistema-visao-industrial\`
- Outra pasta: variável `VISAO_DATA_DIR`. Para não ler nem gravar nada: `--no-persist`.

No Windows, use `run_app.bat` no lugar de `run_app.sh` (suporte ainda não testado
em uma máquina Windows).

#### Atalhos do Teclado no App Desktop:
- `1` : Modo Câmera Pessoal (Normal)
- `2` : Modo Câmera Térmica Simulada
- `3` : Modo Fusão Híbrida
- `4` : Modo Máscara de Movimento
- `M` : Alternar Espelhamento da Webcam
- `ESC` : Fechar aplicativo

---

### 2. Painel Web Local (FastAPI + Stream MJPEG)
Se preferir visualizar e controlar pelo navegador:

```bash
python main.py --host 127.0.0.1 --port 8000
```
Acesse no seu navegador: **http://127.0.0.1:8000**

---

#### Acesso pela rede
Por padrão o painel só aceita conexões de `127.0.0.1` e não pede senha. Ao usar
outro host (ex.: `--host 0.0.0.0`), todas as rotas passam a exigir um token:

```bash
# Token gerado automaticamente; o endereço completo é exibido no terminal
python main.py --host 0.0.0.0

# Ou defina o seu
python main.py --host 0.0.0.0 --api-token "meu-token"   # ou variável VISAO_API_TOKEN
```

Abra `http://IP:8000/?token=SEU_TOKEN` uma vez; o navegador guarda um cookie e as
próximas visitas dispensam o token. Clientes de API podem enviar o cabeçalho
`X-API-Token`. A conexão é HTTP sem criptografia: use apenas em rede confiável.

### 3. Modo Janela Rápida OpenCV
Para uma visualização leve diretamente via OpenCV:

```bash
python main.py --desktop-gui --resolution 640x480 --fps 30 --mode thermal
```

---

## 🧪 Testes Automatizados

O projeto conta com uma suíte de 79 testes automatizados cobrindo API, CLI, app desktop, simulador térmico e detector de movimento:

```bash
pytest --cov=app --cov-report=term-missing -v
```

---

## 📁 Estrutura do Projeto

```
sistema-visao-industrial/
├── app/
│   ├── api.py               # Servidor Web local FastAPI e stream MJPEG
│   ├── camera_capture.py    # Captura da webcam em thread dedicada (não trava a interface)
│   ├── camera_manager.py    # Orquestrador de captura da webcam e telemetria
│   ├── cli.py               # Interface de linha de comando
│   ├── config.py            # Modelos de configuração imutáveis (Pydantic)
│   ├── desktop_app.py       # Aplicativo Desktop Nativo (Tkinter SCADA)
│   ├── desktop_tabs.py      # Abas de controle (câmera, termografia, movimento)
│   ├── desktop_widgets.py   # Tema, fábricas de widgets e geometria do vídeo
│   ├── frame_pump.py        # Thread que processa o vídeo fora da interface
│   ├── persistence.py       # Configuração e eventos salvos em disco
│   ├── frame_utils.py       # Análise em resolução reduzida e percentis por histograma
│   ├── hud_renderer.py      # Renderizador de camadas e informações na tela
│   ├── motion_detector.py   # Motor de detecção e rastreamento de movimento
│   ├── thermal_simulator.py # Motor de simulação termográfica infravermelha
│   └── static/              # Interface Web (HTML5/CSS3/JavaScript)
├── desktop_app.py           # Atalho de inicialização do app desktop
├── main.py                  # Ponto de entrada unificado
├── pyproject.toml           # Configuração de pacote Python
├── requirements.txt         # Lista de dependências
├── run_app.sh               # Script de execução rápida do app local (Linux)
├── run_app.bat              # Script de execução rápida do app local (Windows)
└── tests/                   # Suíte de testes unitários e de integração
```

---

## 📄 Licença

Distribuído sob a licença MIT. Sinta-se livre para usar, modificar e distribuir para fins pessoais, acadêmicos ou comerciais.
