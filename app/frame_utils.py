"""Utilitários compartilhados de análise em resolução reduzida.

Movimento e termografia não precisam de todos os pixels do quadro: analisar uma
cópia reduzida corta o custo por quadro em várias vezes sem alterar o resultado
visível. As coordenadas voltam à escala original antes de sair dos módulos.
"""

from __future__ import annotations

import cv2
import numpy as np

ANALYSIS_MAX_WIDTH = 320
HISTOGRAM_BINS = 256


def compute_analysis_size(width: int, height: int) -> tuple[int, int, float]:
    """Retorna (largura, altura, fator) da resolução de análise para um quadro.

    O fator é quanto o quadro original é maior que o de análise (>= 1.0).
    """
    if width <= ANALYSIS_MAX_WIDTH:
        return width, height, 1.0
    factor = width / float(ANALYSIS_MAX_WIDTH)
    return ANALYSIS_MAX_WIDTH, max(1, round(height / factor)), factor


def downscale_for_analysis(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Reduz a imagem para (largura, altura); devolve a própria imagem se já couber."""
    if (image.shape[1], image.shape[0]) == size:
        return image
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def gray_percentiles(gray_u8: np.ndarray, percents: tuple[float, ...]) -> tuple[float, ...]:
    """Percentis de uma imagem uint8 via histograma de 256 faixas.

    Equivale a np.percentile com precisão de 1 nível de cinza, porém ordens de
    grandeza mais rápido (np.percentile particiona o quadro inteiro a cada chamada).
    """
    hist = cv2.calcHist([gray_u8], [0], None, [HISTOGRAM_BINS], [0, HISTOGRAM_BINS])
    cumulative = np.cumsum(hist.ravel())
    total = float(cumulative[-1])
    if total <= 0.0:
        return tuple(0.0 for _ in percents)
    targets = np.asarray(percents, dtype=np.float64) * (total / 100.0)
    levels = np.searchsorted(cumulative, targets, side="left")
    return tuple(float(min(HISTOGRAM_BINS - 1, int(level))) for level in levels)
