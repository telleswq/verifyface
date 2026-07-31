"""Entrada e saida de imagem. Fronteira BGR <-> RGB.

Este e o UNICO modulo que conhece a convencao BGR do OpenCV. Tudo acima
dele trabalha em RGB, conforme o contrato dos motores.
"""

from pathlib import Path

import cv2
import numpy as np

EXTENSOES_ACEITAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ErroImagem(RuntimeError):
    """Falha ao ler ou converter imagem."""


def bgr_para_rgb(imagem_bgr: np.ndarray) -> np.ndarray:
    """Converte frame do OpenCV para o formato dos motores."""
    return cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)


def rgb_para_bgr(imagem_rgb: np.ndarray) -> np.ndarray:
    """Converte de volta para gravar ou exibir com OpenCV."""
    return cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2BGR)


def carregar_rgb(caminho: Path) -> np.ndarray:
    """Le uma imagem do disco e devolve em RGB.

    Usa imdecode em vez de imread porque imread falha silenciosamente
    (retorna None) com caminhos contendo caracteres nao-ASCII - comum em
    nomes de arquivo em portugues.
    """
    caminho = Path(caminho)
    if not caminho.is_file():
        raise ErroImagem(f"Arquivo nao encontrado: {caminho}")
    if caminho.suffix.lower() not in EXTENSOES_ACEITAS:
        aceitas = ", ".join(sorted(EXTENSOES_ACEITAS))
        raise ErroImagem(
            f"Extensao '{caminho.suffix}' nao suportada. "
            f"Aceitas: {aceitas}"
        )

    dados = np.fromfile(str(caminho), dtype=np.uint8)
    imagem_bgr = cv2.imdecode(dados, cv2.IMREAD_COLOR)
    if imagem_bgr is None:
        raise ErroImagem(
            f"Nao foi possivel decodificar a imagem: {caminho.name}"
        )

    return bgr_para_rgb(imagem_bgr)


def listar_imagens(pasta: Path) -> list[Path]:
    """Lista imagens de uma pasta, em ordem estavel."""
    pasta = Path(pasta)
    if not pasta.is_dir():
        return []
    return sorted(
        p
        for p in pasta.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSOES_ACEITAS
    )


def reduzir(imagem: np.ndarray, escala: float) -> np.ndarray:
    """Reduz a imagem para acelerar a deteccao.

    INTER_AREA e a interpolacao correta para reducao - preserva melhor a
    estrutura do que INTER_LINEAR.
    """
    if not 0 < escala <= 1:
        raise ErroImagem("A escala deve estar entre 0 (exclusivo) e 1")
    if escala == 1:
        return imagem
    return cv2.resize(
        imagem, (0, 0), fx=escala, fy=escala, interpolation=cv2.INTER_AREA
    )
