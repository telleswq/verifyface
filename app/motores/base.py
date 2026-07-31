"""Contrato dos motores de deteccao e embedding.

INVERSAO DE DEPENDENCIA:
    A regra de negocio (isto e a mesma pessoa?) nao deveria saber se o
    embedding veio do dlib, do InsightFace ou de um mock.

    Este modulo define a abstracao. As implementacoes concretas ficam em
    modulos irmaos e sao escolhidas em runtime pela config.

CONVENCAO DE COR - a pegadinha classica:
    OpenCV le imagens em BGR. face_recognition e dlib esperam RGB.

    Alimentar um detector com BGR nao levanta erro: ele simplesmente
    detecta pior e produz embeddings inconsistentes, gerando falso
    negativo que parece problema de limiar.

    Por isso TODO metodo deste contrato recebe RGB, sempre. A conversao
    acontece na fronteira, em app/imagem.py.
"""

from typing import Protocol

import numpy as np

from app.dominio import CaixaRosto


class ErroMotor(RuntimeError):
    """Falha no motor de reconhecimento, com mensagem para o usuario."""


class MotorFacial(Protocol):
    """Interface que toda implementacao deve satisfazer.

    Protocol (structural typing): nao exige heranca. Qualquer classe com
    esta assinatura satisfaz o contrato.
    """

    nome: str
    dimensao_embedding: int

    def detectar(self, imagem_rgb: np.ndarray) -> list[CaixaRosto]:
        """Localiza rostos. Recebe imagem RGB."""
        ...

    def codificar(
        self, imagem_rgb: np.ndarray, caixas: list[CaixaRosto]
    ) -> list[np.ndarray]:
        """Gera um embedding por caixa informada. Recebe imagem RGB.

        A lista retornada tem o mesmo comprimento e a mesma ordem de
        `caixas`. Passar as caixas ja conhecidas evita redetectar.
        """
        ...


def validar_rgb(imagem: np.ndarray) -> None:
    """Valida o formato esperado por todos os motores.

    Nao detecta BGR - isso e impossivel a partir do array. Detecta
    apenas os erros estruturais: canal errado, tipo errado, vazio.
    """
    if imagem is None or imagem.size == 0:
        raise ErroMotor("Imagem vazia.")
    if imagem.ndim != 3 or imagem.shape[2] != 3:
        raise ErroMotor(
            f"Esperado array RGB de 3 canais, recebido shape "
            f"{imagem.shape}. Imagens em escala de cinza ou RGBA "
            f"precisam ser convertidas antes."
        )
    if imagem.dtype != np.uint8:
        raise ErroMotor(
            f"Esperado dtype uint8, recebido {imagem.dtype}."
        )
