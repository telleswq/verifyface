"""Motor falso, deterministico, para testes.

Satisfaz o Protocol MotorFacial sem carregar modelo algum. Sua
existencia e a prova de que a abstracao funciona: se este arquivo
compila e os testes passam, a regra de negocio nao depende do dlib.
"""

import numpy as np

from app.dominio import CaixaRosto
from app.motores.base import validar_rgb


class MotorFalso:
    nome = "falso"
    dimensao_embedding = 128

    def __init__(self, caixas: list[CaixaRosto] | None = None) -> None:
        self.caixas = caixas or []
        self.chamadas_detectar = 0
        self.chamadas_codificar = 0

    def detectar(self, imagem_rgb: np.ndarray) -> list[CaixaRosto]:
        validar_rgb(imagem_rgb)
        self.chamadas_detectar += 1
        return list(self.caixas)

    def codificar(
        self, imagem_rgb: np.ndarray, caixas: list[CaixaRosto]
    ) -> list[np.ndarray]:
        validar_rgb(imagem_rgb)
        self.chamadas_codificar += 1
        # Embedding deterministico derivado da posicao: caixas iguais
        # produzem embeddings iguais, caixas diferentes produzem
        # embeddings diferentes.
        saida = []
        for c in caixas:
            gerador = np.random.default_rng(seed=c.topo * 1000 + c.esquerda)
            saida.append(gerador.random(self.dimensao_embedding))
        return saida
