"""Regra de negocio do reconhecimento.

Transforma embedding em identidade. Nao conhece OpenCV, dlib nem disco -
recebe a galeria pronta e devolve Identificacao.

METRICA: distancia euclidiana (norma L2) entre vetores de 128 dimensoes.
    O modelo do dlib foi treinado com triplet loss otimizando exatamente
    essa distancia, por isso ela e a metrica correta aqui. Cosseno
    funcionaria, mas o limiar historico de 0.6 e calibrado para L2.

REGRA DE DECISAO: vizinho mais proximo (1-NN).
    Com varias amostras por pessoa, vence a MENOR distancia individual -
    nao a media por pessoa.

    Media diluiria uma amostra excelente (mesmo angulo, mesma luz) entre
    amostras medianas. E justamente a amostra excelente que carrega o
    sinal: se uma das fotos cadastradas bate quase perfeitamente, isso e
    evidencia forte, nao um outlier a ser suavizado.

LIMIAR:
    distancia <= limiar  -> a pessoa correspondente
    distancia >  limiar  -> DESCONHECIDO

    O limiar nao e universal. Ver secao de acuracia no README: as taxas
    de erro variam entre grupos demograficos, e um valor calibrado com
    um grupo produz erro desproporcional em outro. Por isso a distancia
    medida acompanha SEMPRE a identificacao - sem ela nao ha auditoria
    nem calibracao possivel.
"""

import numpy as np

from app.config import config
from app.dominio import (
    DESCONHECIDO,
    CaixaRosto,
    Identificacao,
    ResultadoQuadro,
)


class Reconhecedor:
    """Compara embeddings contra uma galeria carregada."""

    def __init__(
        self,
        nomes: list[str],
        matriz: np.ndarray,
        limiar: float | None = None,
    ) -> None:
        """
        Args:
            nomes: nome correspondente a cada LINHA da matriz. Uma
                pessoa com 5 amostras aparece 5 vezes.
            matriz: array (N, dimensao) com todas as amostras.
            limiar: distancia maxima para considerar correspondencia.
        """
        if len(nomes) != matriz.shape[0]:
            raise ValueError(
                f"{len(nomes)} nomes para {matriz.shape[0]} linhas na "
                f"matriz - devem ser iguais."
            )
        self.nomes = nomes
        self.matriz = matriz
        self.limiar = (
            limiar if limiar is not None else config.limiar_distancia
        )

    @property
    def vazio(self) -> bool:
        return self.matriz.shape[0] == 0

    @classmethod
    def da_galeria(
        cls, galeria, limiar: float | None = None
    ) -> "Reconhecedor":
        """Constroi a partir de uma Galeria ja carregada."""
        nomes, matriz = galeria.matriz()
        return cls(nomes=nomes, matriz=matriz, limiar=limiar)

    # --- nucleo -----------------------------------------------------

    def distancias(self, embedding: np.ndarray) -> np.ndarray:
        """Distancia euclidiana do embedding a TODAS as amostras.

        Vetorizado de proposito: um unico np.linalg.norm sobre a matriz
        inteira, em vez de um laco Python por amostra. Em video isso roda
        a cada frame e a diferenca e mensuravel.
        """
        if self.vazio:
            return np.empty(0, dtype=np.float64)

        vetor = np.asarray(embedding, dtype=np.float64)
        if vetor.shape != (self.matriz.shape[1],):
            raise ValueError(
                f"Embedding de shape {vetor.shape}; esperado "
                f"({self.matriz.shape[1]},)."
            )
        return np.linalg.norm(self.matriz - vetor, axis=1)

    def identificar(
        self, caixa: CaixaRosto, embedding: np.ndarray
    ) -> Identificacao:
        """Identifica um unico rosto.

        Com galeria vazia devolve DESCONHECIDO com distancia infinita -
        semanticamente correto: nao existe nada a que se aproximar.
        """
        distancias = self.distancias(embedding)

        if distancias.size == 0:
            return Identificacao(
                caixa=caixa, nome=DESCONHECIDO, distancia=float("inf")
            )

        indice = int(np.argmin(distancias))
        menor = float(distancias[indice])
        nome = self.nomes[indice] if menor <= self.limiar else DESCONHECIDO

        return Identificacao(caixa=caixa, nome=nome, distancia=menor)

    def identificar_varios(
        self,
        caixas: list[CaixaRosto],
        embeddings: list[np.ndarray],
        largura: int,
        altura: int,
    ) -> ResultadoQuadro:
        """Identifica todos os rostos de um quadro."""
        if len(caixas) != len(embeddings):
            raise ValueError(
                f"{len(caixas)} caixas para {len(embeddings)} embeddings."
            )
        return ResultadoQuadro(
            identificacoes=[
                self.identificar(caixa, emb)
                for caixa, emb in zip(caixas, embeddings, strict=True)
            ],
            largura=largura,
            altura=altura,
        )

    # --- diagnostico ------------------------------------------------

    def ranking(
        self, embedding: np.ndarray, topo: int = 5
    ) -> list[tuple[str, float]]:
        """Menores distancias por pessoa, ordenadas.

        Ferramenta de calibracao: mostra nao so quem venceu, mas por
        quanto - e quem ficou logo atras. Uma segunda colocada muito
        proxima da primeira e sinal de que o limiar esta arriscado.
        """
        distancias = self.distancias(embedding)
        if distancias.size == 0:
            return []

        melhor_por_nome: dict[str, float] = {}
        for nome, dist in zip(self.nomes, distancias, strict=True):
            valor = float(dist)
            if nome not in melhor_por_nome or valor < melhor_por_nome[nome]:
                melhor_por_nome[nome] = valor

        return sorted(melhor_por_nome.items(), key=lambda par: par[1])[:topo]
