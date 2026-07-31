"""Testes do reconhecedor.

Usam embeddings sinteticos com distancias conhecidas: o comportamento do
limiar e verificavel exatamente, sem depender de modelo real.
"""

import numpy as np
import pytest

from app.dominio import DESCONHECIDO, CaixaRosto
from app.reconhecedor import Reconhecedor

DIM = 128


def vetor(valor: float) -> np.ndarray:
    """Vetor constante. A distancia entre vetor(a) e vetor(b) e
    |a - b| * sqrt(DIM), tornando as distancias previsiveis."""
    return np.full(DIM, valor, dtype=np.float64)


def caixa() -> CaixaRosto:
    return CaixaRosto(topo=0, direita=10, baixo=10, esquerda=0)


@pytest.fixture
def reconhecedor() -> Reconhecedor:
    """Gabriel em 0.0 (2 amostras), Guilherme em 1.0."""
    return Reconhecedor(
        nomes=["Gabriel", "Gabriel", "Guilherme"],
        matriz=np.vstack([vetor(0.0), vetor(0.02), vetor(1.0)]),
        limiar=0.6,
    )


class TestConstrucao:
    def test_nomes_e_matriz_devem_ter_o_mesmo_tamanho(self):
        with pytest.raises(ValueError, match="devem ser iguais"):
            Reconhecedor(nomes=["A", "B"], matriz=np.zeros((3, DIM)))

    def test_galeria_vazia_e_detectada(self):
        vazio = Reconhecedor(nomes=[], matriz=np.empty((0, DIM)))
        assert vazio.vazio is True


class TestDistancias:
    def test_calcula_uma_distancia_por_amostra(self, reconhecedor):
        assert reconhecedor.distancias(vetor(0.0)).shape == (3,)

    def test_distancia_a_si_mesmo_e_zero(self, reconhecedor):
        assert reconhecedor.distancias(vetor(0.0))[0] == pytest.approx(0.0)

    def test_embedding_de_shape_errado_e_rejeitado(self, reconhecedor):
        with pytest.raises(ValueError, match="esperado"):
            reconhecedor.distancias(np.zeros(64))

    def test_galeria_vazia_devolve_array_vazio(self):
        vazio = Reconhecedor(nomes=[], matriz=np.empty((0, DIM)))
        assert vazio.distancias(vetor(0.0)).size == 0


class TestIdentificacao:
    def test_correspondencia_exata_identifica(self, reconhecedor):
        ident = reconhecedor.identificar(caixa(), vetor(0.0))
        assert ident.nome == "Gabriel"
        assert ident.conhecido is True
        assert ident.distancia == pytest.approx(0.0)

    def test_acima_do_limiar_vira_desconhecido(self, reconhecedor):
        """vetor(0.5) dista ~5.6 de tudo - muito alem de 0.6."""
        ident = reconhecedor.identificar(caixa(), vetor(0.5))
        assert ident.nome == DESCONHECIDO
        assert ident.conhecido is False

    def test_distancia_e_reportada_mesmo_sem_correspondencia(
        self, reconhecedor
    ):
        """Sem a distancia nao ha como calibrar nem auditar."""
        ident = reconhecedor.identificar(caixa(), vetor(0.5))
        assert ident.distancia > 0
        assert np.isfinite(ident.distancia)

    def test_vence_a_menor_distancia_nao_a_media(self):
        """Gabriel tem uma amostra otima e uma pessima; Guilherme, duas
        medianas. Por media Guilherme venceria; por 1-NN, Gabriel."""
        rec = Reconhecedor(
            nomes=["Gabriel", "Gabriel", "Guilherme", "Guilherme"],
            matriz=np.vstack(
                [vetor(0.0), vetor(0.9), vetor(0.3), vetor(0.35)]
            ),
            limiar=10.0,
        )
        assert rec.identificar(caixa(), vetor(0.0)).nome == "Gabriel"

    def test_limiar_mais_estrito_rejeita_o_que_o_frouxo_aceita(self):
        matriz = np.vstack([vetor(0.0)])
        alvo = vetor(0.04)  # dista ~0.45

        frouxo = Reconhecedor(["Gabriel"], matriz, limiar=0.6)
        estrito = Reconhecedor(["Gabriel"], matriz, limiar=0.2)

        assert frouxo.identificar(caixa(), alvo).nome == "Gabriel"
        assert estrito.identificar(caixa(), alvo).nome == DESCONHECIDO

    def test_galeria_vazia_devolve_desconhecido_com_infinito(self):
        vazio = Reconhecedor(nomes=[], matriz=np.empty((0, DIM)))
        ident = vazio.identificar(caixa(), vetor(0.0))
        assert ident.nome == DESCONHECIDO
        assert ident.distancia == float("inf")

    def test_a_caixa_recebida_e_preservada(self, reconhecedor):
        original = CaixaRosto(topo=5, direita=90, baixo=105, esquerda=10)
        ident = reconhecedor.identificar(original, vetor(0.0))
        assert ident.caixa == original


class TestVariosRostos:
    def test_identifica_cada_rosto_do_quadro(self, reconhecedor):
        resultado = reconhecedor.identificar_varios(
            caixas=[caixa(), caixa(), caixa()],
            embeddings=[vetor(0.0), vetor(1.0), vetor(0.5)],
            largura=1920,
            altura=1080,
        )
        assert resultado.total_rostos == 3
        assert len(resultado.conhecidos) == 2
        assert len(resultado.desconhecidos) == 1
        assert resultado.largura == 1920

    def test_quantidades_divergentes_sao_rejeitadas(self, reconhecedor):
        with pytest.raises(ValueError, match="caixas"):
            reconhecedor.identificar_varios(
                caixas=[caixa()],
                embeddings=[vetor(0.0), vetor(1.0)],
                largura=10,
                altura=10,
            )

    def test_quadro_sem_rostos(self, reconhecedor):
        resultado = reconhecedor.identificar_varios([], [], 640, 480)
        assert resultado.total_rostos == 0


class TestRanking:
    def test_ordena_por_distancia_crescente(self, reconhecedor):
        ranking = reconhecedor.ranking(vetor(0.0))
        assert [nome for nome, _ in ranking] == ["Gabriel", "Guilherme"]
        assert ranking[0][1] < ranking[1][1]

    def test_colapsa_amostras_na_melhor_por_pessoa(self, reconhecedor):
        """Gabriel tem 2 amostras mas aparece uma vez no ranking."""
        assert len(reconhecedor.ranking(vetor(0.0))) == 2

    def test_respeita_o_limite_de_resultados(self, reconhecedor):
        assert len(reconhecedor.ranking(vetor(0.0), topo=1)) == 1

    def test_galeria_vazia_devolve_ranking_vazio(self):
        vazio = Reconhecedor(nomes=[], matriz=np.empty((0, DIM)))
        assert vazio.ranking(vetor(0.0)) == []
