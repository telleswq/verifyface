"""Testes do dominio. Nao carregam modelo nem imagem."""

import numpy as np
import pytest

from app.dominio import (
    DESCONHECIDO,
    CaixaRosto,
    Identificacao,
    Pessoa,
    ResultadoQuadro,
)


class TestCaixaRosto:
    def test_calcula_dimensoes(self):
        caixa = CaixaRosto(topo=10, direita=60, baixo=110, esquerda=20)
        assert caixa.largura == 40
        assert caixa.altura == 100
        assert caixa.area == 4000

    def test_caixa_degenerada_e_rejeitada(self):
        with pytest.raises(ValueError, match="invalida"):
            CaixaRosto(topo=100, direita=60, baixo=10, esquerda=20)

    def test_caixa_invertida_horizontalmente_e_rejeitada(self):
        with pytest.raises(ValueError, match="invalida"):
            CaixaRosto(topo=10, direita=20, baixo=110, esquerda=60)

    def test_escalar_multiplica_coordenadas(self):
        pequena = CaixaRosto(topo=10, direita=20, baixo=30, esquerda=5)
        grande = pequena.escalar(4.0)
        assert (grande.topo, grande.direita) == (40, 80)
        assert (grande.baixo, grande.esquerda) == (120, 20)

    def test_escalar_ida_e_volta_preserva_coordenadas(self):
        """Reduzir para detectar e ampliar para desenhar deve fechar."""
        original = CaixaRosto(topo=100, direita=200, baixo=300, esquerda=40)
        volta = original.escalar(0.25).escalar(4.0)
        assert volta == original

    def test_escalar_com_fator_invalido_falha(self):
        caixa = CaixaRosto(topo=1, direita=2, baixo=3, esquerda=0)
        with pytest.raises(ValueError, match="positivo"):
            caixa.escalar(0)

    def test_converte_para_convencao_opencv(self):
        caixa = CaixaRosto(topo=10, direita=60, baixo=110, esquerda=20)
        assert caixa.para_opencv() == (20, 10, 40, 100)


class TestPessoa:
    def test_nome_e_normalizado(self):
        assert Pessoa(nome="  Gabriel  ").nome == "Gabriel"

    def test_nome_vazio_e_rejeitado(self):
        with pytest.raises(ValueError, match="vazio"):
            Pessoa(nome="   ")

    def test_nome_reservado_e_rejeitado(self):
        with pytest.raises(ValueError, match="reservado"):
            Pessoa(nome=DESCONHECIDO)

    def test_acumula_multiplas_amostras(self):
        pessoa = Pessoa(nome="Guilherme")
        assert pessoa.total_amostras == 0
        pessoa.adicionar(np.zeros(128))
        pessoa.adicionar(np.ones(128))
        assert pessoa.total_amostras == 2

    def test_embedding_e_convertido_para_float64(self):
        pessoa = Pessoa(nome="Gabriel")
        pessoa.adicionar([0.1] * 128)
        assert pessoa.embeddings[0].dtype == np.float64


class TestIdentificacao:
    def _caixa(self) -> CaixaRosto:
        return CaixaRosto(topo=0, direita=10, baixo=10, esquerda=0)

    def test_nome_reservado_marca_como_nao_conhecido(self):
        ident = Identificacao(self._caixa(), DESCONHECIDO, 0.85)
        assert ident.conhecido is False

    def test_nome_real_marca_como_conhecido(self):
        ident = Identificacao(self._caixa(), "Gabriel", 0.31)
        assert ident.conhecido is True

    def test_confianca_e_complemento_da_distancia(self):
        assert Identificacao(self._caixa(), "X", 0.30).confianca == 0.7

    def test_confianca_fica_limitada_ao_intervalo(self):
        assert Identificacao(self._caixa(), "X", 1.9).confianca == 0.0
        assert Identificacao(self._caixa(), "X", -0.5).confianca == 1.0

    def test_rotulo_inclui_distancia_por_padrao(self):
        ident = Identificacao(self._caixa(), "Gabriel", 0.4157)
        assert ident.rotulo() == "Gabriel (0.42)"

    def test_rotulo_pode_omitir_distancia(self):
        ident = Identificacao(self._caixa(), "Gabriel", 0.41)
        assert ident.rotulo(com_distancia=False) == "Gabriel"


class TestResultadoQuadro:
    def test_separa_conhecidos_de_desconhecidos(self):
        caixa = CaixaRosto(topo=0, direita=10, baixo=10, esquerda=0)
        resultado = ResultadoQuadro(
            identificacoes=[
                Identificacao(caixa, "Gabriel", 0.3),
                Identificacao(caixa, DESCONHECIDO, 0.9),
                Identificacao(caixa, "Guilherme", 0.4),
            ],
            largura=1920,
            altura=1080,
        )
        assert resultado.total_rostos == 3
        assert len(resultado.conhecidos) == 2
        assert len(resultado.desconhecidos) == 1

    def test_quadro_sem_rostos(self):
        vazio = ResultadoQuadro(identificacoes=[], largura=640, altura=480)
        assert vazio.total_rostos == 0
        assert vazio.conhecidos == []
