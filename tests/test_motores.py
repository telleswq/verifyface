"""Testes do contrato e da fabrica de motores.

Nenhum teste carrega o modelo real.
"""

import numpy as np
import pytest

from app.dominio import CaixaRosto
from app.motores import DISPONIVEIS, obter_motor
from app.motores.base import ErroMotor, validar_rgb
from tests.motor_falso import MotorFalso


def imagem_valida(altura: int = 60, largura: int = 80) -> np.ndarray:
    return np.zeros((altura, largura, 3), dtype=np.uint8)


class TestValidarRgb:
    def test_aceita_uint8_de_tres_canais(self):
        validar_rgb(imagem_valida())

    def test_rejeita_imagem_vazia(self):
        with pytest.raises(ErroMotor, match="vazia"):
            validar_rgb(np.array([], dtype=np.uint8))

    def test_rejeita_escala_de_cinza(self):
        with pytest.raises(ErroMotor, match="3 canais"):
            validar_rgb(np.zeros((60, 80), dtype=np.uint8))

    def test_rejeita_rgba(self):
        with pytest.raises(ErroMotor, match="3 canais"):
            validar_rgb(np.zeros((60, 80, 4), dtype=np.uint8))

    def test_rejeita_dtype_incorreto(self):
        with pytest.raises(ErroMotor, match="uint8"):
            validar_rgb(np.zeros((60, 80, 3), dtype=np.float32))


class TestFabrica:
    def test_seleciona_por_nome(self):
        assert "face_recognition" in DISPONIVEIS

    def test_nome_invalido_levanta_com_opcoes(self):
        with pytest.raises(ValueError, match="face_recognition"):
            obter_motor("inexistente")


class TestContratoComMotorFalso:
    """Prova que a abstracao e suficiente para a regra de negocio."""

    def test_detectar_devolve_caixas_configuradas(self):
        caixas = [CaixaRosto(topo=10, direita=50, baixo=60, esquerda=10)]
        motor = MotorFalso(caixas=caixas)
        assert motor.detectar(imagem_valida()) == caixas
        assert motor.chamadas_detectar == 1

    def test_codificar_respeita_quantidade_e_ordem(self):
        caixas = [
            CaixaRosto(topo=10, direita=50, baixo=60, esquerda=10),
            CaixaRosto(topo=70, direita=50, baixo=120, esquerda=10),
        ]
        embeddings = MotorFalso().codificar(imagem_valida(), caixas)
        assert len(embeddings) == 2
        assert all(e.shape == (128,) for e in embeddings)

    def test_codificar_e_deterministico(self):
        caixa = [CaixaRosto(topo=10, direita=50, baixo=60, esquerda=10)]
        a = MotorFalso().codificar(imagem_valida(), caixa)[0]
        b = MotorFalso().codificar(imagem_valida(), caixa)[0]
        np.testing.assert_array_equal(a, b)

    def test_caixas_distintas_geram_embeddings_distintos(self):
        motor = MotorFalso()
        a = motor.codificar(
            imagem_valida(),
            [CaixaRosto(topo=10, direita=50, baixo=60, esquerda=10)],
        )[0]
        b = motor.codificar(
            imagem_valida(),
            [CaixaRosto(topo=70, direita=50, baixo=120, esquerda=10)],
        )[0]
        assert not np.array_equal(a, b)

    def test_lista_vazia_devolve_lista_vazia(self):
        assert MotorFalso().codificar(imagem_valida(), []) == []

    def test_motor_valida_a_imagem_recebida(self):
        with pytest.raises(ErroMotor):
            MotorFalso().detectar(np.zeros((60, 80), dtype=np.uint8))
