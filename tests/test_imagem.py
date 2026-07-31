"""Testes da fronteira de imagem."""

import cv2
import numpy as np
import pytest

from app.imagem import (
    ErroImagem,
    bgr_para_rgb,
    carregar_rgb,
    listar_imagens,
    reduzir,
    rgb_para_bgr,
)


class TestConversaoDeCor:
    def test_bgr_para_rgb_inverte_os_canais(self):
        bgr = np.zeros((2, 2, 3), dtype=np.uint8)
        bgr[:, :, 0] = 255  # canal azul em BGR
        rgb = bgr_para_rgb(bgr)
        assert rgb[0, 0, 2] == 255  # vira canal azul em RGB
        assert rgb[0, 0, 0] == 0

    def test_ida_e_volta_preserva_a_imagem(self):
        original = np.random.default_rng(7).integers(
            0, 256, (5, 5, 3), dtype=np.uint8
        )
        np.testing.assert_array_equal(
            rgb_para_bgr(bgr_para_rgb(original)), original
        )


class TestCarregarRgb:
    def test_le_arquivo_e_devolve_rgb(self, tmp_path):
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        bgr[:, :, 2] = 200  # vermelho em BGR
        destino = tmp_path / "foto.png"
        cv2.imwrite(str(destino), bgr)

        rgb = carregar_rgb(destino)
        assert rgb.shape == (10, 10, 3)
        assert rgb[0, 0, 0] == 200  # vermelho agora no canal 0

    def test_arquivo_inexistente_levanta(self, tmp_path):
        with pytest.raises(ErroImagem, match="nao encontrado"):
            carregar_rgb(tmp_path / "fantasma.png")

    def test_extensao_nao_suportada_levanta(self, tmp_path):
        alvo = tmp_path / "doc.pdf"
        alvo.write_bytes(b"nao e imagem")
        with pytest.raises(ErroImagem, match="nao suportada"):
            carregar_rgb(alvo)

    def test_arquivo_corrompido_levanta(self, tmp_path):
        alvo = tmp_path / "quebrada.png"
        alvo.write_bytes(b"conteudo invalido")
        with pytest.raises(ErroImagem, match="decodificar"):
            carregar_rgb(alvo)


class TestListarImagens:
    def test_lista_apenas_imagens_em_ordem(self, tmp_path):
        for nome in ["b.jpg", "a.png", "nota.txt", "c.JPEG"]:
            (tmp_path / nome).write_bytes(b"x")
        nomes = [p.name for p in listar_imagens(tmp_path)]
        assert nomes == ["a.png", "b.jpg", "c.JPEG"]

    def test_pasta_inexistente_devolve_vazio(self, tmp_path):
        assert listar_imagens(tmp_path / "nao_existe") == []


class TestReduzir:
    def test_reduz_pelas_dimensoes_esperadas(self):
        grande = np.zeros((400, 800, 3), dtype=np.uint8)
        assert reduzir(grande, 0.25).shape[:2] == (100, 200)

    def test_escala_um_devolve_a_mesma_imagem(self):
        imagem = np.zeros((10, 10, 3), dtype=np.uint8)
        assert reduzir(imagem, 1.0) is imagem

    def test_escala_invalida_levanta(self):
        imagem = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(ErroImagem, match="entre 0"):
            reduzir(imagem, 0)
