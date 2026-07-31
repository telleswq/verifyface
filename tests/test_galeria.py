"""Testes da galeria. Nenhum carrega modelo."""

import numpy as np
import pytest

from app.galeria import ErroGaleria, Galeria, gerar_slug

DIMENSAO = 128
MOTOR = "motor_teste"


@pytest.fixture
def galeria(tmp_path) -> Galeria:
    return Galeria(pasta=tmp_path, motor=MOTOR, dimensao=DIMENSAO)


def emb(semente: int) -> np.ndarray:
    return np.random.default_rng(semente).random(DIMENSAO)


class TestSlug:
    def test_remove_acentos_e_espacos(self):
        assert gerar_slug("Gabriel Borges").startswith("gabriel_borges_")

    def test_nomes_diferentes_geram_slugs_diferentes(self):
        assert gerar_slug("Jose") != gerar_slug("José")

    def test_mesmo_nome_gera_mesmo_slug(self):
        assert gerar_slug("Guilherme") == gerar_slug("Guilherme")

    def test_nome_sem_caractere_alfanumerico_tem_fallback(self):
        assert gerar_slug("!!!").startswith("pessoa_")


class TestRegistroEPersistencia:
    def test_registrar_grava_arquivo(self, galeria, tmp_path):
        galeria.registrar("Gabriel", [emb(1)])
        arquivos = list((tmp_path / "pessoas").glob("*.npz"))
        assert len(arquivos) == 1

    def test_recarregar_recupera_os_dados(self, galeria, tmp_path):
        galeria.registrar("Gabriel", [emb(1), emb(2)])
        galeria.registrar("Guilherme", [emb(3)])

        nova = Galeria(pasta=tmp_path, motor=MOTOR, dimensao=DIMENSAO)
        assert nova.carregar() == 2
        assert nova.nomes() == ["Gabriel", "Guilherme"]
        assert nova.total_amostras == 3

    def test_embedding_sobrevive_ao_ciclo_de_disco(self, galeria, tmp_path):
        original = emb(42)
        galeria.registrar("Gabriel", [original])

        nova = Galeria(pasta=tmp_path, motor=MOTOR, dimensao=DIMENSAO)
        nova.carregar()
        recuperado = nova.pessoas["Gabriel"].embeddings[0]
        np.testing.assert_allclose(recuperado, original)

    def test_registrar_de_novo_acumula_em_vez_de_substituir(self, galeria):
        galeria.registrar("Gabriel", [emb(1)])
        galeria.registrar("Gabriel", [emb(2), emb(3)])
        assert galeria.total_pessoas == 1
        assert galeria.total_amostras == 3

    def test_nome_com_acento_e_preservado(self, galeria, tmp_path):
        galeria.registrar("José da Silva", [emb(1)])
        nova = Galeria(pasta=tmp_path, motor=MOTOR, dimensao=DIMENSAO)
        nova.carregar()
        assert nova.nomes() == ["José da Silva"]

    def test_lista_vazia_de_embeddings_e_rejeitada(self, galeria):
        with pytest.raises(ErroGaleria, match="Nenhum embedding"):
            galeria.registrar("Gabriel", [])


class TestCompatibilidadeDeMotor:
    def test_embedding_de_dimensao_errada_e_rejeitado(self, galeria):
        with pytest.raises(ErroGaleria, match="esperado"):
            galeria.registrar("Gabriel", [np.zeros(512)])

    def test_arquivo_de_outro_motor_e_ignorado(self, galeria, tmp_path):
        galeria.registrar("Gabriel", [emb(1)])

        outra = Galeria(pasta=tmp_path, motor="outro_motor", dimensao=512)
        assert outra.carregar() == 0

    def test_arquivo_corrompido_levanta_erro_claro(self, galeria, tmp_path):
        alvo = tmp_path / "pessoas" / "quebrado.npz"
        alvo.write_bytes(b"nao e um npz")
        with pytest.raises(ErroGaleria, match="invalido"):
            galeria.carregar()


class TestMatriz:
    def test_matriz_tem_uma_linha_por_amostra(self, galeria):
        galeria.registrar("Gabriel", [emb(1), emb(2)])
        galeria.registrar("Guilherme", [emb(3)])

        nomes, matriz = galeria.matriz()
        assert matriz.shape == (3, DIMENSAO)
        assert nomes == ["Gabriel", "Gabriel", "Guilherme"]

    def test_galeria_vazia_devolve_matriz_vazia(self, galeria):
        nomes, matriz = galeria.matriz()
        assert nomes == []
        assert matriz.shape == (0, DIMENSAO)


class TestExclusao:
    def test_remover_apaga_da_memoria_e_do_disco(self, galeria, tmp_path):
        galeria.registrar("Gabriel", [emb(1)])
        galeria.registrar("Guilherme", [emb(2)])

        assert galeria.remover("Gabriel") is True
        assert galeria.nomes() == ["Guilherme"]
        assert len(list((tmp_path / "pessoas").glob("*.npz"))) == 1

    def test_remover_inexistente_devolve_falso(self, galeria):
        assert galeria.remover("Ninguem") is False

    def test_limpar_apaga_tudo(self, galeria, tmp_path):
        galeria.registrar("Gabriel", [emb(1)])
        galeria.registrar("Guilherme", [emb(2)])

        assert galeria.limpar() == 2
        assert galeria.vazia is True
        assert list((tmp_path / "pessoas").glob("*.npz")) == []

    def test_limpar_remove_ate_arquivos_de_outro_motor(
        self, galeria, tmp_path
    ):
        """O direito de eliminacao nao pode depender do motor ativo."""
        outra = Galeria(pasta=tmp_path, motor="outro", dimensao=512)
        outra.registrar("Terceiro", [np.zeros(512)])

        assert galeria.limpar() == 1
        assert list((tmp_path / "pessoas").glob("*.npz")) == []

    def test_limpar_galeria_vazia_nao_falha(self, galeria):
        assert galeria.limpar() == 0
