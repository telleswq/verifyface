"""Persistencia de pessoas conhecidas.

FORMATO: um arquivo .npz por pessoa, em galeria/pessoas/.

    Por que nao um banco unico:
    excluir uma pessoa vira remover um arquivo, sem reescrever nada e
    sem deixar rastro no restante. O direito de eliminacao (LGPD art.
    18, VI) passa a ser propriedade do formato, nao funcionalidade
    adicional que alguem pode esquecer de implementar.

SEGURANCA - allow_pickle=False:
    Arquivos .npy/.npz com pickle habilitado executam codigo arbitrario
    ao serem carregados. Um arquivo malicioso na pasta viraria execucao
    remota de codigo. Desabilitamos explicitamente; embeddings sao
    apenas arrays numericos e nao precisam de pickle.

COMPATIBILIDADE DE MOTOR:
    Cada arquivo registra qual motor gerou seus embeddings. Comparar um
    vetor de 128-d do dlib com um de 512-d do InsightFace nao levanta
    erro - produz distancias sem significado. A galeria recusa a mistura.
"""

import hashlib
import re
import unicodedata
from pathlib import Path

import numpy as np

from app.dominio import Pessoa

NOME_SUBPASTA = "pessoas"
CARACTERES_SLUG = re.compile(r"[^a-z0-9]+")


class ErroGaleria(RuntimeError):
    """Falha ao ler ou gravar a galeria."""


def gerar_slug(nome: str) -> str:
    """Converte um nome em identificador de arquivo seguro.

    O hash curto do nome ORIGINAL evita colisao: "Jose" e "Jose" com
    acento produziriam o mesmo slug textual, mas hashes diferentes.
    """
    normalizado = unicodedata.normalize("NFKD", nome)
    sem_acento = normalizado.encode("ascii", "ignore").decode("ascii")
    base = CARACTERES_SLUG.sub("_", sem_acento.lower()).strip("_")
    digest = hashlib.blake2b(
        nome.encode("utf-8"), digest_size=3
    ).hexdigest()
    return f"{base or 'pessoa'}_{digest}"


class Galeria:
    """Colecao de pessoas conhecidas, persistida em disco."""

    def __init__(self, pasta: Path, motor: str, dimensao: int) -> None:
        self.pasta = Path(pasta) / NOME_SUBPASTA
        self.pasta.mkdir(parents=True, exist_ok=True)
        self.motor = motor
        self.dimensao = dimensao
        self._pessoas: dict[str, Pessoa] = {}

    # --- consulta ---------------------------------------------------

    @property
    def pessoas(self) -> dict[str, Pessoa]:
        return dict(self._pessoas)

    @property
    def total_pessoas(self) -> int:
        return len(self._pessoas)

    @property
    def total_amostras(self) -> int:
        return sum(p.total_amostras for p in self._pessoas.values())

    @property
    def vazia(self) -> bool:
        return self.total_amostras == 0

    def nomes(self) -> list[str]:
        return sorted(self._pessoas)

    # --- escrita ----------------------------------------------------

    def _validar(self, embedding: np.ndarray) -> np.ndarray:
        vetor = np.asarray(embedding, dtype=np.float64)
        if vetor.ndim != 1 or vetor.shape[0] != self.dimensao:
            raise ErroGaleria(
                f"Embedding com shape {vetor.shape}; esperado "
                f"({self.dimensao},) para o motor '{self.motor}'."
            )
        return vetor

    def registrar(self, nome: str, embeddings: list[np.ndarray]) -> Pessoa:
        """Adiciona ou complementa uma pessoa e grava em disco.

        Chamar de novo com o mesmo nome ACUMULA amostras, nao substitui:
        cadastrar varias fotos em momentos diferentes e o fluxo esperado.
        """
        if not embeddings:
            raise ErroGaleria(
                f"Nenhum embedding fornecido para '{nome}'."
            )

        pessoa = self._pessoas.get(nome) or Pessoa(nome=nome)
        for bruto in embeddings:
            pessoa.adicionar(self._validar(bruto))

        self._pessoas[pessoa.nome] = pessoa
        self._gravar(pessoa)
        return pessoa

    def _gravar(self, pessoa: Pessoa) -> None:
        destino = self.pasta / f"{gerar_slug(pessoa.nome)}.npz"
        matriz = np.vstack(pessoa.embeddings)
        try:
            np.savez_compressed(
                destino,
                nome=np.array(pessoa.nome),
                embeddings=matriz,
                motor=np.array(self.motor),
                dimensao=np.array(self.dimensao),
            )
        except OSError as exc:
            raise ErroGaleria(
                f"Falha ao gravar '{pessoa.nome}': {exc}"
            ) from exc

    # --- leitura ----------------------------------------------------

    def carregar(self) -> int:
        """Le todos os arquivos da pasta. Devolve quantas pessoas leu.

        Arquivos de outro motor sao IGNORADOS com aviso silencioso, nao
        levantam erro: uma galeria antiga nao deve impedir o programa de
        subir com um motor novo.
        """
        self._pessoas.clear()
        for arquivo in sorted(self.pasta.glob("*.npz")):
            pessoa = self._ler_arquivo(arquivo)
            if pessoa is not None:
                self._pessoas[pessoa.nome] = pessoa
        return self.total_pessoas

    def _ler_arquivo(self, arquivo: Path) -> Pessoa | None:
        try:
            with np.load(arquivo, allow_pickle=False) as dados:
                motor = str(dados["motor"])
                if motor != self.motor:
                    return None
                nome = str(dados["nome"])
                matriz = np.asarray(dados["embeddings"], dtype=np.float64)
        except (OSError, ValueError, KeyError) as exc:
            raise ErroGaleria(
                f"Arquivo de galeria invalido ({arquivo.name}): {exc}"
            ) from exc

        if matriz.ndim != 2 or matriz.shape[1] != self.dimensao:
            raise ErroGaleria(
                f"Arquivo {arquivo.name} tem embeddings de shape "
                f"{matriz.shape}; esperado (N, {self.dimensao})."
            )

        pessoa = Pessoa(nome=nome)
        for linha in matriz:
            pessoa.adicionar(linha)
        return pessoa

    # --- comparacao -------------------------------------------------

    def matriz(self) -> tuple[list[str], np.ndarray]:
        """Devolve (nomes, matriz N x dimensao) para calculo vetorizado.

        Cada linha da matriz corresponde ao nome de mesmo indice na
        lista. Uma pessoa com 5 amostras aparece 5 vezes.

        Existe para que a distancia contra TODA a galeria seja uma unica
        operacao numpy, em vez de um laco Python por pessoa - a
        diferenca aparece em video, onde isso roda a cada frame.
        """
        if self.vazia:
            return [], np.empty((0, self.dimensao), dtype=np.float64)

        nomes: list[str] = []
        blocos: list[np.ndarray] = []
        for nome in self.nomes():
            pessoa = self._pessoas[nome]
            nomes.extend([nome] * pessoa.total_amostras)
            blocos.append(np.vstack(pessoa.embeddings))

        return nomes, np.vstack(blocos)

    # --- exclusao (LGPD art. 18, VI) --------------------------------

    def remover(self, nome: str) -> bool:
        """Exclui uma pessoa da memoria e do disco."""
        if nome not in self._pessoas:
            return False
        self._pessoas.pop(nome)
        arquivo = self.pasta / f"{gerar_slug(nome)}.npz"
        arquivo.unlink(missing_ok=True)
        return True

    def limpar(self) -> int:
        """Exclui TODA a galeria. Devolve quantos arquivos removeu.

        Implementa o direito de eliminacao de forma completa: remove
        todo .npz da pasta, inclusive os de outro motor que nao estavam
        carregados em memoria.
        """
        removidos = 0
        for arquivo in self.pasta.glob("*.npz"):
            arquivo.unlink(missing_ok=True)
            removidos += 1
        self._pessoas.clear()
        return removidos
