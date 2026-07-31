"""Entidades do dominio.

REGRA: este modulo nao importa OpenCV, dlib nem face_recognition.

Motivo: a regra de negocio (isto e a mesma pessoa? qual a confianca?)
independe da biblioteca que produziu o embedding. Mantendo o dominio
puro, ela e testavel em milissegundos, sem carregar modelo nem imagem -
e sobrevive a troca de motor.
"""

from dataclasses import dataclass, field

import numpy as np

DESCONHECIDO = "Desconhecido"


@dataclass(frozen=True)
class CaixaRosto:
    """Retangulo de um rosto na imagem.

    A ordem dos campos segue a convencao do face_recognition
    (topo, direita, baixo, esquerda) e nao a do OpenCV (x, y, w, h).
    A conversao acontece na camada de anotacao.
    """

    topo: int
    direita: int
    baixo: int
    esquerda: int

    def __post_init__(self) -> None:
        if self.baixo <= self.topo or self.direita <= self.esquerda:
            raise ValueError(
                f"Caixa invalida: topo={self.topo} direita={self.direita} "
                f"baixo={self.baixo} esquerda={self.esquerda}"
            )

    @property
    def largura(self) -> int:
        return self.direita - self.esquerda

    @property
    def altura(self) -> int:
        return self.baixo - self.topo

    @property
    def area(self) -> int:
        return self.largura * self.altura

    def escalar(self, fator: float) -> "CaixaRosto":
        """Reescala as coordenadas.

        Essencial para performance de video: detectamos num frame
        reduzido e devolvemos a caixa ao tamanho original com
        escalar(1 / escala_processamento).
        """
        if fator <= 0:
            raise ValueError("O fator de escala deve ser positivo")
        return CaixaRosto(
            topo=int(self.topo * fator),
            direita=int(self.direita * fator),
            baixo=int(self.baixo * fator),
            esquerda=int(self.esquerda * fator),
        )

    def para_opencv(self) -> tuple[int, int, int, int]:
        """Converte para (x, y, largura, altura) do OpenCV."""
        return (self.esquerda, self.topo, self.largura, self.altura)


@dataclass
class Pessoa:
    """Pessoa cadastrada na galeria.

    Guarda MULTIPLOS embeddings de proposito: uma foto so captura um
    angulo e uma iluminacao. Cadastrar de 3 a 5 fotos variadas reduz
    drasticamente o falso negativo.
    """

    nome: str
    embeddings: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.nome = self.nome.strip()
        if not self.nome:
            raise ValueError("O nome da pessoa nao pode ser vazio")
        if self.nome == DESCONHECIDO:
            raise ValueError(f"'{DESCONHECIDO}' e um nome reservado")

    @property
    def total_amostras(self) -> int:
        return len(self.embeddings)

    def adicionar(self, embedding: np.ndarray) -> None:
        self.embeddings.append(np.asarray(embedding, dtype=np.float64))


@dataclass(frozen=True)
class Identificacao:
    """Resultado de identificar um rosto detectado.

    A distancia e SEMPRE exposta, mesmo quando ha correspondencia. Sem
    ela nao ha como calibrar o limiar nem auditar por que o sistema
    errou - e, dados os diferenciais de acuracia entre grupos
    demograficos, auditar e requisito, nao luxo.
    """

    caixa: CaixaRosto
    nome: str
    distancia: float

    @property
    def conhecido(self) -> bool:
        return self.nome != DESCONHECIDO

    @property
    def confianca(self) -> float:
        """Converte distancia em score de 0 a 1, apenas para exibicao.

        AVISO: nao e probabilidade. E uma transformacao linear
        monotonica da distancia euclidiana, util para leitura humana e
        inutil para decisao estatistica. Para decidir, use a distancia.
        """
        return round(max(0.0, min(1.0, 1.0 - self.distancia)), 3)

    def rotulo(self, com_distancia: bool = True) -> str:
        """Texto a desenhar sobre o retangulo."""
        if not com_distancia:
            return self.nome
        return f"{self.nome} ({self.distancia:.2f})"


@dataclass(frozen=True)
class ResultadoQuadro:
    """Todas as identificacoes de um unico frame ou foto."""

    identificacoes: list[Identificacao]
    largura: int
    altura: int

    @property
    def total_rostos(self) -> int:
        return len(self.identificacoes)

    @property
    def conhecidos(self) -> list[Identificacao]:
        return [i for i in self.identificacoes if i.conhecido]

    @property
    def desconhecidos(self) -> list[Identificacao]:
        return [i for i in self.identificacoes if not i.conhecido]
