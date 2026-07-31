"""Configuracao central da aplicacao.

Sobrescrevivel por variavel de ambiente com prefixo RF_
(ex.: RF_LIMIAR_DISTANCIA=0.5) ou por arquivo .env.
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

RAIZ = Path(__file__).resolve().parent.parent


class Configuracao(BaseSettings):
    """Configuracao validada na inicializacao."""

    model_config = SettingsConfigDict(
        env_prefix="RF_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # --- caminhos ---------------------------------------------------
    pasta_galeria: Path = RAIZ / "galeria"
    pasta_midia: Path = RAIZ / "midia"
    pasta_saida: Path = RAIZ / "saida"

    # --- motor de reconhecimento ------------------------------------
    # "face_recognition" hoje; "insightface" quando existir.
    motor: str = "face_recognition"

    # "hog"  = rapido, CPU, menos preciso em rostos de perfil
    # "cnn"  = mais preciso, porem exige CUDA para ser viavel
    #
    # EM macOS: nao existe CUDA (nem Apple Silicon nem Intel). O modelo
    # cnn roda em CPU a varios segundos por frame - inutilizavel em
    # video. Mantenha "hog".
    modelo_deteccao: Literal["hog", "cnn"] = "hog"

    # Reamostragens ao detectar. Valores maiores encontram rostos
    # menores ao custo de tempo.
    upsample_deteccao: int = 1

    # --- reconhecimento ---------------------------------------------
    # Distancia euclidiana maxima entre embeddings para considerar a
    # mesma pessoa. 0.6 e o padrao historico do dlib; abaixo de 0.5 fica
    # conservador (menos falso positivo, mais falso negativo).
    #
    # ATENCAO: este valor NAO e universal. Ver secao de acuracia no
    # README. Calibre com evidencia antes de confiar.
    limiar_distancia: float = 0.6

    # --- captura de video -------------------------------------------
    # Indice da camera. Atencao no macOS: com Continuity Camera ativa o
    # iPhone pode assumir o indice 0. Use tools/preflight_macos.py para
    # listar os indices disponiveis.
    indice_camera: int = 0

    # --- performance de video ---------------------------------------
    # Detectar em frame reduzido e MUITO mais rapido. As coordenadas sao
    # reescaladas de volta antes do desenho.
    escala_processamento: float = 0.25

    # Processa 1 a cada N frames, reaproveitando o resultado anterior
    # entre eles. 1 = todo frame (lento), 3 e um bom equilibrio.
    intervalo_frames: int = 3

    # --- anotacao ---------------------------------------------------
    espessura_retangulo: int = 2
    exibir_distancia: bool = True

    def preparar(self) -> None:
        """Garante que os diretorios existem."""
        for pasta in (self.pasta_galeria, self.pasta_midia, self.pasta_saida):
            pasta.mkdir(parents=True, exist_ok=True)


config = Configuracao()
config.preparar()
