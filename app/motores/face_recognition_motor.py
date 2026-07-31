"""Motor baseado em face_recognition (dlib).

CARACTERISTICAS:
    - Embedding de 128 dimensoes, distancia euclidiana.
    - Detector "hog" roda em CPU; "cnn" exige CUDA (indisponivel em Mac).

LIMITACOES CONHECIDAS:
    - Biblioteca sem release desde 2020.
    - Diferenciais de acuracia entre grupos demograficos documentados
      pelo NIST FRVT Part 3. Ver README.
"""

import numpy as np

from app.config import config
from app.dominio import CaixaRosto
from app.motores.base import ErroMotor, validar_rgb


class MotorFaceRecognition:
    nome = "face_recognition"
    dimensao_embedding = 128

    def __init__(
        self,
        modelo_deteccao: str | None = None,
        upsample: int | None = None,
    ) -> None:
        self.modelo_deteccao = modelo_deteccao or config.modelo_deteccao
        self.upsample = (
            upsample if upsample is not None else config.upsample_deteccao
        )
        # Import tardio: carregar os modelos custa tempo e memoria.
        # Fazendo aqui, importar o modulo continua barato e os testes
        # com motor falso nunca tocam no dlib.
        try:
            import face_recognition
        except BaseException as exc:
            # BaseException: face_recognition chama quit() quando os
            # modelos estao ausentes, levantando SystemExit.
            raise ErroMotor(
                "Nao foi possivel carregar face_recognition. Verifique "
                "se face_recognition_models esta instalado."
            ) from exc
        self._fr = face_recognition

    def detectar(self, imagem_rgb: np.ndarray) -> list[CaixaRosto]:
        validar_rgb(imagem_rgb)
        try:
            posicoes = self._fr.face_locations(
                imagem_rgb,
                number_of_times_to_upsample=self.upsample,
                model=self.modelo_deteccao,
            )
        except Exception as exc:
            raise ErroMotor(
                f"Falha na deteccao ({type(exc).__name__}): {exc}"
            ) from exc

        return [
            CaixaRosto(topo=t, direita=d, baixo=b, esquerda=e)
            for (t, d, b, e) in posicoes
        ]

    def codificar(
        self, imagem_rgb: np.ndarray, caixas: list[CaixaRosto]
    ) -> list[np.ndarray]:
        validar_rgb(imagem_rgb)
        if not caixas:
            return []

        posicoes = [
            (c.topo, c.direita, c.baixo, c.esquerda) for c in caixas
        ]
        try:
            codigos = self._fr.face_encodings(
                imagem_rgb, known_face_locations=posicoes
            )
        except Exception as exc:
            raise ErroMotor(
                f"Falha ao gerar embedding ({type(exc).__name__}): {exc}"
            ) from exc

        if len(codigos) != len(caixas):
            raise ErroMotor(
                f"Esperados {len(caixas)} embeddings, gerados "
                f"{len(codigos)}."
            )

        return [np.asarray(c, dtype=np.float64) for c in codigos]
