"""Selecao de motor em tempo de execucao."""

from app.config import config
from app.motores.base import ErroMotor, MotorFacial, validar_rgb
from app.motores.face_recognition_motor import MotorFaceRecognition

DISPONIVEIS = {
    "face_recognition": MotorFaceRecognition,
}

__all__ = [
    "ErroMotor",
    "MotorFacial",
    "obter_motor",
    "validar_rgb",
]


def obter_motor(nome: str | None = None) -> MotorFacial:
    """Instancia o motor configurado.

    Sobrescreva com RF_MOTOR=... quando houver alternativa.
    """
    escolhido = nome or config.motor
    classe = DISPONIVEIS.get(escolhido)
    if classe is None:
        opcoes = ", ".join(DISPONIVEIS)
        raise ValueError(
            f"Motor '{escolhido}' invalido. Opcoes: {opcoes}"
        )
    return classe()
