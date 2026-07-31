"""Verifica o ambiente macOS antes de rodar o pipeline.

Existe porque as falhas mais caras deste projeto no macOS sao
SILENCIOSAS: sem permissao de camera o OpenCV entrega frames pretos sem
levantar excecao, e o detector CNN "funciona" porem em velocidade
inutilizavel por ausencia de CUDA.
"""

import platform
import shutil
import sys

OK = "\033[92m OK \033[0m"
AVISO = "\033[93mAVISO\033[0m"
FALHA = "\033[91mFALHA\033[0m"


def linha(marca: str, titulo: str, detalhe: str = "") -> None:
    print(f"[{marca}] {titulo:<34} {detalhe}")


def checar_plataforma() -> str:
    arquitetura = platform.machine()
    macos = platform.mac_ver()[0] or "desconhecida"
    if platform.system() != "Darwin":
        linha(AVISO, "Sistema", f"{platform.system()} - script e para macOS")
        return arquitetura
    rotulo = "Apple Silicon" if arquitetura == "arm64" else "Intel"
    linha(OK, "Sistema", f"macOS {macos} - {rotulo} ({arquitetura})")
    return arquitetura


def checar_homebrew(arquitetura: str) -> None:
    caminho = shutil.which("brew")
    if not caminho:
        linha(FALHA, "Homebrew", "nao encontrado - instale em brew.sh")
        return
    esperado = "/opt/homebrew" if arquitetura == "arm64" else "/usr/local"
    marca = OK if caminho.startswith(esperado) else AVISO
    linha(marca, "Homebrew", caminho)


def checar_cmake() -> None:
    if shutil.which("cmake"):
        linha(OK, "cmake", "presente (necessario ao dlib)")
    else:
        linha(FALHA, "cmake", "ausente - rode: brew install cmake")


def checar_bibliotecas() -> None:
    for modulo, rotulo in [
        ("numpy", "numpy"),
        ("cv2", "OpenCV"),
        ("dlib", "dlib"),
        ("face_recognition", "face_recognition"),
    ]:
        try:
            mod = __import__(modulo)
            versao = getattr(mod, "__version__", "importado")
            linha(OK, rotulo, str(versao))
        except BaseException as exc:
            # BaseException, nao Exception: face_recognition chama quit()
            # no import quando os modelos estao ausentes, levantando
            # SystemExit. Com except Exception o script inteiro morria
            # aqui, escondendo as verificacoes seguintes.
            if isinstance(exc, SystemExit):
                detalhe = "falta face_recognition_models"
            else:
                detalhe = f"{type(exc).__name__}: {str(exc)[:40]}"
            linha(FALHA, rotulo, detalhe)


def checar_cuda() -> None:
    """Confirma a ausencia de CUDA - esperada em qualquer Mac."""
    try:
        import dlib

        if dlib.DLIB_USE_CUDA:
            linha(OK, "CUDA", "disponivel (incomum em macOS)")
        else:
            linha(
                AVISO,
                "CUDA",
                "ausente (normal) - use modelo_deteccao=hog",
            )
    except Exception:
        linha(AVISO, "CUDA", "dlib indisponivel para verificar")


def checar_camera() -> None:
    """Tenta capturar um frame real e detectar imagem toda preta."""
    try:
        import cv2
        import numpy as np
    except Exception:
        linha(FALHA, "Camera", "OpenCV indisponivel")
        return

    captura = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not captura.isOpened():
        captura.release()
        linha(
            FALHA,
            "Camera",
            "nao abriu - verifique Privacidade > Camera",
        )
        return

    ok, frame = captura.read()
    captura.release()

    if not ok or frame is None:
        linha(FALHA, "Camera", "abriu mas nao entregou frame")
        return

    altura, largura = frame.shape[:2]
    if float(np.mean(frame)) < 1.0:
        linha(
            FALHA,
            "Camera",
            f"{largura}x{altura} porem TODA PRETA - falta permissao",
        )
        return

    linha(OK, "Camera", f"{largura}x{altura}, imagem valida")


def listar_cameras(maximo: int = 4) -> None:
    """Lista indices disponiveis.

    Util quando a Continuity Camera do iPhone assume o indice 0 e rouba
    a captura da webcam interna.
    """
    try:
        import cv2
    except Exception:
        return

    achados = []
    for indice in range(maximo):
        captura = cv2.VideoCapture(indice, cv2.CAP_AVFOUNDATION)
        if captura.isOpened():
            largura = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
            altura = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
            achados.append(f"{indice}:{largura}x{altura}")
        captura.release()

    detalhe = ", ".join(achados) if achados else "nenhum"
    linha(OK if achados else FALHA, "Indices de camera", detalhe)


def main() -> int:
    print("\n=== Preflight macOS ===\n")
    arquitetura = checar_plataforma()
    checar_homebrew(arquitetura)
    checar_cmake()
    print()
    checar_bibliotecas()
    checar_cuda()
    print()
    checar_camera()
    listar_cameras()
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
