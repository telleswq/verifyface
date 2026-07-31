# Reconhecimento Facial

Detecta rostos em fotos, identifica pessoas previamente cadastradas e
desenha retangulos anotados em video. **Projeto de estudo.**

## Stack

| Camada | Tecnologia |
|---|---|
| Deteccao e embedding | face_recognition (dlib) |
| Video e desenho | OpenCV |
| CLI | Typer |
| Qualidade | ruff, pytest |

## Pre-requisitos

- Python 3.11+
- cmake e Command Line Tools (o dlib compila C++ na instalacao)

## Instalacao

    uv venv --python 3.12
    source .venv/bin/activate
    brew install cmake
    uv pip install -r requirements-dev.txt

## Privacidade e conformidade

Dado biometrico e dado pessoal sensivel (LGPD art. 5, II) e seu
tratamento exige consentimento especifico e destacado (art. 11, I).

Decisoes de projeto derivadas disso:

- Toda a galeria fica em disco local; nada trafega para servico externo.
- `galeria/` e `midia/` estao no .gitignore - embeddings nunca sao
  versionados.
- Existe comando de exclusao total da galeria.
- O sistema so identifica pessoas explicitamente cadastradas. Rostos
  desconhecidos sao marcados como desconhecidos e nao sao persistidos.

## Limitacao conhecida - acuracia por grupo demografico

O NIST FRVT Part 3 (2019) documentou diferenciais significativos de taxa
de falso positivo entre grupos demograficos na maioria dos algoritmos
avaliados. O modelo do dlib nao e excecao.

Consequencia pratica: um limiar unico de distancia nao serve para toda
populacao. O sistema sempre expoe a distancia medida junto da
identificacao, para que o limiar possa ser calibrado com evidencia em
vez de chute.

## Aviso

Uso educacional, com pessoas que consentiram. Nao se destina a
vigilancia, identificacao de terceiros ou operacao em espaco publico.
