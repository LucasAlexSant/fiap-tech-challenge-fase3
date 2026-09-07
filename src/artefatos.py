"""Persistência legível e identificação da execução."""
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(caminho: Path) -> str:
    h = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()


def salvar_json(caminho: Path, dados):
    def converter(valor):
        if isinstance(valor, np.generic):
            return valor.item()
        if isinstance(valor, Path):
            return str(valor)
        raise TypeError(type(valor).__name__)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_suffix(caminho.suffix + ".tmp")
    temporario.write_text(json.dumps(dados, ensure_ascii=False, indent=2,
                                    allow_nan=False, default=converter) + "\n", encoding="utf-8")
    temporario.replace(caminho)


def ler_json(caminho: Path):
    return json.loads(caminho.read_text(encoding="utf-8"))
