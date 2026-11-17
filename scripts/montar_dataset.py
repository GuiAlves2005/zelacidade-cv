"""Junta uma ou mais fontes em formato YOLO, unifica a classe em 'buraco' e divide treino/val/teste.

Cada fonte é uma pasta com subpastas de imagens e rótulos YOLO (.txt), em qualquer nível
(ex.: o zip do Roboflow tem train/images, train/labels, valid/..., test/...). Pares sem rótulo
são tratados como imagens NEGATIVAS (sem buraco), úteis para reduzir falsos positivos —
coloque fotos de asfalto bom, bueiros, sombras e manchas com um .txt vazio.

A divisão é feita por fonte com semente fixa (70/20/10), então cada fonte tem uma parte no TESTE —
com mais de uma fonte, isso permite medir o desempenho separado por fonte.

Uso:
  python scripts/montar_dataset.py --fonte dados/brutos/roboflow_pothole:publico --saida dados/zelacidade
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml

EXT_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def pares(pasta: Path) -> list[tuple[Path, Path | None]]:
    imgs = [p for p in pasta.rglob("*") if p.suffix.lower() in EXT_IMG]
    saida = []
    for img in imgs:
        # rótulo correspondente: troca a pasta 'images' por 'labels' e a extensão por .txt
        partes = list(img.parts)
        if "images" in partes:
            partes[len(partes) - 1 - partes[::-1].index("images")] = "labels"
        rot = Path(*partes).with_suffix(".txt")
        saida.append((img, rot if rot.exists() else None))
    return saida


def unificar_rotulo(origem: Path | None) -> tuple[str, int]:
    """Força todas as classes para 0 ('buraco'). Descarta linhas malformadas."""
    if origem is None:
        return "", 0
    linhas = []
    for l in origem.read_text(encoding="utf-8").splitlines():
        v = l.split()
        if len(v) == 5 and all(0 <= float(x) <= 1 for x in v[1:]):
            linhas.append("0 " + " ".join(v[1:]))
    return "\n".join(linhas) + ("\n" if linhas else ""), len(linhas)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonte", action="append", required=True, help="pasta:apelido")
    ap.add_argument("--saida", type=Path, default=Path("dados/zelacidade"))
    ap.add_argument("--proporcao", nargs=3, type=float, default=[0.7, 0.2, 0.1])
    ap.add_argument("--semente", type=int, default=42)
    args = ap.parse_args()

    if args.saida.exists():  # esvazia sem apagar a pasta em si (ela pode ser uma junção para outro disco)
        for filho in args.saida.iterdir():
            shutil.rmtree(filho) if filho.is_dir() and not filho.is_symlink() else filho.unlink()
    for split in ("train", "val", "test"):
        (args.saida / "images" / split).mkdir(parents=True)
        (args.saida / "labels" / split).mkdir(parents=True)

    rng = random.Random(args.semente)
    estat = {"fontes": {}, "splits": Counter(), "caixas": Counter(), "negativas": Counter()}
    for item in args.fonte:
        pasta, apelido = item.rsplit(":", 1)
        lista = pares(Path(pasta))
        rng.shuffle(lista)
        n = len(lista)
        n_tr, n_va = int(n * args.proporcao[0]), int(n * args.proporcao[1])
        cortes = {"train": lista[:n_tr], "val": lista[n_tr:n_tr + n_va], "test": lista[n_tr + n_va:]}
        estat["fontes"][apelido] = {"pasta": pasta, "imagens": n, **{k: len(v) for k, v in cortes.items()}}
        for split, itens in cortes.items():
            for i, (img, rot) in enumerate(itens):
                nome = f"{apelido}_{i:05d}"
                shutil.copy2(img, args.saida / "images" / split / f"{nome}{img.suffix.lower()}")
                texto, n_caixas = unificar_rotulo(rot)
                (args.saida / "labels" / split / f"{nome}.txt").write_text(texto, encoding="utf-8")
                estat["splits"][split] += 1
                estat["caixas"][split] += n_caixas
                estat["negativas"][split] += n_caixas == 0

    cfg = {"path": str(args.saida.resolve()), "train": "images/train", "val": "images/val", "test": "images/test",
           "names": {0: "buraco"}}
    (args.saida / "zelacidade.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    estat = {k: dict(v) if isinstance(v, Counter) else v for k, v in estat.items()}
    (args.saida / "estatisticas.json").write_text(json.dumps(estat, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(estat, indent=2, ensure_ascii=False))
    print(f"Configuração do dataset: {args.saida / 'zelacidade.yaml'}")


if __name__ == "__main__":
    main()
