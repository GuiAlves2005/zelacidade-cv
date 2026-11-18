"""Treino customizado do detector de buracos (fine-tuning de YOLO pré-treinado no COCO).

Por que fine-tuning? O modelo já sabe extrair bordas, texturas e formas de milhões de imagens;
só ensinamos a nova classe 'buraco' com algumas centenas de exemplos — converge em poucos minutos numa GPU de 8 GB.

Uso:
  python scripts/treinar.py                                   # yolo11n, 100 épocas, 640 px
  python scripts/treinar.py --modelo yolo11s.pt --epocas 150  # modelo maior (mais preciso, mais lento)
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parents[1]


def relativo(caminho: str | Path) -> str:
    """Caminho relativo à raiz do projeto, com '/' (o repositório é público: nada de caminho absoluto da máquina)."""
    p = Path(caminho)
    for base in (RAIZ, RAIZ.resolve()):
        try:
            return p.relative_to(base).as_posix()
        except ValueError:
            pass
    return p.name if p.is_absolute() else p.as_posix()


def anonimizar_pesos(caminho: Path, nome_run: str) -> None:
    """Troca os caminhos absolutos que o Ultralytics grava em train_args do checkpoint por caminhos relativos.
    Só metadados: os pesos não mudam. O ONNX exportado depois herda esses metadados já limpos."""
    import torch

    ckpt = torch.load(caminho, map_location="cpu", weights_only=False)
    dicionarios = [ckpt.get("train_args")] + [getattr(ckpt.get(k), "args", None) for k in ("model", "ema")]
    for args in (d for d in dicionarios if isinstance(d, dict)):
        args.update(data="dados/zelacidade/zelacidade.yaml", project="runs", save_dir=f"runs/{nome_run}")
        for k, v in args.items():
            if isinstance(v, str) and (":\\" in v or ":/" in v) and not v.startswith("http"):
                args[k] = relativo(v)
    torch.save(ckpt, caminho)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", default=str(RAIZ / "dados/zelacidade/zelacidade.yaml"))
    ap.add_argument("--modelo", default="yolo11n.pt", help="pesos pré-treinados (yolo11n/s/m, yolov8n...)")
    ap.add_argument("--epocas", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lote", type=int, default=16, help="reduza para 8 se faltar memória de GPU")
    ap.add_argument("--workers", type=int, default=4,
                    help="processos de carga de imagens; no Windows cada um reimporta o torch (RAM/pagefile)")
    ap.add_argument("--nome", default="buraco_yolo11n")
    args = ap.parse_args()

    modelo = YOLO(args.modelo)
    modelo.train(
        data=args.dados,
        epochs=args.epocas,
        imgsz=args.imgsz,
        batch=args.lote,
        workers=args.workers,
        patience=25,          # early stopping se o mAP de validação parar de subir
        project=str(RAIZ / "runs"),
        name=args.nome,
        exist_ok=True,
        seed=42,
        deterministic=True,
        # aumentos de dados coerentes com fotos de rua tiradas de celular
        hsv_v=0.4, degrees=5.0, translate=0.1, scale=0.5, fliplr=0.5, mosaic=1.0,
        plots=True,
    )
    melhor = RAIZ / "runs" / args.nome / "weights" / "best.pt"
    destino = RAIZ / "modelos" / "buraco.pt"
    destino.parent.mkdir(exist_ok=True)
    shutil.copy2(melhor, destino)
    anonimizar_pesos(destino, args.nome)
    registro = dict(vars(args), dados=relativo(args.dados))
    (RAIZ / "modelos" / "treino.json").write_text(json.dumps(registro, indent=2), encoding="utf-8")
    print(f"Melhor peso copiado para {destino}")


if __name__ == "__main__":
    main()
