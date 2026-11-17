"""Avalia o detector no conjunto de TESTE (nunca visto no treino) e gera as evidências do relatório.

Saídas em resultados/:
  metricas.json / metricas.md     mAP50, mAP50-95, precisão, recall, F1, FPS — geral e por fonte
  val_teste/                      curvas PR/F1, matriz de confusão (geradas pelo Ultralytics)
  exemplos/                       imagens anotadas: acertos
  erros/                          falsos positivos (FP) e falsos negativos (FN) para a análise de erros

Uso:  python scripts/avaliar.py --pesos modelos/buraco.pt --conf 0.35
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "resultados"


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    uniao = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / uniao if uniao > 0 else 0.0


def gabarito(rotulo: Path, w: int, h: int) -> list[list[float]]:
    caixas = []
    if rotulo.exists():
        for l in rotulo.read_text().splitlines():
            _, cx, cy, bw, bh = map(float, l.split())
            caixas.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return caixas


def validar(modelo: YOLO, yaml_dados: Path, nome: str, conf: float) -> dict:
    m = modelo.val(data=str(yaml_dados), split="test", conf=0.001, iou=0.6, project=str(RES), name=nome,
                   exist_ok=True, plots=True, verbose=False)
    p, r = float(m.box.mp), float(m.box.mr)
    vel = m.speed
    ms = vel["preprocess"] + vel["inference"] + vel["postprocess"]
    return {"mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4),
            "precisao": round(p, 4), "recall": round(r, 4), "F1": round(2 * p * r / (p + r + 1e-9), 4),
            "ms_por_imagem": round(ms, 2), "fps": round(1000 / ms, 1) if ms else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", default=str(RAIZ / "modelos/buraco.pt"))
    ap.add_argument("--dados", default=str(RAIZ / "dados/zelacidade/zelacidade.yaml"))
    ap.add_argument("--conf", type=float, default=0.35, help="limiar de confiança usado no MVP")
    args = ap.parse_args()

    RES.mkdir(exist_ok=True)
    modelo = YOLO(args.pesos)
    cfg = yaml.safe_load(Path(args.dados).read_text(encoding="utf-8"))
    raiz_dados = Path(cfg["path"])
    imgs_teste = sorted((raiz_dados / cfg["test"]).glob("*"))

    metricas = {"geral": validar(modelo, Path(args.dados), "val_teste", args.conf)}

    # por fonte (apelido = prefixo do nome do arquivo, definido em montar_dataset.py)
    fontes = sorted({p.stem.rsplit("_", 1)[0] for p in imgs_teste})
    if len(fontes) > 1:
        for f in fontes:
            lista = RES / f"teste_{f}.txt"
            lista.write_text("\n".join(str(p) for p in imgs_teste if p.stem.startswith(f + "_")), encoding="utf-8")
            sub = dict(cfg, test=str(lista))
            sub_yaml = RES / f"teste_{f}.yaml"
            sub_yaml.write_text(yaml.safe_dump(sub, allow_unicode=True), encoding="utf-8")
            metricas[f"fonte:{f}"] = validar(modelo, sub_yaml, f"val_{f}", args.conf)

    # galeria de acertos e erros no limiar de confiança do produto
    for d in ("exemplos", "erros"):
        shutil.rmtree(RES / d, ignore_errors=True)
        (RES / d).mkdir()
    contagem = {"TP": 0, "FP": 0, "FN": 0}
    for img_path in imgs_teste:
        pred = modelo.predict(str(img_path), conf=args.conf, verbose=False)[0]
        h, w = pred.orig_shape
        gt = gabarito(raiz_dados / "labels" / "test" / f"{img_path.stem}.txt", w, h)
        pr = [b.xyxy[0].tolist() for b in pred.boxes]
        usados, tp = set(), 0
        for p in pr:
            melhor = max(range(len(gt)), key=lambda i: iou(p, gt[i]), default=None)
            if melhor is not None and melhor not in usados and iou(p, gt[melhor]) >= 0.5:
                usados.add(melhor); tp += 1
        fp, fn = len(pr) - tp, len(gt) - tp
        contagem["TP"] += tp; contagem["FP"] += fp; contagem["FN"] += fn
        anot = pred.plot()
        for g in gt:  # gabarito em verde
            cv2.rectangle(anot, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), (0, 200, 0), 2)
        if fp or fn:
            cv2.imwrite(str(RES / "erros" / f"{img_path.stem}_FP{fp}_FN{fn}.jpg"), anot)
        elif tp and len(list((RES / "exemplos").iterdir())) < 12:
            cv2.imwrite(str(RES / "exemplos" / f"{img_path.stem}.jpg"), anot)
    metricas["contagem_conf_produto"] = {"conf": args.conf, **contagem}

    (RES / "metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
    linhas = ["| Conjunto | mAP50 | mAP50-95 | Precisão | Recall | F1 | ms/img | FPS |", "|---|---|---|---|---|---|---|---|"]
    for k, v in metricas.items():
        if k.startswith("contagem"):
            continue
        linhas.append(f"| {k} | {v['mAP50']:.3f} | {v['mAP50_95']:.3f} | {v['precisao']:.3f} | {v['recall']:.3f} | "
                      f"{v['F1']:.3f} | {v['ms_por_imagem']} | {v['fps']} |")
    linhas.append(f"\nNo limiar de confiança {args.conf}: TP={contagem['TP']} · FP={contagem['FP']} · FN={contagem['FN']}")
    (RES / "metricas.md").write_text("\n".join(linhas), encoding="utf-8")
    print("\n".join(linhas))


if __name__ == "__main__":
    main()
