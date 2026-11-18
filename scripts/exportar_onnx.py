"""Gera o pacote de integração do detector para o backend do MVP.

Saída em entrega_onnx/:
  best.onnx        imgsz 640, opset 17, entrada estática [1,3,640,640], saída bruta [1,5,8400] (sem NMS)
  classes.json     {"0": "buraco"}
  config.json      limiar de confiança, IoU do NMS, pré-processamento e regra de severidade
  amostras/        5 imagens do conjunto de teste
  esperado.json    resultado esperado de cada amostra (caixas, confiança, área, severidade)
  anotadas/        as 5 amostras desenhadas, para conferência visual

O esperado.json é calculado rodando o PRÓPRIO best.onnx no onnxruntime com src/visao/onnx_ref.py,
e comparado com a predição do .pt no Ultralytics (mesmo número de caixas, IoU ≥ 0,9).

Uso:  python scripts/exportar_onnx.py --pesos modelos/buraco.pt --conf 0.35
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from visao.onnx_ref import COR_BORDA, IOU_NMS_PADRAO, TAMANHO, letterbox, posprocessar, preprocessar  # noqa: E402
from visao.util import LIMIAR_ALTO, LIMIAR_MEDIO, Deteccao, severidade  # noqa: E402


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def escolher_amostras(pasta_img: Path, pasta_rot: Path, n: int = 5) -> list[Path]:
    """Mistura de casos: sem buraco, 1 buraco, vários buracos — para testar o NMS e a severidade."""
    por_qtd: dict[str, list[Path]] = {"0": [], "1": [], "2+": []}
    for p in sorted(pasta_img.glob("*")):
        rot = pasta_rot / f"{p.stem}.txt"
        k = len(rot.read_text().splitlines()) if rot.exists() else 0
        por_qtd["0" if k == 0 else "1" if k == 1 else "2+"].append(p)
    ordem = [("2+", 2), ("1", 2), ("0", 1)]
    escolhidas = [p for chave, q in ordem for p in por_qtd[chave][:q]]
    resto = [p for lst in por_qtd.values() for p in lst if p not in escolhidas]
    return (escolhidas + resto)[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", default=str(RAIZ / "modelos/buraco.pt"))
    ap.add_argument("--dados", default=str(RAIZ / "dados/zelacidade/zelacidade.yaml"))
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--saida", default=str(RAIZ / "entrega_onnx"))
    args = ap.parse_args()

    import onnxruntime as ort
    from ultralytics import YOLO

    saida = Path(args.saida)
    shutil.rmtree(saida, ignore_errors=True)
    (saida / "amostras").mkdir(parents=True)
    (saida / "anotadas").mkdir()

    # 1) exportar
    modelo = YOLO(args.pesos)
    caminho_onnx = Path(modelo.export(format="onnx", imgsz=TAMANHO, opset=args.opset, dynamic=False,
                                      simplify=True, half=False))
    shutil.copy2(caminho_onnx, saida / "best.onnx")
    sessao = ort.InferenceSession(str(saida / "best.onnx"), providers=["CPUExecutionProvider"])
    ent, sai = sessao.get_inputs()[0], sessao.get_outputs()[0]
    assert list(sai.shape) == [1, 5, 8400], f"saída inesperada {sai.shape} — confira se o modelo tem 1 classe"

    # 2) metadados
    (saida / "classes.json").write_text(json.dumps({"0": "buraco"}, ensure_ascii=False), encoding="utf-8")
    config = {
        "modelo": Path(args.pesos).name, "opset": args.opset,
        "entrada": {"nome": ent.name, "forma": ent.shape, "tipo": "float32"},
        "saida": {"nome": sai.name, "forma": sai.shape,
                  "layout": "linhas 0-3 = cx, cy, w, h em pixels do quadro 640 (com borda); linha 4 = score 0-1 (sigmoid já aplicado); sem NMS"},
        "preprocessamento": {"letterbox": TAMANHO, "centralizado": True, "cor_borda": COR_BORDA,
                             "redimensionamento": "bilinear, mantendo proporção, ganho = min(640/h, 640/w)",
                             "canais": "RGB", "normalizacao": "÷255", "layout": "NCHW"},
        "posprocessamento": {"limiar_confianca": args.conf, "iou_nms": IOU_NMS_PADRAO,
                             "desfazer_letterbox": "x = (x - borda_x) / ganho; y = (y - borda_y) / ganho; recortar aos limites da imagem"},
        "severidade": {"area_relativa": "(x2-x1)*(y2-y1) / (largura*altura) na imagem original",
                       "regra": f"nenhuma se 0 caixas; alta se maior área ≥ {LIMIAR_ALTO} ou ≥ 4 caixas; "
                                f"media se maior área ≥ {LIMIAR_MEDIO} ou ≥ 2 caixas; senão baixa"},
    }
    (saida / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    # 3) amostras e resultado esperado
    cfg = yaml.safe_load(Path(args.dados).read_text(encoding="utf-8"))
    raiz = Path(cfg["path"])
    amostras = escolher_amostras(raiz / cfg["test"], raiz / "labels" / "test")
    esperado, divergencias = {}, []
    for i, p in enumerate(amostras, 1):
        nome = f"teste_{i}{p.suffix.lower()}"
        shutil.copy2(p, saida / "amostras" / nome)
        bgr = cv2.imread(str(p))
        h, w = bgr.shape[:2]
        tensor, ganho, borda = preprocessar(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        bruto = sessao.run([sai.name], {ent.name: tensor})[0]
        dets = posprocessar(bruto, ganho, borda, w, h, args.conf, IOU_NMS_PADRAO)
        sev = severidade([Deteccao(d["classe"], d["confianca"], tuple(d["bbox_xyxy"]), d["area_relativa"]) for d in dets])
        esperado[nome] = {"largura": w, "altura": h, "n_caixas": len(dets), "severidade": sev, "deteccoes": dets}

        # conferência com o Ultralytics (.pt) recebendo o MESMO quadro 640×640 do contrato
        # (o predict padrão usa letterbox retangular, ex. 640×384, e não serviria de referência)
        quadro, _, _ = letterbox(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ref = modelo.predict(cv2.cvtColor(quadro, cv2.COLOR_RGB2BGR), conf=args.conf, iou=IOU_NMS_PADRAO,
                             imgsz=TAMANHO, verbose=False)[0]
        ref_caixas = [[min(max((v - borda[k % 2]) / ganho, 0), (w, h)[k % 2]) for k, v in enumerate(b.xyxy[0].tolist())]
                      for b in ref.boxes]
        if len(ref_caixas) != len(dets) or any(max((iou(d["bbox_xyxy"], r) for r in ref_caixas), default=0) < 0.9
                                              for d in dets):
            divergencias.append(nome)

        anot = bgr.copy()
        for d in dets:
            x1, y1, x2, y2 = map(int, d["bbox_xyxy"])
            cv2.rectangle(anot, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(anot, f"buraco {d['confianca']:.2f}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(anot, f"severidade: {sev}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(str(saida / "anotadas" / f"teste_{i}.jpg"), anot)

    (saida / "esperado.json").write_text(json.dumps(esperado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(config["entrada"] | {"saida": config["saida"]["forma"]}, ensure_ascii=False))
    for nome, e in esperado.items():
        print(f"{nome}: {e['n_caixas']} caixa(s), severidade {e['severidade']}")
    print("Conferência ONNX × Ultralytics:", "OK" if not divergencias else f"divergiu em {divergencias}")
    print(f"Pacote em {saida}")


if __name__ == "__main__":
    main()
