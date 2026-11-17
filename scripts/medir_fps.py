"""Mede o FPS do detector num vídeo, em TODOS os quadros, com o mesmo predict do tempo_real.py.

O tempo_real.py mostra na tela uma média móvel do FPS de inferência; este script grava números
reprodutíveis para o relatório em resultados/fps_video.json:
  - inferência: só o modelo.predict (pré-processamento + rede + NMS), como no tempo_real.py;
  - pipeline:   ler o quadro + predict + desenhar as caixas + gravar o vídeo anotado.
Os 10 primeiros predicts servem de aquecimento da GPU e não entram na média.

Uso:  python scripts/medir_fps.py --fonte videos/rua.mp4
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonte", default=str(RAIZ / "videos/rua.mp4"))
    ap.add_argument("--pesos", default=str(RAIZ / "modelos/buraco.pt"))
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--saida", default=str(RAIZ / "resultados/fps_video.json"))
    args = ap.parse_args()

    modelo = YOLO(args.pesos)
    cap = cv2.VideoCapture(args.fonte)
    if not cap.isOpened():
        raise SystemExit(f"Não consegui abrir {args.fonte!r}")
    w, h, fps_video = int(cap.get(3)), int(cap.get(4)), cap.get(5) or 30
    ok, quadro = cap.read()
    for _ in range(10):
        modelo.predict(quadro, conf=args.conf, imgsz=args.imgsz, verbose=False)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    inferencia, pipeline = [], []
    with tempfile.TemporaryDirectory() as tmp:
        gravador = cv2.VideoWriter(str(Path(tmp) / "descartavel.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps_video, (w, h))
        while True:
            t0 = time.perf_counter()
            ok, quadro = cap.read()
            if not ok:
                break
            t1 = time.perf_counter()
            res = modelo.predict(quadro, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            t2 = time.perf_counter()
            gravador.write(res.plot(line_width=2))
            t3 = time.perf_counter()
            inferencia.append(t2 - t1)
            pipeline.append(t3 - t0)
        gravador.release()
    cap.release()

    inf, tot = np.array(inferencia), np.array(pipeline)
    resultado = {
        "video": Path(args.fonte).name, "resolucao": f"{w}x{h}", "fps_do_video": round(fps_video, 2), "quadros": len(inf),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", "conf": args.conf, "imgsz": args.imgsz,
        "inferencia_ms_media": round(inf.mean() * 1000, 2), "inferencia_ms_p95": round(float(np.percentile(inf, 95)) * 1000, 2),
        "fps_inferencia": round(1 / inf.mean(), 1),
        "pipeline_ms_media": round(tot.mean() * 1000, 2), "fps_pipeline": round(1 / tot.mean(), 1),
    }
    Path(args.saida).write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
