"""Detecção de buracos em TEMPO REAL — webcam, vídeo gravado ou stream (celular como câmera IP).

Mostra caixas, confiança, severidade estimada e FPS na tela. Opcionalmente grava o vídeo anotado
(para o vídeo de demonstração) e registra os quadros com buraco num CSV.

Uso:
  python scripts/tempo_real.py --fonte 0                                  # webcam
  python scripts/tempo_real.py --fonte videos/rua_tucuruvi.mp4 --gravar resultados/demo.mp4
  python scripts/tempo_real.py --fonte http://192.168.0.10:8080/video     # app "IP Webcam" no celular
Teclas: q = sair · espaço = pausar · s = salvar quadro
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from visao.util import resultados_para_deteccoes, severidade  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
CORES = {"nenhuma": (160, 160, 160), "baixa": (0, 200, 255), "media": (0, 140, 255), "alta": (0, 0, 255)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonte", default="0")
    ap.add_argument("--pesos", default=str(RAIZ / "modelos/buraco.pt"))
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--gravar", default=None, help="caminho do .mp4 anotado")
    ap.add_argument("--sem-janela", action="store_true", help="processa sem abrir janela (servidor)")
    args = ap.parse_args()

    modelo = YOLO(args.pesos)
    import numpy as np
    for _ in range(3):  # aquecimento: a 1ª inferência na GPU inicializa o CUDA e distorceria o FPS médio
        modelo.predict(np.zeros((args.imgsz, args.imgsz, 3), np.uint8), conf=args.conf, imgsz=args.imgsz, verbose=False)
    fonte =int(args.fonte) if args.fonte.isdigit() else args.fonte
    cap = cv2.VideoCapture(fonte)
    if not cap.isOpened():
        raise SystemExit(f"Não consegui abrir a fonte {args.fonte!r}")

    gravador = None
    log_path = RAIZ / "resultados" / "tempo_real_log.csv"
    log_path.parent.mkdir(exist_ok=True)
    log = open(log_path, "w", newline="", encoding="utf-8")
    escritor = csv.writer(log)
    escritor.writerow(["quadro", "t_s", "n_buracos", "conf_max", "severidade", "fps"])

    quadro, fps_suave, inicio, pausado = 0, 0.0, time.perf_counter(), False
    tempo_inferencia = 0.0  # soma do predict em todos os quadros → FPS médio real no fim
    while True:
        if not pausado:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            res = modelo.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            dt = time.perf_counter() - t0
            tempo_inferencia += dt
            fps_suave = 0.9 * fps_suave + 0.1 * (1 / dt) if fps_suave else 1 / dt  # média móvel, só para a tela
            h, w = frame.shape[:2]
            dets = resultados_para_deteccoes(res, w, h)
            sev = severidade(dets)
            anot = res.plot(line_width=2)
            cv2.rectangle(anot, (0, 0), (w, 42), (0, 0, 0), -1)
            texto = f"ZelaCidade | buracos: {len(dets)} | severidade: {sev} | {fps_suave:.1f} FPS"
            cv2.putText(anot, texto, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.8, CORES[sev], 2)
            if dets:
                escritor.writerow([quadro, round(time.perf_counter() - inicio, 2), len(dets),
                                   round(max(d.confianca for d in dets), 3), sev, round(fps_suave, 1)])
            if args.gravar:
                if gravador is None:
                    Path(args.gravar).parent.mkdir(parents=True, exist_ok=True)
                    fps_vid = cap.get(cv2.CAP_PROP_FPS) or 25
                    gravador = cv2.VideoWriter(args.gravar, cv2.VideoWriter_fourcc(*"mp4v"), fps_vid, (w, h))
                gravador.write(anot)
            quadro += 1
        if not args.sem_janela:
            cv2.imshow("ZelaCidade - deteccao de buracos (q para sair)", anot)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            if tecla == ord(" "):
                pausado = not pausado
            if tecla == ord("s"):
                cv2.imwrite(str(RAIZ / "resultados" / f"quadro_{quadro:05d}.jpg"), anot)

    cap.release()
    log.close()
    if gravador:
        gravador.release()
    cv2.destroyAllWindows()
    fps_medio = quadro / tempo_inferencia if tempo_inferencia else 0.0
    print(f"{quadro} quadros processados · FPS médio de inferência (todos os quadros) = {fps_medio:.1f} "
          f"({1000 / fps_medio if fps_medio else 0:.1f} ms/quadro) · log em {log_path.relative_to(RAIZ).as_posix()}")
    print("Para o FPS do pipeline completo (ler + detectar + desenhar + gravar), use scripts/medir_fps.py.")


if __name__ == "__main__":
    main()
