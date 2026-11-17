"""Implementação de referência do pré e pós-processamento do best.onnx (sem Ultralytics).

Serve de especificação para quem integra o modelo em outro backend: os resultados de
`esperado.json` (gerado por scripts/exportar_onnx.py) saem exatamente deste código.

Entrada  "images"  float32 [1, 3, 640, 640], RGB, valores 0–1 (÷255), letterbox centralizado com borda 114.
Saída    "output0" float32 [1, 5, 8400]: linhas 0–3 = cx, cy, w, h em pixels do quadro 640×640 (com a borda);
         linha 4 = score da classe 'buraco' (já passou por sigmoid, 0–1). Sem NMS.
"""
from __future__ import annotations

import numpy as np

TAMANHO = 640
COR_BORDA = 114
CONF_PADRAO = 0.35
IOU_NMS_PADRAO = 0.7  # mesmo valor padrão do predict do Ultralytics


def letterbox(img_rgb: np.ndarray, tamanho: int = TAMANHO) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Redimensiona mantendo a proporção e centraliza num quadro tamanho×tamanho com borda 114.
    Retorna (imagem, ganho, (borda_x, borda_y))."""
    import cv2

    h, w = img_rgb.shape[:2]
    ganho = min(tamanho / h, tamanho / w)
    nw, nh = round(w * ganho), round(h * ganho)
    bx, by = (tamanho - nw) / 2, (tamanho - nh) / 2
    redim = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else img_rgb
    topo, base = round(by - 0.1), round(by + 0.1)
    esq, dir_ = round(bx - 0.1), round(bx + 0.1)
    quadro = cv2.copyMakeBorder(redim, topo, base, esq, dir_, cv2.BORDER_CONSTANT, value=(COR_BORDA,) * 3)
    return quadro, ganho, (bx, by)


def preprocessar(img_rgb: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
    quadro, ganho, borda = letterbox(img_rgb)
    tensor = quadro.transpose(2, 0, 1)[None].astype(np.float32) / 255.0  # HWC → NCHW
    return np.ascontiguousarray(tensor), ganho, borda


def nms(caixas: np.ndarray, scores: np.ndarray, iou_max: float) -> list[int]:
    ordem = scores.argsort()[::-1]
    manter = []
    while ordem.size:
        i = ordem[0]
        manter.append(int(i))
        x1 = np.maximum(caixas[i, 0], caixas[ordem[1:], 0])
        y1 = np.maximum(caixas[i, 1], caixas[ordem[1:], 1])
        x2 = np.minimum(caixas[i, 2], caixas[ordem[1:], 2])
        y2 = np.minimum(caixas[i, 3], caixas[ordem[1:], 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area = lambda c: (c[..., 2] - c[..., 0]) * (c[..., 3] - c[..., 1])  # noqa: E731
        iou = inter / (area(caixas[i]) + area(caixas[ordem[1:]]) - inter + 1e-9)
        ordem = ordem[1:][iou <= iou_max]
    return manter


def posprocessar(saida: np.ndarray, ganho: float, borda: tuple[float, float], largura: int, altura: int,
                 conf: float = CONF_PADRAO, iou_max: float = IOU_NMS_PADRAO) -> list[dict]:
    """[1, 5, 8400] → lista de detecções em pixels da imagem ORIGINAL, ordenadas por confiança."""
    p = saida[0].T                      # [8400, 5]
    p = p[p[:, 4] >= conf]
    if not len(p):
        return []
    cx, cy, w, h, s = p.T
    caixas = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    caixas[:, [0, 2]] = (caixas[:, [0, 2]] - borda[0]) / ganho   # desfaz o letterbox
    caixas[:, [1, 3]] = (caixas[:, [1, 3]] - borda[1]) / ganho
    caixas[:, [0, 2]] = caixas[:, [0, 2]].clip(0, largura)
    caixas[:, [1, 3]] = caixas[:, [1, 3]].clip(0, altura)
    dets = []
    for i in nms(caixas, s, iou_max):
        x1, y1, x2, y2 = caixas[i]
        dets.append({"classe": "buraco", "confianca": round(float(s[i]), 4),
                     "bbox_xyxy": [round(float(v), 1) for v in (x1, y1, x2, y2)],
                     "area_relativa": round(float((x2 - x1) * (y2 - y1) / (largura * altura)), 4)})
    return dets
