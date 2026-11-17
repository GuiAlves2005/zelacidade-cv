"""API de visão consumida pelo MVP da ZelaCidade.

Rodar:  uvicorn api.main:app --host 0.0.0.0 --port 8002

POST /detectar  (multipart, campo "imagem")
  → {"detectado": true, "n_buracos": 2, "severidade": "media", "categoria_sugerida": "buraco_via",
     "deteccoes": [{"classe": "buraco", "confianca": 0.87, "bbox_xyxy": [...], "area_relativa": 0.031}],
     "imagem_anotada_base64": "...", "modelo": "buraco.pt", "latencia_ms": 41.3}

Fluxo no MVP: o morador envia a foto → /detectar sugere a categoria "buraco_via" e a severidade →
o morador confirma ou corrige (pessoa no circuito) → a ocorrência é gravada → a Zela (IA generativa) orienta.

LGPD: a imagem é processada em memória, os metadados EXIF (incl. GPS) são descartados, nada é gravado em disco.
"""
from __future__ import annotations

import base64
import io
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import secrets  # noqa: E402

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from visao.onnx_ref import IOU_NMS_PADRAO, TAMANHO, letterbox  # noqa: E402
from visao.util import Deteccao, limpar_imagem, severidade  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
PESOS = os.getenv("VISAO_PESOS", str(RAIZ / "modelos/buraco.pt"))
CONF = float(os.getenv("VISAO_CONF", "0.35"))
MAX_BYTES = 8 * 1024 * 1024

modelo = YOLO(PESOS)


@asynccontextmanager
async def ciclo_de_vida(_app: FastAPI):
    """Aquece o modelo ao subir: a 1ª inferência na GPU inicializa o CUDA e levaria segundos
    na 1ª foto do morador. Usa o mesmo caminho das requisições reais (letterbox 640×640).
    A imagem preta não gera candidatos, então uma passada com ruído e conf baixa inicializa também o NMS."""
    preta = np.zeros((TAMANHO, TAMANHO, 3), np.uint8)
    for _ in range(3):
        detectar_contrato(preta)
    ruido = np.random.default_rng(0).integers(0, 256, (TAMANHO, TAMANHO, 3), dtype=np.uint8)
    modelo.predict(ruido, conf=0.001, iou=IOU_NMS_PADRAO, imgsz=TAMANHO, verbose=False)
    yield


app = FastAPI(title="ZelaCidade — Visão (detecção de buracos)", version="1.0.0", lifespan=ciclo_de_vida)
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("VISAO_CORS", "*").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["*"])


API_KEY = os.getenv("VISAO_API_KEY", "")


def verificar_chave(x_api_key: str | None = Header(default=None)) -> None:
    """Se VISAO_API_KEY estiver definida, exige o cabeçalho X-API-Key igual a ela."""
    if API_KEY and not (x_api_key and secrets.compare_digest(x_api_key, API_KEY)):
        raise HTTPException(status_code=401, detail="X-API-Key ausente ou inválida.")


@app.get("/saude")
def saude() -> dict:
    return {"status": "ok", "modelo": Path(PESOS).name, "classes": list(modelo.names.values()), "conf": CONF}


@app.post("/detectar", dependencies=[Depends(verificar_chave)])
async def detectar(imagem: UploadFile = File(...), anotar: bool = True) -> dict:
    if imagem.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, "Envie JPG, PNG ou WEBP.")
    dados = await imagem.read()
    if len(dados) > MAX_BYTES:
        raise HTTPException(413, "Imagem maior que 8 MB.")
    try:
        img = limpar_imagem(dados)
    except Exception:
        raise HTTPException(400, "Arquivo de imagem inválido.")

    t0 = time.perf_counter()
    dets = detectar_contrato(np.array(img))
    saida = {
        "detectado": bool(dets),
        "n_buracos": len(dets),
        "severidade": severidade(dets),
        "categoria_sugerida": "buraco_via" if dets else None,
        "deteccoes": [d.para_dict() for d in dets],
        "modelo": Path(PESOS).name,
        "latencia_ms": round((time.perf_counter() - t0) * 1000, 1),
        "aviso": "Sugestão automática: confirme a categoria antes de registrar.",
    }
    if anotar:
        buf = io.BytesIO()
        anotar_imagem(img, dets).save(buf, "JPEG", quality=85)
        saida["imagem_anotada_base64"] = base64.b64encode(buf.getvalue()).decode()
    return saida


def detectar_contrato(rgb: np.ndarray) -> list[Deteccao]:
    """Mesmo pré-processamento do best.onnx (contrato com o backend): letterbox 640×640 centralizado,
    borda 114, conf CONF, NMS IoU 0,7. As caixas voltam para os pixels da imagem original."""
    h, w = rgb.shape[:2]
    quadro, ganho, (bx, by) = letterbox(rgb)
    res = modelo.predict(np.ascontiguousarray(quadro[:, :, ::-1]), conf=CONF, iou=IOU_NMS_PADRAO,  # RGB → BGR
                         imgsz=TAMANHO, verbose=False)[0]
    dets = []
    for box in res.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, x2 = (min(max((v - bx) / ganho, 0.0), w) for v in (x1, x2))
        y1, y2 = (min(max((v - by) / ganho, 0.0), h) for v in (y1, y2))
        dets.append(Deteccao(modelo.names[int(box.cls[0])], float(box.conf[0]), (x1, y1, x2, y2),
                             (x2 - x1) * (y2 - y1) / float(w * h)))
    return sorted(dets, key=lambda d: d.confianca, reverse=True)


def anotar_imagem(img, dets: list[Deteccao]):
    from PIL import ImageDraw

    anot = img.copy()
    desenho = ImageDraw.Draw(anot)
    espessura = max(2, round(max(img.size) / 400))
    for d in dets:
        x1, y1, x2, y2 = d.bbox_xyxy
        desenho.rectangle((x1, y1, x2, y2), outline=(255, 40, 40), width=espessura)
        desenho.text((x1 + 3, max(0, y1 - 12)), f"{d.classe} {d.confianca:.2f}", fill=(255, 40, 40))
    return anot
