"""Funções puras (testáveis sem GPU): severidade, privacidade de imagem, formatação de detecções."""
from __future__ import annotations

import io
from dataclasses import asdict, dataclass

from PIL import Image, ImageOps

CLASSES = ["buraco"]

# Heurística de severidade pela área relativa da maior detecção (documentada no relatório).
# Não é um modelo treinado: serve para PRIORIZAR a triagem; a confirmação é humana (moradores/gestor).
LIMIAR_ALTO = 0.05    # buraco ocupa >= 5% da imagem
LIMIAR_MEDIO = 0.015  # >= 1,5%


@dataclass
class Deteccao:
    classe: str
    confianca: float
    bbox_xyxy: tuple[float, float, float, float]
    area_relativa: float

    def para_dict(self) -> dict:
        d = asdict(self)
        d["confianca"] = round(d["confianca"], 4)
        d["area_relativa"] = round(d["area_relativa"], 4)
        d["bbox_xyxy"] = [round(v, 1) for v in d["bbox_xyxy"]]
        return d


def severidade(deteccoes: list[Deteccao]) -> str:
    if not deteccoes:
        return "nenhuma"
    maior = max(d.area_relativa for d in deteccoes)
    if maior >= LIMIAR_ALTO or len(deteccoes) >= 4:
        return "alta"
    if maior >= LIMIAR_MEDIO or len(deteccoes) >= 2:
        return "media"
    return "baixa"


def limpar_imagem(dados: bytes, lado_max: int = 1280) -> Image.Image:
    """LGPD: reaplica a orientação, descarta TODOS os metadados (EXIF com GPS, modelo do celular, data)
    e reduz a imagem. A localização da ocorrência é informada pelo morador no mapa, não pelo EXIF."""
    img = Image.open(io.BytesIO(dados))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((lado_max, lado_max))
    limpa = Image.new("RGB", img.size)
    limpa.paste(img)
    return limpa


def resultados_para_deteccoes(resultado, largura: int, altura: int) -> list[Deteccao]:
    """Converte um `ultralytics.engine.results.Results` em lista de Deteccao."""
    dets = []
    area_img = float(largura * altura)
    for box in resultado.boxes:
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        cls = int(box.cls[0])
        nome = resultado.names.get(cls, str(cls)) if hasattr(resultado, "names") else CLASSES[cls]
        dets.append(Deteccao(nome, float(box.conf[0]), (x1, y1, x2, y2), (x2 - x1) * (y2 - y1) / area_img))
    return dets
