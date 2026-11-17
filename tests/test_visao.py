"""Testes que rodam sem GPU e sem o modelo treinado (pytest -q)."""
import io
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from visao.util import Deteccao, limpar_imagem, severidade  # noqa: E402


def _det(area):
    return Deteccao("buraco", 0.9, (0, 0, 10, 10), area)


def test_severidade():
    assert severidade([]) == "nenhuma"
    assert severidade([_det(0.005)]) == "baixa"
    assert severidade([_det(0.02)]) == "media"
    assert severidade([_det(0.005), _det(0.004)]) == "media"
    assert severidade([_det(0.08)]) == "alta"
    assert severidade([_det(0.001)] * 4) == "alta"


def test_limpar_imagem_remove_exif_e_reduz():
    img = Image.new("RGB", (3000, 2000), "gray")
    exif = Image.Exif()
    exif[0x010F] = "Fabricante do celular"   # Make
    exif[0x010E] = "Rua Exemplo 123"          # ImageDescription (simula dado sensível)
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    limpa = limpar_imagem(buf.getvalue())
    assert max(limpa.size) == 1280
    saida = io.BytesIO()
    limpa.save(saida, "JPEG")
    assert len(Image.open(io.BytesIO(saida.getvalue())).getexif()) == 0


def test_montar_dataset_unifica_classes_e_divide(tmp_path):
    fonte = tmp_path / "rf"
    for split in ("train", "valid"):
        (fonte / split / "images").mkdir(parents=True)
        (fonte / split / "labels").mkdir(parents=True)
        for i in range(10):
            Image.new("RGB", (64, 64)).save(fonte / split / "images" / f"{split}{i}.jpg")
            rot = "3 0.5 0.5 0.2 0.2\nlixo\n" if i % 3 else ""  # classe 3 → 0; linha malformada descartada
            (fonte / split / "labels" / f"{split}{i}.txt").write_text(rot)
    saida = tmp_path / "ds"
    subprocess.run([sys.executable, str(RAIZ / "scripts/montar_dataset.py"), "--fonte", f"{fonte}:publico",
                    "--saida", str(saida)], check=True, capture_output=True)
    est = json.loads((saida / "estatisticas.json").read_text())
    assert est["splits"] == {"train": 14, "val": 4, "test": 2}
    for txt in (saida / "labels").rglob("*.txt"):
        for linha in txt.read_text().splitlines():
            assert linha.startswith("0 ") and len(linha.split()) == 5
    assert (saida / "zelacidade.yaml").exists()


def test_onnx_ref_letterbox_e_decodificacao():
    import numpy as np
    from visao.onnx_ref import posprocessar, preprocessar

    img = np.zeros((480, 1280, 3), dtype=np.uint8)          # imagem larga → borda em cima e embaixo
    tensor, ganho, (bx, by) = preprocessar(img)
    assert tensor.shape == (1, 3, 640, 640) and tensor.dtype == np.float32
    assert abs(tensor[0, 0, 0, 0] - 114 / 255) < 1e-6        # borda 114 normalizada
    assert ganho == 0.5 and bx == 0 and by == 200

    # caixa real (100,100)-(300,200) na imagem original → coordenadas no quadro 640
    x1, y1, x2, y2 = 100 * ganho + bx, 100 * ganho + by, 300 * ganho + bx, 200 * ganho + by
    saida = np.zeros((1, 5, 8400), dtype=np.float32)
    for j, (dx, s) in enumerate([(0, 0.9), (1, 0.8), (0, 0.2)]):  # 2 duplicatas + 1 abaixo do limiar
        saida[0, :, j] = [(x1 + x2) / 2 + dx, (y1 + y2) / 2, x2 - x1, y2 - y1, s]
    dets = posprocessar(saida, ganho, (bx, by), 1280, 480, conf=0.35, iou_max=0.7)
    assert len(dets) == 1 and dets[0]["confianca"] == 0.9
    assert dets[0]["bbox_xyxy"] == [100.0, 100.0, 300.0, 200.0]
    assert dets[0]["area_relativa"] == round(200 * 100 / (1280 * 480), 4)


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def test_api_bate_com_esperado_do_onnx():
    """A API (.pt) segue o contrato do best.onnx: mesmas caixas, confianças e severidade do esperado.json.
    Precisa de modelos/buraco.pt e do pacote gerado por scripts/exportar_onnx.py (pulado se faltarem)."""
    import pytest

    pacote = RAIZ / "entrega_onnx"
    if not (RAIZ / "modelos/buraco.pt").exists() or not (pacote / "esperado.json").exists():
        pytest.skip("rode scripts/treinar.py e scripts/exportar_onnx.py antes")
    from fastapi.testclient import TestClient

    sys.path.insert(0, str(RAIZ))
    from api.main import CONF, app

    esperado = json.loads((pacote / "esperado.json").read_text(encoding="utf-8"))
    config = json.loads((pacote / "config.json").read_text(encoding="utf-8"))
    assert CONF == config["posprocessamento"]["limiar_confianca"]
    cliente = TestClient(app)
    for nome, esp in esperado.items():
        with open(pacote / "amostras" / nome, "rb") as f:
            r = cliente.post("/detectar", files={"imagem": (nome, f, "image/jpeg")}, params={"anotar": "false"})
        assert r.status_code == 200, r.text
        api = r.json()
        assert api["n_buracos"] == esp["n_caixas"], nome
        assert api["severidade"] == esp["severidade"], nome
        for e, a in zip(esp["deteccoes"], api["deteccoes"]):
            assert abs(e["confianca"] - a["confianca"]) < 0.02, nome
            assert _iou(e["bbox_xyxy"], a["bbox_xyxy"]) > 0.95, nome
