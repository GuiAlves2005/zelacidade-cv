# Dados

As imagens **não** ficam no Git (tamanho e licença do dataset de origem). Para reproduzir:

## 1. Dataset público
1. Acesse o **Pothole Dataset** do Roboflow Public: <https://public.roboflow.com/object-detection/pothole>
   (665 imagens, classe única `pothole`; publicado originalmente no Kaggle por Atikur Rahman Chitholian).
2. Clique em *Download* → formato **YOLOv8** → *download zip to computer*.
3. Descompacte em `dados/brutos/roboflow_pothole/` (ficam as pastas `train/`, `valid/`, `test/`).
4. Procedência e licença (conferidas na página de download e no `README.dataset.txt` do zip):

   | Campo | Valor |
   |---|---|
   | Licença | **ODbL v1.0** (Open Database License) |
   | Autor | Atikur Rahman Chitholian (<https://github.com/chitholian/Potholes-Detection>) |
   | Origem | Kaggle → Roboflow Public (redividido 70/20/10 pelo Roboflow) |
   | Link | <https://public.roboflow.com/object-detection/pothole> |
   | Conteúdo do zip (YOLOv8) | train 465 · valid 133 · test 67 imagens (665 no total, 1.740 caixas da classe `pothole`) |

## 2. Montar o dataset final
```bash
python scripts/montar_dataset.py --fonte dados/brutos/roboflow_pothole:publico --saida dados/zelacidade
```
Junta as divisões do zip, unifica a classe em `buraco`, redivide 70/20/10 com semente 42 e gera
`dados/zelacidade/{images,labels}/{train,val,test}`, `zelacidade.yaml` e `estatisticas.json`.

## 3. Vídeo de teste de tempo real
`videos/rua.mp4` (10 s, 1920×1080, 30 fps): gravado pela equipe em rua de São Paulo, só para teste de tempo real,
placa desfocada, sem áudio. Não faz parte do treino. Fica no Git junto com o vídeo anotado `resultados/demo.mp4`.
