# FGO-OCR

Fate/Grand Order **畫面文字**用的 CPU OCR。與周回本體（FGO-Vision-Agent / FGO-GrandAutomator）分開，互不依賴。

只認封閉詞表：職階、Extra I/II、冠位研鑽戰關卡名、羅馬數字、AP。不訓通用 OCR，也不做 detection（裁切由本體 YOLO 負責）。

## 兩個模型

| 任務 | 模型 | 輸出 |
|---|---|---|
| Extra I / II、職階 tab | 9 類分類器 | `I` `II` `saber` … |
| 關卡名、AP | rec ONNX | 字串 |

推論一律 `onnxruntime` CPU。本體 GPU 留給 YOLO。

## 安裝

```powershell
cd F:\MyOwnProject\FGO-OCR
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

需要日文字型（Windows 內建即可）：

```powershell
set FGO_OCR_FONT=C:\Windows\Fonts\YuGothM.ttc
```

## 合成訓練圖

```powershell
set FGO_OCR_N=8000
python -m fgo_ocr synth
```

輸出 `data/train/*.jpg`、`data/rec_gt.txt`（Paddle rec：`相對路徑<TAB>標籤`）。

把本體 `plans/quests/*.json` 的 `label` 追加到 `assets/labels.txt`。實機 tab／關卡卡另外放到 `data/real/` 並寫標籤。

## 接到周回本體

本體只做：

```python
from fgo_ocr.infer import read
text = read(crop_bgr)
```

或設 `OCR_BACKEND=fgo` 之後再接到 `OCREngine`。`models/rec.onnx` 還沒有時不要接。

## 目錄

```
FGO-OCR/
  fgo_ocr/          合成、推論
  assets/           labels.txt、charset.txt
  data/             合成圖（不進 git）
  models/           rec.onnx（不進 git）
```
