# Connector Detection — AI + AOI 連接器瑕疵檢測系統

針對 AOI（自動光學檢測）輸出影像，自動偵測連接器上的以下瑕疵：

- **Pin 異物 / 髒污**：pin 上出現雜質或污染物
- **Pin 易位**：連接器位移導致 pin 排列錯誤
- **遮擋**：連接器整體偏移，造成部分 pin 被遮蓋

---

## 推論流程總覽

```
AOI 截圖
  │
  ▼
[YOLO #1]  →  Connector Crop（排除周邊其他元件）
  │
  ▼
[DINOv2 + kNN]  →  型號分類（已知型號 / Unknown）
  │
  ▼ (僅目標型號)
[YOLO #2]  →  pin_row crop  +  metal_body crop
  │
  ▼
[方向正規化]  →  統一為正位（pin 排在上、金屬本體在下）
  │
  ├──▶ [PatchCore — pin_row]   →  OK / NG
  └──▶ [PatchCore — metal_body] →  OK / NG
```

**目前狀態：** YOLO #1、YOLO #2、DINOv2+kNN 均已訓練完成。
本專案負責「方向正規化」、「PatchCore 訓練」及「PatchCore 驗證」三個階段。

---

## 環境需求

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) 套件管理器

```bash
# 安裝所有依賴（包含 anomalib、ultralytics、torch 等）
uv sync

# 安裝為可編輯套件（讓 CLI 指令可用）
uv pip install -e .
```

主要依賴版本：

| 套件 | 版本 |
|---|---|
| anomalib | ≥ 2.5.0 |
| torch | ≥ 2.12.0 |
| ultralytics | ≥ 8.4.65 |
| scikit-learn | ≥ 1.9.0 |
| opencv-python | ≥ 4.13.0 |

---

## 專案結構

```
Connector_Detection/
├── configs/
│   ├── patchcore_pin.yaml        # Pin row PatchCore 超參數
│   └── patchcore_body.yaml       # Metal body PatchCore 超參數
│
├── data/                         # 訓練與測試資料（由 normalize-orientation 產生）
│   ├── pin_row/
│   │   ├── train/good/           ← PatchCore 訓練用正常影像
│   │   └── test/
│   │       ├── good/             ← 測試用正常影像（計算 AUROC 用）
│   │       └── bad/              ← 測試用瑕疵影像（計算 AUROC 用）
│   └── metal_body/
│       └── (same structure)
│
├── models/
│   ├── yolo_connector.pt         # YOLO #1（自行放置）
│   ├── yolo_parts.pt             # YOLO #2（自行放置）
│   └── patchcore/
│       ├── pin_row/              ← 訓練後 checkpoint 存放位置
│       └── metal_body/
│
├── results/
│   ├── pin_row/
│   │   ├── OK/                   ← 驗證結果：正常影像
│   │   └── NG/                   ← 驗證結果：異常影像
│   └── metal_body/
│       └── (same structure)
│
├── scripts/                      # 入口腳本（thin wrappers）
│   ├── normalize_orientation.py
│   ├── train_patchcore.py
│   └── validate_patchcore.py
│
└── src/connector_detection/
    ├── orientation.py             # 方向正規化核心邏輯
    └── commands/
        ├── normalize.py           # normalize CLI 實作
        ├── train.py               # train CLI 實作
        └── validate.py            # validate CLI 實作
```

---

## 使用指南

### Step 1 — 準備訓練資料（方向正規化）

將 YOLO #1 + DINOv2 篩選出的目標型號連接器影像，用 YOLO #2 偵測各部位並統一方向，輸出 pin_row / metal_body crops。

```bash
# 正常連接器 → 訓練資料（train/good）
uv run scripts/normalize_orientation.py /path/to/normal_connectors \
    --yolo-model models/yolo_parts.pt \
    --split train \
    --label good

# 正常連接器 → 測試資料（test/good）
uv run scripts/normalize_orientation.py /path/to/normal_test \
    --yolo-model models/yolo_parts.pt \
    --split test \
    --label good

# 瑕疵連接器 → 測試資料（test/bad）— 可選，用於計算 AUROC/F1
uv run scripts/normalize_orientation.py /path/to/defective_connectors \
    --yolo-model models/yolo_parts.pt \
    --split test \
    --label bad
```

**選項說明：**

| 選項 | 預設值 | 說明 |
|---|---|---|
| `--yolo-model` | 必填 | YOLO #2 模型路徑 |
| `--data-dir` | `data/` | 輸出資料根目錄 |
| `--split` | `train` | `train` 或 `test` |
| `--label` | `good` | `good` 或 `bad` |
| `--conf` | `0.25` | YOLO 偵測信心閾值 |
| `--skip-existing` | `false` | 已存在的輸出檔案跳過不覆蓋 |

**方向判斷邏輯：**

YOLO #2 偵測 pin_row（class 0）和 metal_body（class 1）的中心點，比較相對位置決定旋轉角度：

| pin_row 相對位置 | 旋轉修正 |
|---|---|
| 在 metal_body 上方（正位） | 0°（不旋轉） |
| 在右側 | 逆時針 90° |
| 在下方（倒置） | 180° |
| 在左側 | 順時針 90° |

---

### Step 2 — 訓練 PatchCore

PatchCore 是 one-class 方法，只需要正常影像訓練。通常 1 個 epoch 即可完成（feature extraction + coreset 建立）。

```bash
# 訓練 pin row 模型
uv run scripts/train_patchcore.py pin_row

# 訓練 metal body 模型
uv run scripts/train_patchcore.py metal_body

# 指定自訂路徑
uv run scripts/train_patchcore.py pin_row \
    --data-dir data/pin_row \
    --output-dir models/patchcore/pin_row \
    --config configs/patchcore_pin.yaml
```

**選項說明：**

| 選項 | 預設值 | 說明 |
|---|---|---|
| `--data-dir` | `data/<component>` | 資料根目錄 |
| `--output-dir` | `models/patchcore/<component>` | Checkpoint 輸出位置 |
| `--config` | `configs/patchcore_<component>.yaml` | YAML 超參數設定檔 |

訓練完成後 checkpoint 儲存於：
```
models/patchcore/pin_row/Patchcore/v0/weights/lightning/model.ckpt
```

---

### Step 3 — 驗證與評估

```bash
# 自動計算最佳 threshold（需有 test/good 和 test/bad 資料）
uv run scripts/validate_patchcore.py pin_row \
    --ckpt-path models/patchcore/pin_row/Patchcore/v0/weights/lightning/model.ckpt

# 手動指定 threshold
uv run scripts/validate_patchcore.py pin_row \
    --ckpt-path models/patchcore/pin_row/Patchcore/v0/weights/lightning/model.ckpt \
    --threshold 0.45

# 指定自訂路徑
uv run scripts/validate_patchcore.py metal_body \
    --ckpt-path models/patchcore/metal_body/Patchcore/v0/weights/lightning/model.ckpt \
    --test-dir data/metal_body/test \
    --output-dir results/metal_body
```

**選項說明：**

| 選項 | 預設值 | 說明 |
|---|---|---|
| `--ckpt-path` | 必填 | 訓練好的 .ckpt 路徑 |
| `--test-dir` | `data/<component>/test` | 測試影像目錄 |
| `--output-dir` | `results/<component>` | OK/NG 輸出根目錄 |
| `--threshold` | 自動 | 手動閾值，省略時自動計算 |

**Threshold 自動判斷規則：**
1. 有 `test/good` + `test/bad` → 計算 **F1-optimal threshold**
2. 只有未標記影像 → 使用 **95th percentile**
3. 手動指定 `--threshold` → 優先使用

**輸出內容：**

| 輸出 | 說明 |
|---|---|
| `results/<component>/OK/` | 推論為正常的影像（複製） |
| `results/<component>/NG/` | 推論為異常的影像（複製） |
| `results/<component>/scores_<component>.csv` | 每張影像的 anomaly score 與預測結果 |
| `results/<component>/validation_report_<component>.png` | 四格圖表報告 |

**驗證報告圖表（2×2）：**

```
┌─────────────────────┬─────────────────────┐
│  Score 分布直方圖    │     ROC Curve        │
│  (OK/NG 疊加)        │  (含 AUROC 數值)     │
├─────────────────────┼─────────────────────┤
│   Confusion Matrix   │  Precision-Recall    │
│  (Precision/Recall/F1)│  Curve (含 AP)      │
└─────────────────────┴─────────────────────┘
```

> 若測試資料無地面真相標籤（只有單一目錄），後三張圖表會顯示「N/A」，僅 Score 分布圖有效。

---

## 設定檔調整

編輯 `configs/patchcore_pin.yaml` 或 `configs/patchcore_body.yaml`：

```yaml
backbone: wide_resnet50_2      # 特徵提取 backbone
layers:
  - layer2                     # 中階紋理特徵
  - layer3                     # 高階語意特徵
pre_trained: true
coreset_sampling_ratio: 0.1    # 記憶體庫採樣比例（↑ 提升準確率，↑ 記憶體用量）
num_neighbors: 9               # kNN 近鄰數（↑ 更平滑，↓ 更敏感）
input_size: [224, 224]
train_batch_size: 32
eval_batch_size: 32
num_workers: 4
```

**調參建議：**

| 問題 | 調整方向 |
|---|---|
| 漏判（NG 被判為 OK） | 降低 `threshold` 或降低 `coreset_sampling_ratio` |
| 誤判過多（OK 被判為 NG） | 提高 `threshold` |
| 訓練資料少（< 100 張） | 提高 `coreset_sampling_ratio` 至 0.3～1.0 |
| 顯卡記憶體不足 | 降低 `train_batch_size` 或 `input_size` |

---

## 安裝後 CLI 指令

安裝完成（`uv pip install -e .`）後，可直接使用以下指令：

```bash
normalize-orientation <input_dir> --yolo-model <model.pt> [options]
train-patchcore <component> [options]
validate-patchcore <component> --ckpt-path <model.ckpt> [options]
```

---

## 常見問題

**Q: YOLO #2 偵測不到 pin_row 或 metal_body？**
- 確認 `--conf` 不要設太高（預設 0.25）。
- 確認輸入影像已是 YOLO #1 crop 後的連接器，不是完整 AOI 截圖。
- 檢查 `normalize_orientation.py` 輸出的 `[SKIP]` / `[WARN]` 訊息。

**Q: 訓練時沒有 test 資料怎麼辦？**
- 只需要 `data/<component>/train/good/` 就可以啟動訓練。
- anomalib 會自動從 train/good 切分 20% 作為 validation set。
- 後續取得瑕疵樣本後，補充 `test/bad/` 再跑 validate 即可得到 AUROC 等指標。

**Q: Checkpoint 路徑怎麼找？**
```bash
find models/patchcore -name "*.ckpt"
```
