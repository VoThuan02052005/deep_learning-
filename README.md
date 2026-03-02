# 🏠 House Price Predictor

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red?logo=streamlit)](https://streamlit.io)
[![NumPy](https://img.shields.io/badge/NumPy-MLP%20from%20scratch-013243?logo=numpy)](https://numpy.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-preprocessing-F7931E?logo=scikitlearn)](https://scikit-learn.org)

**Hệ thống dự đoán giá bất động sản Việt Nam — End-to-End ML Pipeline**

[Cài đặt](#-cài-đặt) · [Chạy dự án](#%EF%B8%8F-hướng-dẫn-chạy) · [Mô hình](#-mô-hình) · [Cấu trúc](#-cấu-trúc-dự-án)

</div>

---

## 📌 Giới thiệu

**House Price Predictor** dự đoán giá nhà đất tại Việt Nam (đơn vị: **tỷ VNĐ**) dựa trên các đặc trưng: diện tích, số phòng, vị trí, loại hình, pháp lý.

Điểm nổi bật:
- **MLP tự triển khai** bằng NumPy thuần (Adam optimizer, Dropout, Early Stopping, He init)
- **Pipeline hoàn chỉnh** từ crawl → clean → validate → feature engineering → train → serve
- **Streamlit app** với filter quận/huyện động theo tỉnh + chuẩn hoá pháp lý thô realtime
- Dữ liệu: **~116.000 tin đăng bán** từ batdongsan.com.vn

---

## ▶️ Hướng dẫn chạy

### 🚀 Quick Start — Nhận file ZIP

> **Dành cho người nhận file `house_price_app.zip`** — chạy ngay, không cần train lại.

```bash
# Bước 1: Giải nén
unzip house_price_app.zip -d project_price
cd project_price

# Bước 2: Cài dependencies
pip install -r requirements.txt

# Bước 3: Chạy app
streamlit run src/app/app.py
```

🌍 Truy cập: **http://localhost:8501**

---

### Option B — Clone từ Git (có model sẵn)

```bash
git clone <repo-url>
cd project_price
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run src/app/app.py
```

🌍 Truy cập: **http://localhost:8501**

---

### Option B — Chạy toàn bộ pipeline (từ đầu)

> Dùng khi bạn muốn train lại model từ dữ liệu.

```bash
# 0. Cài môi trường (nếu chưa)
pip install -r requirements.txt

# 1. Thu thập dữ liệu (bỏ qua nếu đã có data/raw/)
python3 -m src.data.etl.extract

# 2. Làm sạch + kiểm định + tách Bán/Cho Thuê
python3 run_pipeline_clean.py
# → Output: data/staging/data_ban.csv  (~116K dòng)

# 3. Lưu preprocessor
python3 -m src.pipelines.save_preprocessor
# → Output: models/preprocessor.pkl

# 4. Train các mô hình
python3 -m src.pipelines.train_mlp          # ← mô hình chính
python3 -m src.pipelines.train_linear       # Ridge, Lasso
python3 -m src.pipelines.train_tree         # RandomForest, GradBoost
python3 -m src.pipelines.train_baseline     # MeanBaseline
# → Output: models/*.joblib

# 5. Chạy web app
streamlit run src/app/app.py
```

### Kiểm tra model tồn tại

```bash
ls models/
# Cần có: mlp_model.joblib  preprocessor.pkl
```

---

## 🧠 Mô hình

### CustomMLP (mô hình chính)

> Triển khai hoàn toàn bằng **NumPy** — không phụ thuộc framework deep learning.

**Kiến trúc:** `Input → Dense(256) → Dense(128) → Dense(64) → Output(1)`

| Kỹ thuật | Chi tiết |
|---|---|
| Activation | ReLU |
| Optimizer | Adam (β₁=0.9, β₂=0.999) |
| Regularization | L2 (λ=1e-4) + Dropout (rate=0.2) |
| Weight Init | He initialization |
| Gradient Clipping | `clip(-5, 5)` |
| Early Stopping | patience=30 (val_loss) |
| LR Scheduler | Decay ×0.5 mỗi 8 epoch không cải thiện |
| Batch Size | 512, Epochs tối đa 2000 |

### Inference Pipeline

```
User input
    ↓ ① Raw preprocessing  (dates, missing flags, standardize_legal)
    ↓ ② Feature Engineering (time features, cyclic encoding, log1p)
    ↓ ③ OHE transform
    ↓ ④ scaler_X.transform()
    ↓ ⑤ model.predict()          →  ŷ_scaled
    ↓ ⑥ scaler_y.inverse_transform()  →  log(price)
    ↓ ⑦ np.expm1()               →  price (tỷ VNĐ)
```

### Đánh giá (trên tập test)

| Metric | MLP | GradBoost | Ridge |
|---|---|---|---|
| **MAE** (tỷ) | — | — | — |
| **RMSE** (tỷ) | — | — | — |
| **R²** | — | — | — |

> *Chạy `train_mlp.py` hoặc các pipeline tương ứng để xem kết quả đánh giá.*

---

## 📁 Cấu trúc dự án

```
project_price/
│
├── 📄 config.yaml                 # Cấu hình Crawler & DB
├── 📄 run_pipeline_clean.py       # Entrypoint: clean + validate + tách data
├── 📄 requirements.txt
│
├── 📂 data/
│   ├── raw/                       # Dữ liệu thô (gia_nha.csv)
│   ├── staging/                   # Sau pipeline: data_ban.csv, data_cho_thue.csv
│   └── processed/                 # Sau feature engineering
│
├── 📂 models/                     # Model và preprocessor serialized
│   ├── mlp_model.joblib
│   ├── preprocessor.pkl
│   ├── linear_*.joblib
│   └── tree_*.joblib
│
├── 📂 logs/                       # Log từ các pipeline
│
└── 📂 src/
    ├── 📂 data/
    │   ├── etl/                   # Selenium scraper
    │   ├── clean/                 # 10 module chuẩn hoá thô
    │   └── validate/              # Rename, missing values, validate ranges
    │
    ├── 📂 features/
    │   └── build_feature.py       # Feature engineering pipeline
    │
    ├── 📂 models/
    │   ├── custom_mlp.py          
    │   ├── baseline.py
    │   ├── linear.py
    │   └── tree.py
    │
    ├── 📂 pipelines/
    │   ├── save_preprocessor.py
    │   ├── train_baseline.py
    │   ├── train_linear.py
    │   ├── train_tree.py
    │   └── train_mlp.py
    │
    ├── 📂 app/
    │   └── app.py                 # Streamlit web app
    │
    └── 📂 utils/
        ├── logger.py              # Logger wrapper
        ├── metrics.py             # MAE / RMSE / R²
        ├── io.py                  # File I/O helpers
        └── timer.py               # Timing decorator
```

---

## 🌐 Web App — Tính năng

| Tính năng | Mô tả |
|---|---|
| **Lọc quận/huyện động** | Tự động lọc theo tỉnh đã chọn từ dữ liệu training |
| **Pháp lý thô** | Nhập "sổ hồng", "hợp đồng",... → tự chuẩn hoá realtime |
| **Pipeline trace** | Expander hiển thị giá trị tại từng bước inference |
| **Model caching** | `@st.cache_resource` — load model 1 lần duy nhất |
| **Ví dụ mẫu** | Nút "✨ Dùng ví dụ mẫu" điền sẵn giá trị |

---

## 📊 Dữ liệu

| Thuộc tính | Giá trị |
|---|---|
| **Nguồn** | batdongsan.com.vn |
| **Số lượng (bán)** | ~116.000 dòng |
| **Tỉnh/Thành phố** | 63 tỉnh thành Việt Nam |
| **Khoảng giá** | 0.5 — 200+ tỷ VNĐ |
| **Features gốc** | area, bedrooms, bathrooms, city, district, property_type, legal_status, posted_date |

---

## 📝 Quy ước code

- **Logger:** Dùng `src/utils/logger.py` thay vì `print()`
- **Metrics:** Luôn evaluate trên giá gốc (sau `inverse_transform` + `expm1`)
- **Paths:** Chạy từ thư mục gốc `project_price/` (không `cd src/`)
- **Models:** Lưu bằng `joblib.dump()` vào `models/`

---


