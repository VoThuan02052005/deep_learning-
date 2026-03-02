import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime

# ─── path setup (hoạt động cả local lẫn Streamlit Cloud) ───────────────────
_here = os.path.dirname(os.path.abspath(__file__))       # .../src/app
_root = os.path.abspath(os.path.join(_here, "../.."))     # project root
if _root not in sys.path:
    sys.path.insert(0, _root)


def standardize_legal(text: str) -> str:
    """Mirror of the training-pipeline's standardize_legal logic."""
    if not isinstance(text, str) or not text.strip():
        return "Unknown"
    t = text.lower().strip()
    if any(kw in t for kw in ['sổ đỏ', 'sổ hồng', 'sđcc', 'sổ riêng', 'đã có sổ', 'có sổ']):
        return "So_Do_So_Hong"
    if any(kw in t for kw in ['hợp đồng', 'hđmb', 'hdmb', 'góp vốn']):
        return "Hop_Dong"
    if any(kw in t for kw in ['vi bằng', 'giấy tay']):
        return "Vi_Bang"
    if any(kw in t for kw in ['đang chờ', 'đợi sổ']):
        return "Dang_Cho_So"
    return "Other_Unknown"

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .prediction-card {
        background-color: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        text-align: center;
        border-left: 5px solid #007bff;
    }
    .prediction-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #007bff;
    }
    .pipeline-step {
        background: white;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        border-left: 4px solid #28a745;
        font-size: 0.88rem;
        color: #333;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7bcf,#2e7bcf);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# GOOGLE DRIVE MODEL IDs  (thay bằng ID thực của bạn)
# ─────────────────────────────────────────────────────────
# Hướng dẫn lấy ID:
#   1. Upload file lên Google Drive
#   2. Chuột phải → "Chia sẻ" → "Bất kỳ ai có đường liên kết" → Sao chép liên kết
#   3. Link dạng: https://drive.google.com/file/d/<FILE_ID>/view
#   4. Chép <FILE_ID> vào đây
GDRIVE_IDS = {
    "models/mlp_model.joblib":  "1eN-B2YIwfw_cLE9Re_krf9Pj7qsDQnM5",
    "models/preprocessor.pkl":  "1E0i7z96gcbdtn7UN9GsjqIDdASRp3pgB",
}
# Kích thước tối thiểu để phát hiện file bị lỗi (tải về HTML thay vì file thật)
MIN_SIZES = {
    "models/mlp_model.joblib": 50 * 1024 * 1024,   # ≥ 50 MB
    "models/preprocessor.pkl": 5 * 1024,            # ≥ 5 KB
}

def _download_if_missing():
    """Download model files from Google Drive if not present (Streamlit Cloud)."""
    os.makedirs("models", exist_ok=True)
    for path, file_id in GDRIVE_IDS.items():
        fname = path.split("/")[-1]

        # Kiểm tra nếu file đã tồn tại nhưng quá nhỏ (có thể là HTML lỗi)
        if os.path.exists(path):
            size = os.path.getsize(path)
            if size < MIN_SIZES.get(path, 0):
                st.warning(f"⚠️ Phát hiện {fname} bị lỗi ({size} bytes), đang tải lại...")
                os.remove(path)

        if not os.path.exists(path):
            # Chỉ cần gdown khi thực sự thiếu file
            try:
                import gdown
            except ImportError:
                st.error("⚠️ gdown chưa được cài. Chạy: pip install gdown")
                st.stop()

            with st.spinner(f"⬇️ Đang tải {fname} từ Google Drive..."):
                result = gdown.download(id=file_id, output=path, quiet=False, fuzzy=True)
                if result is None or not os.path.exists(path):
                    st.error(f"❌ Tải {fname} thất bại. Kiểm tra file có được chia sẻ công khai chưa.")
                    st.stop()
                size = os.path.getsize(path)
                if size < MIN_SIZES.get(path, 0):
                    os.remove(path)
                    st.error(f"❌ {fname} bị lỗi (chỉ {size} bytes). Google Drive có thể chưa chia sẻ công khai.")
                    st.stop()


# ─────────────────────────────────────────────────────────
# CACHED LOADING
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    """Load model and preprocessor objects with caching."""
    from src.models.custom_mlp import CustomMLP   # import here — after sys.path is ready

    _download_if_missing()

    model_path = "models/mlp_model.joblib"
    prep_path  = "models/preprocessor.pkl"

    if not os.path.exists(model_path) or not os.path.exists(prep_path):
        st.error("⚠️ Không tìm thấy file model. Vui lòng cập nhật GDRIVE_IDS trong app.py.")
        st.stop()

    model        = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)

    # Validate đúng loại object
    if not isinstance(preprocessor, dict):
        st.error(f"❌ preprocessor.pkl sai định dạng (type: {type(preprocessor).__name__}). "
                 f"File sizes — mlp: {os.path.getsize(model_path)//1024}KB, "
                 f"prep: {os.path.getsize(prep_path)//1024}KB")
        st.stop()
    if not hasattr(model, "predict"):
        st.error(f"❌ mlp_model.joblib sai định dạng (type: {type(model).__name__}).")
        st.stop()

    return model, preprocessor



@st.cache_data
def build_city_district_map(csv_path: str = "data/staging/data_ban.csv") -> dict:
    """Build {city: [sorted districts...]} mapping from the training dataset."""
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path, usecols=["city", "district"])
    mapping = (
        df.dropna(subset=["city", "district"])
        .groupby("city")["district"]
        .unique()
        .apply(sorted)
        .apply(list)
        .to_dict()
    )
    return mapping


# ─────────────────────────────────────────────────────────
# PREDICTION PIPELINE
# Flow: User input → Raw preprocessing → Feature Engineering
#       → Scaler_X → Model (predict log(price))
#       → Inverse scaler_y → expm1 → Predicted price (tỷ VNĐ)
# ─────────────────────────────────────────────────────────
def predict_price(input_df, model, preprocessor):
    """
    Full inference pipeline:
      1. Raw data preprocessing
      2. Feature Engineering
      3. Scaler_X  (numerical features)
      4. Model     → predict log(price) [scaled]
      5. Inverse scaler_y
      6. expm1
    Returns predicted price in tỷ VNĐ and a dict of intermediate steps.
    """
    steps = {}

    # ── Step 1: Raw data preprocessing ──────────────────
    df = input_df.copy()

    # Inject date info for feature engineering
    now_str = datetime.now().strftime("%Y-%m-%d")
    df["posted_date"] = now_str
    df["crawl_date"]  = now_str

    # Missing-value flags (mirrors training pipeline)
    df["bedrooms_is_missing"]  = df["bedrooms"].isna().astype(int)
    df["bathrooms_is_missing"] = df["bathrooms"].isna().astype(int)
    df["bedrooms"]  = df["bedrooms"].fillna(0)
    df["bathrooms"] = df["bathrooms"].fillna(0)

    steps["step1_raw"] = df[["area", "bedrooms", "bathrooms",
                              "city", "district", "property_type",
                              "transaction_type", "legal_status"]].to_dict(orient="records")[0]

    # ── Step 2: Feature Engineering ──────────────────────
    df["posted_date"] = pd.to_datetime(df["posted_date"])
    df["crawl_date"]  = pd.to_datetime(df["crawl_date"])

    df["posted_year"]     = df["posted_date"].dt.year
    df["posted_month"]    = df["posted_date"].dt.month
    df["posted_wday"]     = df["posted_date"].dt.weekday
    df["days_on_market"]  = (df["crawl_date"] - df["posted_date"]).dt.days

    # Cyclic time encoding
    df["month_sin"] = np.sin(2 * np.pi * df["posted_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["posted_month"] / 12)
    df["wday_sin"]  = np.sin(2 * np.pi * df["posted_wday"]  / 7)
    df["wday_cos"]  = np.cos(2 * np.pi * df["posted_wday"]  / 7)

    # Log-transform skewed time feature
    df["days_on_market"] = np.log1p(np.maximum(0, df["days_on_market"]))

    time_feats = ["posted_year", "days_on_market",
                  "month_sin", "month_cos", "wday_sin", "wday_cos"]
    df[time_feats] = df[time_feats].fillna(-1)

    steps["step2_features"] = {k: round(float(df[k].iloc[0]), 5) for k in time_feats}

    # ── Step 3: Categorical encoding ─────────────────────
    cat_cols = preprocessor["cat_cols"]
    df[cat_cols] = df[cat_cols].fillna("Unknown").astype(str)
    X_cat = preprocessor["ohe"].transform(df[cat_cols])
    cat_names = preprocessor["ohe"].get_feature_names_out(cat_cols)
    X_cat_df  = pd.DataFrame(X_cat, columns=cat_names, index=df.index)

    # ── Step 4: Scaler_X — scale numerical features ──────
    num_cols = preprocessor["num_cols"]

    # Log-transform skewed numerical features (mirrors training)
    skewed_cols = [c for c in num_cols if c not in time_feats]
    for col in skewed_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    X_num        = df[num_cols]
    X_num_scaled = preprocessor["scaler_X"].transform(X_num)
    X_num_scaled_df = pd.DataFrame(X_num_scaled, columns=num_cols, index=df.index)

    steps["step3_scaler_X"] = {k: round(float(X_num_scaled_df[k].iloc[0]), 5) for k in num_cols}

    # ── Step 5: Assemble feature matrix ──────────────────
    X = pd.concat([X_num_scaled_df, X_cat_df], axis=1).astype(np.float64)
    X = X[preprocessor["feature_names"]]   # enforce column order

    # ── Step 6: Model → predict log(price) [scaled] ──────
    y_pred_scaled = model.predict(np.asarray(X))
    steps["step4_model_output"] = float(y_pred_scaled.ravel()[0])

    # ── Step 7: Inverse scaler_y → log(price) ────────────
    y_pred_log = preprocessor["scaler_y"].inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    )
    steps["step5_inverse_scaler_y"] = float(y_pred_log.ravel()[0])

    # ── Step 8: expm1 → price (tỷ VNĐ) ──────────────────
    y_pred = float(np.expm1(y_pred_log).ravel()[0])
    steps["step6_expm1_price_ty"] = round(y_pred, 4)

    return y_pred, steps


# ─────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────
def main():
    st.title("🏠 House Price Predictor")
    st.markdown("Dự đoán giá nhà tại Việt Nam bằng mô hình **MLP** được huấn luyện trên dữ liệu thực tế.")

    model, preprocessor = load_assets()
    city_district_map   = build_city_district_map()

    # ── Sidebar ─────────────────────────────────────────
    st.sidebar.header("📍 Thông tin bất động sản")

    if st.sidebar.button("✨ Dùng ví dụ mẫu"):
        st.session_state.area       = 120.0
        st.session_state.bedrooms   = 3.0
        st.session_state.bathrooms  = 2.0
        st.session_state.city       = "Hồ Chí Minh"
        st.session_state.district   = "Quận 1"
        st.session_state.prop_type  = "Nhà riêng"
        st.session_state.legal      = "Sổ hồng"

    st.sidebar.subheader("📐 Đặc điểm vật lý")
    area      = st.sidebar.number_input("Diện tích (m²)", min_value=1.0, max_value=10000.0,
                                         value=float(st.session_state.get("area", 80.0)))
    bedrooms  = st.sidebar.slider("Số phòng ngủ",  0, 10, int(st.session_state.get("bedrooms", 2)))
    bathrooms = st.sidebar.slider("Số phòng tắm",  0, 10, int(st.session_state.get("bathrooms", 2)))

    st.sidebar.subheader("🗺️ Vị trí & Pháp lý")

    # ── City selectbox ────────────────────────────────────
    all_cities = sorted(preprocessor["ohe"].categories_[0].tolist())
    default_city = st.session_state.get("city", "Hà Nội")
    if default_city not in all_cities:
        default_city = all_cities[0]
    city = st.sidebar.selectbox(
        "Tỉnh / Thành phố",
        all_cities,
        index=all_cities.index(default_city)
    )

    # ── District selectbox — filtered by selected city ────
    districts_for_city = city_district_map.get(city, [])
    # Fallback: if city has no districts in dataset, show all
    if not districts_for_city:
        districts_for_city = sorted(preprocessor["ohe"].categories_[1].tolist())

    # Reset district when city changes
    if st.session_state.get("_last_city") != city:
        st.session_state["_last_city"] = city
        st.session_state["district"] = districts_for_city[0]

    default_district = st.session_state.get("district", districts_for_city[0])
    if default_district not in districts_for_city:
        default_district = districts_for_city[0]

    district = st.sidebar.selectbox(
        "Quận / Huyện",
        districts_for_city,
        index=districts_for_city.index(default_district),
        help=f"{len(districts_for_city)} quận/huyện thuộc {city} trong dữ liệu"
    )
    prop_type = st.sidebar.selectbox("Loại bất động sản",
                                      preprocessor["ohe"].categories_[2],
                                      index=list(preprocessor["ohe"].categories_[2]).index(
                                          st.session_state.get("prop_type", preprocessor["ohe"].categories_[2][0])))

    st.sidebar.subheader("📄 Pháp lý (nhập dạng thô)")
    legal_raw = st.sidebar.text_input(
        "Tình trạng pháp lý",
        value=st.session_state.get("legal", ""),
        placeholder="VD: sổ hồng, hợp đồng, vi bằng, đang chờ sổ...",
        help="Nhập giá trị thô, hệ thống sẽ tự chuẩn hoá."
    )
    legal_standardized = standardize_legal(legal_raw)
    st.sidebar.caption(f"→ Chuẩn hoá: **{legal_standardized}**")

    # ── Main content ─────────────────────────────────────
    st.markdown("---")
    col_info, col_result = st.columns([1, 1])

    with col_info:
        st.subheader("📋 Thông tin nhập vào")
        st.write(f"**Diện tích:** {area} m²")
        st.write(f"**Phòng ngủ:** {bedrooms}")
        st.write(f"**Phòng tắm:** {bathrooms}")
        st.write(f"**Vị trí:** {district}, {city}")
        st.write(f"**Loại:** {prop_type}")
        st.write(f"**Pháp lý (thô):** {legal_raw if legal_raw else '(trống)'}")
        st.write(f"**Pháp lý (chuẩn hoá):** `{legal_standardized}`")

    with col_result:
        if st.button("🚀 Dự đoán giá"):
            input_data = pd.DataFrame([{
                "area":             area,
                "bedrooms":         float(bedrooms),
                "bathrooms":        float(bathrooms),
                "city":             city,
                "district":         district,
                "property_type":    prop_type,
                "transaction_type": "Bán",
                "legal_status":     legal_standardized,
            }])

            with st.spinner("Đang phân tích..."):
                try:
                    price, steps = predict_price(input_data, model, preprocessor)

                    # Result card
                    st.markdown(f"""
                        <div class="prediction-card">
                            <p style="margin-bottom:0.5rem;font-size:1.2rem;color:#666;">Giá dự đoán</p>
                            <div class="prediction-value">{price:,.2f} tỷ VNĐ</div>
                            <p style="margin-top:1rem;font-size:0.85rem;color:#888;">
                                *Dựa trên mô hình MLP huấn luyện trên dữ liệu lịch sử
                            </p>
                        </div>
                    """, unsafe_allow_html=True)

                    # Pipeline trace
                    with st.expander("🔍 Chi tiết pipeline dự đoán", expanded=False):
                        st.markdown(f"""
                        <div class="pipeline-step">
                            <strong>① User input</strong> →
                            area={area}, bedrooms={bedrooms}, bathrooms={bathrooms},
                            city={city}, district={district}
                        </div>
                        <div class="pipeline-step">
                            <strong>② Raw data preprocessing</strong> →
                            Thêm ngày đăng, xử lý missing (bedrooms_is_missing, bathrooms_is_missing)<br>
                            &nbsp;&nbsp;Pháp lý: <em>"{legal_raw}"</em> → chuẩn hoá → <code>{legal_standardized}</code>
                        </div>
                        <div class="pipeline-step">
                            <strong>③ Feature Engineering</strong> →
                            posted_year={steps['step2_features'].get('posted_year','-')},
                            month_sin={steps['step2_features'].get('month_sin','-'):.4f},
                            month_cos={steps['step2_features'].get('month_cos','-'):.4f},
                            log1p(days_on_market)={steps['step2_features'].get('days_on_market','-'):.4f}
                        </div>
                        <div class="pipeline-step">
                            <strong>④ Scaler_X</strong> →
                            Chuẩn hoá đặc trưng số (StandardScaler)
                        </div>
                        <div class="pipeline-step">
                            <strong>⑤ Model (predict log(price))</strong> →
                            raw output = <code>{steps['step4_model_output']:.6f}</code>
                        </div>
                        <div class="pipeline-step">
                            <strong>⑥ Inverse scaler_y</strong> →
                            log(price) = <code>{steps['step5_inverse_scaler_y']:.6f}</code>
                        </div>
                        <div class="pipeline-step">
                            <strong>⑦ expm1</strong> →
                            <strong>{steps['step6_expm1_price_ty']:.4f} tỷ VNĐ</strong>
                        </div>
                        """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Lỗi dự đoán: {e}")
                    st.info("Vui lòng kiểm tra lại thông tin đầu vào.")

    st.markdown("---")
    st.info("💡 **Lưu ý:** Giá bất động sản có thể dao động lớn tuỳ theo vị trí cụ thể và tình hình thị trường hiện tại.")


if __name__ == "__main__":
    main()
