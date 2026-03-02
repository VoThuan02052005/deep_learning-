import numpy as np
import pandas as pd

def clean_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    # Chuẩn hóa legal_status
    def standardize_legal(text):
        if pd.isna(text) or not isinstance(text, str):
            return "Unknown"
        text = text.lower().strip()
        if any(kw in text for kw in ['sổ đỏ', 'sổ hồng', 'sđcc', 'sổ riêng', 'đã có sổ', 'có sổ']):
            return "So_Do_So_Hong"
        if any(kw in text for kw in ['hợp đồng', 'hđmb', 'hdmb', 'góp vốn']):
            return "Hop_Dong"
        if any(kw in text for kw in ['vi bằng', 'giấy tay']):
            return "Vi_Bang"
        if any(kw in text for kw in ['đang chờ', 'đợi sổ']):
            return "Dang_Cho_So"
        return "Other_Unknown"

    data['legal_status'] = data['legal_status'].apply(standardize_legal)

    # Các cột số cần điền missing
    num_cols = ['bedrooms', 'bathrooms']

    for col in num_cols:
        # Cột chỉ báo missing
        data[f'{col}_is_missing'] = data[col].isna().astype(int)

        # Convert sang float để tránh lỗi median với pd.NA
        data[col] = data[col].astype(float)

        # Global median (cả cột)
        global_median = data[col].median()
        if pd.isna(global_median):
            global_median = 0.0

        def fill_group_median(x):
            if x.isna().all() or len(x) == 0:
                return x.fillna(global_median)
            median_val = x.median()
            if pd.isna(median_val):
                return x.fillna(global_median)
            return x.fillna(median_val)
        data[col] = data.groupby(['property_type', 'city'])[col].transform(fill_group_median)
        data[col] = data[col].fillna(global_median)

    data = data.dropna(subset=['posted_date'])

    return data
