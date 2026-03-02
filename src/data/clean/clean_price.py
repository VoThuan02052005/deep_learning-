import re
import pandas as pd
import numpy as np

def muc_gia(data: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hoá cột 'Mức giá' và quy đổi về đơn vị chuẩn:
    - Bán: Quy đổi về tỷ (Billion VND).
    - Cho thuê: Quy đổi về triệu/tháng (Million VND/month).
    """
    if 'Mức giá' not in data.columns:
        return data

    def parse_price(row):
        text = str(row['Mức giá']).lower().strip()
        area = row.get('Diện tích', 0)
        
        if any(kw in text for kw in ['thỏa thuận', 'thoả thuận', 'nan', 'none']):
            return np.nan, np.nan

        # Regex để bắt số (chấp nhận cả 5,5 và 5.5)
        match = re.search(r"(\d+(?:[.,]\d+)?)", text)
        if not match:
            return np.nan, np.nan
        
        val = float(match.group(1).replace(',', '.'))
        unit = text[match.end():].strip()

        # Logic quy đổi
        is_rental = "thuê" in str(row.get('Loại giao dịch', '')).lower() or "tháng" in unit
        
        if is_rental:
            # Chuẩn hóa về TRIỆU/tháng
            if 'tỷ' in unit:
                return val * 1000, "triệu/tháng"
            if 'nghìn' in unit or 'k' in unit:
                return val / 1000, "triệu/tháng"
            if '/m²' in unit and area > 0:
                return val * area, "triệu/tháng"
            return val, "triệu/tháng"
        else:
            # Chuẩn hóa về TỶ
            if 'triệu' in unit or 'tr' in unit:
                if '/m²' in unit and area > 0:
                    return (val * area) / 1000, "tỷ"
                return val / 1000, "tỷ"
            if 'tỷ' in unit:
                if '/m²' in unit and area > 0: # Hiếm nhưng có thể xảy ra tỷ/m2
                    return val * area, "tỷ"
                return val, "tỷ"
            return val, unit # Giữ nguyên nếu không khớp tỷ/triệu

    # Áp dụng parse
    res = data.apply(parse_price, axis=1)
    data["Giá"] = res.apply(lambda x: x[0])
    data["Đơn vị(Mức giá)"] = res.apply(lambda x: x[1])

    return data
