from .common import np, pd, DataFrame


def validate_price(data: DataFrame) -> DataFrame:
    """
    Kiểm tra và lọc dữ liệu theo cột giá.

    Các bước:
    1. Loại bỏ bản ghi có giá NaN.
    2. Loại bỏ bản ghi có giá <= 0.
    3. Loại bỏ outlier cực đoan (> 99.5th percentile) cho từng loại giao dịch
       (Bán và Cho Thuê có ngưỡng khác nhau nên lọc riêng theo nhóm).

    Input cột: 'Giá' (float, đã được chuẩn hóa bởi clean_price.py)
    """
    # Bỏ NaN và giá <= 0
    data = data.dropna(subset=["Giá"])
    data = data[data["Giá"] > 0]

    # Lọc outlier theo từng nhóm giao dịch (Bán / Cho Thuê có đơn vị giá khác nhau)
    def filter_outlier(group):
        upper = group["Giá"].quantile(0.995)
        return group[group["Giá"] <= upper]

    if "Loại giao dịch" in data.columns:
        # Giữ lại cột 'Loại giao dịch' bằng cách không dùng include_groups=False 
        # hoặc xử lý để group key không bị mất.
        data = (
            data
            .groupby("Loại giao dịch", group_keys=False)
            .apply(filter_outlier)
            .reset_index(drop=True)
        )
    else:
        upper = data["Giá"].quantile(0.995)
        data = data[data["Giá"] <= upper]

    return data
