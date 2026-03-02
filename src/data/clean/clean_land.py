import pandas as pd


def xoa_giao_dich_cho_thue(data: pd.DataFrame) -> pd.DataFrame:
    """
    Loại bỏ các bản ghi có Loại giao dịch là 'Cho thuê'.

    Mục đích:
    - Chỉ giữ lại dữ liệu mua bán
    - Phục vụ bài toán dự đoán giá nhà đất

    Input:
    - data (pd.DataFrame)

    Output:
    - pd.DataFrame đã loại bỏ các giao dịch cho thuê
    """


    # Chuẩn hóa chuỗi trước khi lọc
    mask = (
        data["Loại giao dịch"]
        .astype(str)
        .str.lower()
        .str.contains("thuê", na=False)
    )

    data = data[~mask]

    return data
