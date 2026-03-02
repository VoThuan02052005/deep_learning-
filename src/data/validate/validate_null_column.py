from .common import np, pd , DataFrame


def xoa_cot_null_nhieu(data: pd.DataFrame) -> pd.DataFrame:
    """
    Loại bỏ các cột có tỷ lệ giá trị thiếu (null) quá cao trong DataFrame.

    Các bước xử lý:
    1. Tính tỷ lệ giá trị null của từng cột.
    2. Ghi log tỷ lệ null để phục vụ kiểm tra và debug.
    3. Loại bỏ các cột có tỷ lệ null lớn hơn 40%.
    4. Ghi log danh sách các cột còn lại sau khi loại bỏ.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame đầu vào cần được làm sạch.

    Returns
    -------
    pd.DataFrame
        DataFrame sau khi đã loại bỏ các cột có nhiều giá trị null.
    """

    null_ratio = data.isnull().mean().sort_values(ascending=False)
    cols_to_drop = null_ratio[null_ratio > 0.5].index.tolist()
    data = data.drop(cols_to_drop, axis=1)
    return data
