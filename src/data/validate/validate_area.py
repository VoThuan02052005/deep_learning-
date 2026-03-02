
from .common import np, pd , DataFrame

def validate_area(data: DataFrame) -> DataFrame:
    """
    Kiểm tra và làm sạch cột diện tích trong DataFrame.

    Các bước xử lý:
    1. Loại bỏ các bản ghi có diện tích <= 0.
    2. Loại bỏ các bản ghi có diện tích bị thiếu (NaN).

    Parameters
    ----------
    data : DataFrame
        Dữ liệu đầu vào chứa thông tin bất động sản, bao gồm cột 'Diện tích'.

    Returns
    -------
    DataFrame
        DataFrame đã được làm sạch, chỉ giữ lại các bản ghi có diện tích hợp lệ.
    """
    data = data[(data["Diện tích"] >= 5) & (data["Diện tích"] <= 1000)]
    data = data.dropna(subset=["Diện tích"])

    return data