from pandas import DataFrame

def validate_location(data: DataFrame) -> DataFrame:
    """
    Kiểm tra và làm sạch thông tin vị trí trong DataFrame.

    Các bước xử lý:
    1. Loại bỏ các bản ghi thiếu thông tin 'Thành phố'.
    2. Loại bỏ các bản ghi thiếu thông tin 'Quận/huyện'.

    Parameters
    ----------
    data : DataFrame
        Dữ liệu đầu vào chứa thông tin vị trí bất động sản.

    Returns
    -------
    DataFrame
        DataFrame đã được làm sạch, chỉ giữ lại các bản ghi có đầy đủ thông tin vị trí.
    """

    # Giữ lại các dòng có đầy đủ thông tin thành phố và quận/huyện
    data = data[
        data["Thành phố"].notna() &
        data["Quận/huyện"].notna()
    ]

    return data
