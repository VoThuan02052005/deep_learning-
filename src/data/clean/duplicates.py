from src.data.clean.common import pd


def xoa_trung_lap(data: pd.DataFrame) -> pd.DataFrame:
    """
    Loại bỏ các bản ghi trùng nhau hoàn toàn trong DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame sau khi đã loại bỏ các dòng trùng.
    """
    data = data.drop_duplicates()
    return data