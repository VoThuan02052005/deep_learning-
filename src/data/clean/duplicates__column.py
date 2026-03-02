from src.data.clean.common import pd


def xoa_trung_lap_theo_cot(
    data: pd.DataFrame,
    subset: list[str],
    keep: str = "first"
) -> pd.DataFrame:
    """
    Loại bỏ các bản ghi trùng nhau theo các cột chỉ định.

    Parameters
    ----------
    subset : list[str]
        Danh sách cột dùng để kiểm tra trùng lặp.
    keep : {"first", "last", False}
        Giữ bản ghi nào.

    Returns
    -------
    pd.DataFrame
    """
    return data.drop_duplicates(subset=subset, keep=keep)
