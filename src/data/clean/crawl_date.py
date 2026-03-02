from src.data.clean.common import pd, DataFrame

def crawl_date(data : DataFrame) -> DataFrame :
    """
    Chuẩn hóa cột "craw_date" trong dataframe bất động sản.

    Hàm chuyển các giá trị ngày lấy dữ liệu ( 2025-08-07 ) về định dạng datetime64
    các giá trị không hợp lệ thì gá giá trị NAN.
    :param data:
    ------------
    data : pandas.DataFrame
    :return:
    ----------
    pandas.DataFrame
    """
    data["crawl_date"] = pd.to_datetime(
        data["crawl_date"],
        errors="coerce"

    )
    return data