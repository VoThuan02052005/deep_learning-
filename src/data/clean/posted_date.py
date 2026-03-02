from src.data.clean.common import pd, DataFrame

def ngay_dang(data : DataFrame) -> DataFrame:
    """
    Chuẩn hoá cột 'Ngày đăng' trong DataFrame bất động sản.

    Hàm chuyển các giá trị ngày đăng từ nhiều định dạng chuỗi
    (ví dụ: '05-06-2025', '24/06/2025') về kiểu datetime64.
    Các giá trị không hợp lệ sẽ được gán NaT.

    Parameters
    ----------
    data : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """
    data["Ngày đăng"] = (
        data["Ngày đăng"]
        .astype(str)
        .str.replace("/", "-")
        .str.strip()
        .replace({"nan": None})
    )

    data["Ngày đăng"] = pd.to_datetime(
        data["Ngày đăng"],
        dayfirst=True,
        errors="coerce"
    )

    return data
