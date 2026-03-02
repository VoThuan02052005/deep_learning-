from src.data.clean.common import pd, DataFrame

def mat_tien(data : DataFrame) -> DataFrame:
    """
    Chuẩn hoá cột 'Mặt tiền' trong DataFrame bất động sản.

    Hàm chuyển các giá trị trong cột 'Mặt tiền' từ dạng chuỗi
    (ví dụ: "4m", "5.5 m") sang kiểu số thực (float).
    Ký tự đơn vị "m" được loại bỏ, các giá trị không hợp lệ
    hoặc không thể chuyển đổi sẽ được gán NaN.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame chứa cột 'Mặt tiền'.

    Returns
    -------
    pandas.DataFrame
        DataFrame sau khi chuẩn hoá cột 'Mặt tiền'.
    """
    data["Mặt tiền"] = (
        data["Mặt tiền"]
        .astype(str)
        .str.replace("m", "")
        .str.strip()
        .pipe(pd.to_numeric, errors='coerce')
    )
    return data
