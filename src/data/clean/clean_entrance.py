from src.data.clean.common import pd, DataFrame

def duong_vao(data : DataFrame) -> DataFrame:
    """
    Chuẩn hoá cột 'Đường vào' trong DataFrame bất động sản.

    Hàm chuyển các giá trị trong cột 'Đường vào' từ dạng chuỗi
    (ví dụ: "3m", "5.5 m") sang kiểu số thực (float).
    Ký tự đơn vị "m" được loại bỏ, các giá trị không hợp lệ
    hoặc không thể chuyển đổi sẽ được gán NaN.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame chứa cột 'Đường vào'.

    Returns
    -------
    pandas.DataFrame
        DataFrame sau khi chuẩn hoá cột 'Đường vào'.
    """
    data["Đường vào"] = (
        data["Đường vào"]
        .astype(str)
        .str.replace("m", "")
        .str.strip()
        .pipe(pd.to_numeric, errors='coerce')
    )
    return data
