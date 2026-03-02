from src.data.clean.common import pd, DataFrame
def so_phong_tam(data : DataFrame ) -> DataFrame:
    """
    Chuẩn hoá cột 'Số phòng tắm, vệ sinh' trong DataFrame bất động sản.

    Hàm này chuyển các giá trị trong cột 'Số phòng tắm, vệ sinh' từ dạng
    chuỗi (ví dụ: "2 phòng", "1") sang kiểu số nguyên (Int64).
    Từ khoá "phòng" sẽ được loại bỏ, các giá trị không hợp lệ hoặc
    không thể chuyển đổi sẽ được gán NaN.

    Kiểu dữ liệu Int64 (nullable integer) được sử dụng để cho phép
    tồn tại giá trị NaN sau khi chuyển đổi.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame chứa cột 'Số phòng tắm, vệ sinh'.

    Returns
    -------
    pandas.DataFrame
        DataFrame sau khi chuẩn hoá cột 'Số phòng tắm, vệ sinh'.
    """
    data["Số phòng tắm, vệ sinh"] = (
        data["Số phòng tắm, vệ sinh"]
        .astype(str)
        .str.replace("phòng", "")
        .str.strip()
        .pipe(pd.to_numeric, errors='coerce')
        .astype("Int64")
    )
    return data
