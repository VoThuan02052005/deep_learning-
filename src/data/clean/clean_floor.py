from src.data.clean.common import pd, DataFrame

def so_tang(data : DataFrame) -> DataFrame:
    """
    Chuẩn hoá cột 'Số tầng' trong DataFrame bất động sản.

    Hàm chuyển các giá trị trong cột 'Số tầng' từ dạng chuỗi
    (ví dụ: "3 tầng", "5") sang kiểu số nguyên (Int64).
    Từ khoá "tầng" được loại bỏ, các giá trị không hợp lệ hoặc
    không thể chuyển đổi sẽ được gán NaN.

    Kiểu Int64 (nullable integer) được sử dụng để cho phép
    tồn tại giá trị NaN sau khi chuẩn hoá.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame chứa cột 'Số tầng'.

    Returns
    -------
    pandas.DataFrame
        DataFrame sau khi chuẩn hoá cột 'Số tầng'.
    """
    data["Số tầng"] = (
        data["Số tầng"]
        .astype(str)
        .str.replace("tầng", "")
        .str.strip()
        .pipe(pd.to_numeric, errors='coerce')
        .astype("Int64")
    )
    return data
