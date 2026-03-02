from src.data.clean.common import pd, DataFrame
def so_phong_ngu(data : DataFrame) -> DataFrame:
    """
        Chuẩn hoá cột "Số phòng ngủ" trong DataFrame bất động sản.

        Hàm chuyển các giá trị Số phòng ngủ dạng chuỗi (ví dụ: "9 phòng", "9")
        về số nguyên int64 . loại bỏ chữ , xóa khoảng trắng.

        Các giá trị không hợp lệ hoặc không thể chuyển đổi sẽ được gán NaN.

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame chứa cột "Số phòng ngủ".

        Returns
        -------
        pandas.DataFrame
            DataFrame sau khi chuẩn hoá cột "Số phòng ngủ".
    """
    data["Số phòng ngủ"] = (
        data["Số phòng ngủ"]
        .astype(str)
        .str.replace("phòng", "")
        .str.strip()
        .pipe(pd.to_numeric, errors='coerce')
        .astype("float")
    )
    return data