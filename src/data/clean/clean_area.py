from src.data.clean.common import pd, DataFrame

def dien_tich(data : DataFrame) -> DataFrame:
    """
        Chuẩn hoá cột 'Diện tích' trong DataFrame bất động sản.

        Hàm chuyển các giá trị diện tích dạng chuỗi (ví dụ: "96 m²", "120,5 m²")
        về số thực (float). Các ký tự đơn vị được loại bỏ, dấu phẩy được
        chuyển thành dấu chấm.

        Các giá trị không hợp lệ hoặc không thể chuyển đổi sẽ được gán NaN.

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame chứa cột 'Diện tích'.

        Returns
        -------
        pandas.DataFrame
            DataFrame sau khi chuẩn hoá cột 'Diện tích'.
    """

    data['Diện tích'] = (
        data['Diện tích']
        .astype(str)
        .str.lower()
        .str.replace("m²", "")
        .str.replace(',', '.', regex=False)
        .pipe(pd.to_numeric, errors='coerce')
    )
    return data
