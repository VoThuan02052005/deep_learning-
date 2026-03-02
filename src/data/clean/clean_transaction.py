from src.data.clean.common import np, DataFrame

def loai_hinh_dat(data : DataFrame) -> DataFrame:
    """
    Chuẩn hoá cột 'Loại hình đất' trong DataFrame bất động sản.

    Hàm này ánh xạ các giá trị trong cột 'Loại hình đất' về một tập
    nhãn cố định (ví dụ: Nhà biệt thự, Căn hộ chung cư, Bán đất, ...),
    dựa trên việc kiểm tra chuỗi con (substring matching).

    Nếu một giá trị không khớp với bất kỳ loại nào trong danh sách,
    giá trị gốc sẽ được giữ nguyên.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame chứa cột 'Loại hình đất' cần được chuẩn hoá.

    Returns
    -------
    pandas.DataFrame
        DataFrame sau khi chuẩn hoá cột 'Loại hình đất'.

    Notes
    -----
    - So khớp chuỗi không phân biệt NaN (na=False).
    - Mỗi dòng chỉ được gán vào loại đầu tiên khớp trong danh sách conditions.
    - Thứ tự các điều kiện trong 'conditions' là quan trọng.
    """
    col = data['Loại hình đất']

    conditions = [
        col.str.contains("Nhà biệt thự", na=False),
        col.str.contains("Căn hộ chung cư", na=False),
        col.str.contains("Nhà mặt phố", na=False),
        col.str.contains("Bán đất", na=False),
        col.str.contains("Văn phòng", na=False),
        col.str.contains("Nhà riêng", na=False),
        col.str.contains("Condotel", na=False),
        col.str.contains("Đất nền", na=False),
        col.str.contains("Shophouse", na=False),
        col.str.contains("Nhà trọ", na=False),
        col.str.contains("Chung cư mini, căn hộ", na=False),
        col.str.contains("Kho", na=False),
        col.str.contains("Trang trại", na=False),
        col.str.contains("Cửa hàng", na=False),
        col.str.contains("Loại bất động sản khác", na=False)
    ]

    choices = [
        "Nhà biệt thự",
        "Căn hộ chung cư",
        "Nhà mặt phố",
        "Bán đất",
        "Văn phòng",
        "Nhà riêng",
        "Condotel",
        "Đất nền",
        "Shophouse",
        "Nhà trọ",
        "Chung cư mini, căn hộ",
        "Kho",
        "Trang trại",
        "Cửa hàng",
        "Loại bất động sản khác"

    ]

    data['Loại hình đất'] = np.select(conditions, choices, default=col)

    return data