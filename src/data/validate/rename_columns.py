def rename_columns(data):
    """
    Đổi tên các cột dữ liệu từ tiếng Việt sang tiếng Anh
    để thống nhất schema cho các bước xử lý và huấn luyện mô hình.
    """
    column_mapping = {
        "Loại giao dịch": "transaction_type",
        "Giá": "price",
        "Đơn vị(Mức giá)": "price_unit",
        "Diện tích": "area",
        "Số phòng ngủ": "bedrooms",
        "Số phòng tắm, vệ sinh": "bathrooms",
        "Số tầng": "floors",
        "Mặt tiền": "frontage",
        "Đường vào": "road_width",
        "Thành phố": "city",
        "Quận/huyện": "district",
        "Loại hình đất": "property_type",
        "Pháp lý": "legal_status",
        "Nội thất": "interior",
        "Hướng nhà": "house_direction",
        "Hướng ban công": "balcony_direction",
        "Ngày đăng": "posted_date"
    }

    data = data.copy()
    data = data.rename(columns=column_mapping)
    return data

