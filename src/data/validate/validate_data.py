from .validate_price import *
from .rename_columns import *
from .validate_null_column import *
from .validate_area import *
from .validate_location import *
from .missing_values import *
import pandas as pd


def validate_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Thực hiện bước VALIDATE trong pipeline dữ liệu.

    Các bước:
    - Kiểm tra và lọc diện tích
    - Chuẩn hóa và validate giá
    - Kiểm tra thông tin vị trí
    - Chuẩn hóa tên cột
    - Loại bỏ các cột có tỷ lệ null cao
    """


    try:


        data = validate_area(data)
        data = validate_price(data)
        data = validate_location(data)
        data = rename_columns(data)
        data = xoa_cot_null_nhieu(data)

        data = clean_missing_values(data)
    except ValueError as e:
        raise

    return data


