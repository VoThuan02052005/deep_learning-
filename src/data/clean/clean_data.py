"""
CHUẨN HÓA DỮ LIỆU THÔ (TRANSFORM STAGE)

Mục tiêu:
- Làm sạch và chuẩn hóa dữ liệu bất động sản sau bước crawl / ingest
- Chuyển dữ liệu thô về dạng số, chuẩn định dạng để dùng cho:
  + validate
  + feature engineering
  + modeling

Các bước chính:
- Xử lý giá trị thiếu / sai định dạng
- Chuẩn hóa giá, diện tích, số phòng, số tầng,...
- Phân tích và chuẩn hóa ngày đăng
- Loại bỏ các bản ghi trùng lặp

Lưu ý:
- KHÔNG lọc theo loại giao dịch ở bước này.
- Việc tách Bán / Cho Thuê được thực hiện ở run_pipeline_clean.py sau validate.

Output:
- staging/data_ban.csv
- staging/data_cho_thue.csv
"""


from .clean_transaction import *
from .clean_area import *
from .clean_price import *
from .clean_floor import *
from .clean_facade import *
from .posted_date import *
from .clean_entrance import *
from .clean_bathroom import *
from .clean_bedroom import *
from .crawl_date import *
# clean_land (xoa_giao_dich_cho_thue) được chuyển sang run_pipeline_clean.py
from .duplicates import *
from .duplicates__column import *

from src.utils.logger import Logger


logger = Logger("transform")

def clean_data(data):
    """
    Thực hiện bước TRANSFORM trong pipeline dữ liệu.

    Input:
    - data (pd.DataFrame): dữ liệu thô sau bước clean cơ bản

    Output:
    - pd.DataFrame: dữ liệu đã được chuẩn hóa, sẵn sàng cho validate / feature engineering
    """
    logger.info("Start transformer data ...")
    logger.info(f"Initial shape: {data.shape}")
    k = len(data)
    logger.info(f"số bản ghi trước khi xóa: {k}")
    try:
        data = loai_hinh_dat(data)
        data = dien_tich(data)
        data = muc_gia(data)
        data = so_phong_ngu(data)
        data = so_tang(data)
        data = so_phong_tam(data)
        data = mat_tien(data)
        data = duong_vao(data)
        data = ngay_dang(data)
        data = crawl_date(data)
        data = xoa_trung_lap(data)
        data = xoa_trung_lap_theo_cot(
            data,
            subset=[
                "Loại giao dịch", "Thành phố", "Quận/huyện", "Loại hình đất",
                "Mức giá", "Diện tích", "Số phòng ngủ", "Số phòng tắm, vệ sinh",
                "Số tầng", "Hướng nhà", "Hướng ban công", "Mặt tiền",
                "Đường vào", "Pháp lý", "Nội thất", "Ngày đăng"
            ],
            keep="first"
        )

    except Exception as e:
        logger.exception(f"Transformer failed: {e}")
    g = len(data)
    logger.info(f"số bản ghi đã xóa: {k - g}")
    logger.info(f"số bản ghi sau khi xóa: {g}")
    logger.info(f"Final shape: {data.shape}")
    return data