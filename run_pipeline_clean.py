import logging
import pandas as pd
from src.data.clean.clean_data import clean_data
from src.data.validate.validate_data import validate_data
from src.utils.io import save_csv

# Cấu hình logging theo chuẩn best practice
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("pipeline_clean")

if __name__ == "__main__":
    logger.info("Bắt đầu quá trình làm sạch dữ liệu...")

    # 1. Tải dữ liệu thô
    raw_path = "data/raw/gia_nha.csv"
    logger.info(f"Đang tải dữ liệu từ: {raw_path}")
    data = pd.read_csv(raw_path)

    initial_count = len(data)
    logger.info(f"Số lượng dữ liệu ban đầu: {initial_count} dòng")

    # 2. Bước làm sạch (Clean)
    logger.info("Đang thực hiện bước làm sạch (Clean)...")
    data = clean_data(data)
    after_clean_count = len(data)
    removed_clean = initial_count - after_clean_count
    logger.info(f"Sau khi làm sạch: còn {after_clean_count} dòng (Đã loại bỏ: {removed_clean} dòng)")

    # 3. Bước kiểm định (Validate)
    logger.info("Đang thực hiện bước kiểm định (Validate)...")
    data = validate_data(data)
    final_count = len(data)
    removed_validate = after_clean_count - final_count
    logger.info(f"Sau khi kiểm định: còn {final_count} dòng (Đã loại bỏ thêm: {removed_validate} dòng)")

    total_removed = initial_count - final_count
    logger.info(f"Tổng cộng đã loại bỏ: {total_removed} dòng")

    # 4. Tách bộ dữ liệu: Bán vs Cho Thuê
    # Cột 'transaction_type' đã được rename từ 'Loại giao dịch' ở bước validate
    logger.info("Đang tách dữ liệu thành 2 bộ: Bán và Cho Thuê...")

    mask_cho_thue = data["transaction_type"].astype(str).str.lower().str.contains("thuê", na=False)

    data_cho_thue = data[mask_cho_thue].copy()
    data_ban      = data[~mask_cho_thue].copy()

    logger.info(f"Bộ BÁN:       {len(data_ban):,} dòng")
    logger.info(f"Bộ CHO THUÊ:  {len(data_cho_thue):,} dòng")

    # 5. Lưu 2 bộ dữ liệu riêng biệt
    path_ban      = "data/staging/data_ban.csv"
    path_cho_thue = "data/staging/data_cho_thue.csv"

    save_csv(data_ban,      path_ban)
    save_csv(data_cho_thue, path_cho_thue)

    logger.info(f"Đã lưu bộ BÁN       → {path_ban}")
    logger.info(f"Đã lưu bộ CHO THUÊ  → {path_cho_thue}")
    logger.info("Pipeline hoàn tất.")
