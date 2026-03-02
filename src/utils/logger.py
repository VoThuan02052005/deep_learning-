from pathlib import Path
from datetime import datetime
import sys

# Khởi tạo thư mục log ngay tại cấp module
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


class Logger:
    def __init__(self, name: str = "experiment", log_to_file: bool = True):
        self.name = name
        self.log_to_file = log_to_file
        # Tạo timestamp một lần duy nhất khi bắt đầu một phiên chạy benchmark
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = LOG_DIR / f"{name}_{timestamp}.log"

    def _write(self, level: str, msg: str):
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Thêm tên logger (self.name) vào dòng log để dễ phân biệt
        line = f"[{time_str}] [{self.name}] [{level}] {msg}"

        # In ra console
        print(line)

        # Ghi vào file nếu được yêu cầu
        if self.log_to_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception as e:
                print(f"Failed to write log to file: {e}")

    def info(self, msg: str):
        self._write("INFO", msg)

    def warning(self, msg: str):
        self._write("WARNING", msg)

    def error(self, msg: str):
        self._write("ERROR", msg)

    def success(self, msg: str):
        """Thêm level SUCCESS để đánh dấu khi train xong hoặc lưu model xong"""
        self._write("SUCCESS", msg)
