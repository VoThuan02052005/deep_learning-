"""
Trích xuất dữ liệu bất động sản thô từ nguồn (CSV / Web Scraping).
Output: data_raw/
"""
# khai báo thư viện cần thiết
import time
import random
import csv
import os
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import sqlite3
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from src.utils.logger import Logger
import pandas as pd
# ==== CONFIG ====
CONFIG_PATH = "config.yaml"

def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config()

CRAWL_CONFIG = config["CRAWL"]

START_PAGE = CRAWL_CONFIG["START_PAGE"]
END_PAGE = CRAWL_CONFIG["END_PAGE"]
BATCH_SIZE = CRAWL_CONFIG["BATCH_SIZE"]
SLEEP_MIN = CRAWL_CONFIG["SLEEP_BETWEEN_BATCH_MIN"]
SLEEP_MAX = CRAWL_CONFIG["SLEEP_BETWEEN_BATCH_MAX"]

USER_DATA_DIR = config["USER_DATA_DIR"]
PROFILE_DIR = config["PROFILE_DIR"]
CSV_FILE = "data/raw/gia_nha.csv"


logger = Logger("extract")

def crawl_batch(base_url, start_page, end_page):
    logger.info(f"Crawl batch: trang {start_page} → {end_page}")
    return crawl_all_listing_urls(
        base_url=base_url,
        pages_start=start_page,
        pages_end=end_page
    )

# ==== UTILS ====
def create_driver():
    options = Options()
    prefs = {"profile.default_content_setting_values.notifications": 2}
    options.add_experimental_option("prefs", prefs)
    options.add_argument(USER_DATA_DIR)
    options.add_argument(PROFILE_DIR)
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--disable-blink-features=AutomationControlled")
    service = Service(ChromeDriverManager().install())


    return webdriver.Chrome(service=service, options=options)




def random_wait(a=2, b=5):
    time.sleep(random.uniform(a, b))

# ==== SCRAPER ====
def get_page_links(driver):
    soup = BeautifulSoup(driver.page_source, "html.parser")
    links = soup.find_all("a", class_="js__product-link-for-product-id")
    full_links = []
    for link in links:
        href = link.get('href')
        if href and ("https://batdongsan.com.vn" + href) not in full_links:
            full_links.append("https://batdongsan.com.vn" + href)
    return full_links

def crawl_all_listing_urls(base_url, pages_start, pages_end):
    all_links = []
    for i in range(pages_start, pages_end + 1):
        current_url = base_url + f"/p{i}"
        logger.info(f"Đang tải trang {i}")
        driver = create_driver()
        try:
            driver.get(current_url)
            page_links = get_page_links(driver)
            if not page_links:
                logger.info(f"Không có bài đăng ở trang {i}.")
                continue
            all_links.extend(page_links)
        except Exception as e:
            logger.error(f"Lỗi khi tải trang {i}: {str(e)}")
            break
        finally:
            driver.quit()
    return all_links

def parse_listing_info(url):
    driver = create_driver()

    try:

        driver.get(url)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        thong_tin = []
        try:
            info_div_1 = soup.find("div", class_="re__breadcrumb")
            links = info_div_1.find_all("a", class_="re__link-se")
            thong_tin = [a.get_text(strip=True) for a in links]
        except Exception as e:
            logger.error(f"Lỗi breadcrumb: {str(e)}")

        data = {
            'Loại giao dịch': thong_tin[0] if len(thong_tin) > 0 else '',
            'Thành phố': thong_tin[1] if len(thong_tin) > 1 else '',
            'Quận/huyện': thong_tin[2] if len(thong_tin) > 2 else '',
            'Loại hình đất': thong_tin[3] if len(thong_tin) > 3 else '',
        }

        try:
            info_div_2 = soup.find("div", class_="re__pr-other-info-display")
            specs = info_div_2.find_all("div", class_="re__pr-specs-content-item")
            for item in specs:
                title = item.find("span", class_="re__pr-specs-content-item-title").get_text(strip=True)
                value = item.find("span", class_="re__pr-specs-content-item-value").get_text(strip=True)
                data[title] = value
        except Exception as e:
            logger.error(f"Lỗi phân tích thông tin chi tiết: {str(e)}")

        try:
            info_div_4 = soup.find("div", class_="re__pr-short-info-item js__pr-config-item")
            if info_div_4:
                list_text_3 = info_div_4.find("span", class_="value").get_text().strip()
                title_4 = info_div_4.find("span", class_="title").get_text().strip()
                data[title_4] = list_text_3
        except Exception as e:
            logger.error(f"Lỗi ngắn gọn: {str(e)}")
        crawl_date = datetime.now().strftime("%Y-%m-%d")
        data["crawl_date"] = crawl_date

        return data
    finally:
        driver.quit()

# Đa luồng để crawl dữ liệu nhanh hơn
def scrape_details(link_list, max_workers=5):
    all_data = []

    def safe_parse(url):
        try:
            return parse_listing_info(url)
        except Exception as e:
            logger.info(f"Lỗi khi crawl {url}: {str(e)}")
            return {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(safe_parse, link): link for link in link_list}
        for idx, future in enumerate(as_completed(futures)):
            url = futures[future]
            try:
                data = future.result()
                if data:
                    all_data.append(data)
                logger.info(f"Đã crawl xong bài {idx+1}/{len(link_list)}")
            except Exception as e:
                logger.info(f"Lỗi ở bài {url}: {str(e)}")

    return all_data

def save_to_csv(data, file_path=CSV_FILE):
    headers = [
        'Loại giao dịch', 'Thành phố', 'Quận/huyện', 'Loại hình đất',
        'Mức giá', 'Diện tích', 'Số phòng ngủ', 'Số phòng tắm, vệ sinh',
        'Số tầng', 'Hướng nhà', 'Hướng ban công', 'Mặt tiền',
        'Đường vào', 'Pháp lý', 'Nội thất', 'Ngày đăng', 'crawl_date'
    ]

    file_exists = os.path.isfile(file_path)

    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=headers)

        if not file_exists:
            writer.writeheader()

        for row in data:
            row_data = {}

            for h in headers:
                if h == 'Mức giá':
                    row_data[h] = row.get('Mức giá') or row.get('Khoảng giá', '')
                else:
                    row_data[h] = row.get(h, '')

            writer.writerow(row_data)

    logger.info(f"Dữ liệu đã được lưu vào {file_path}")


# ==== DATABASE ====
def init_db():
    db_path = "urls.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS crawled_urls (
            url TEXT PRIMARY KEY
        )
    """)
    conn.commit()
    conn.close()

def is_new_url(url):
    db_path = "urls.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM crawled_urls WHERE url = ?", (url,))
    result = cursor.fetchone()
    conn.close()
    return result is None

def save_new_url(url):
    db_path = "urls.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO crawled_urls (url) VALUES (?)", (url,))
    conn.commit()
    conn.close()

def filter_new_links(link_list):
    return [url for url in link_list if is_new_url(url)]

# ==== MAIN ====
def extract():
    logger.info("Start extracting raw data")
    init_db()

    urls = {
        "Thuê": "https://batdongsan.com.vn/nha-dat-cho-thue"
    }

    for loai, base_url in urls.items():
        logger.info(f"\n=== Bắt đầu crawl loại: {loai} ===")

        page = START_PAGE
        while page <= END_PAGE:
            batch_start = page
            batch_end = min(page + BATCH_SIZE - 1, END_PAGE)

            # 1️⃣ Crawl link
            links = crawl_batch(base_url, batch_start, batch_end)

            # 2️⃣ Lọc link mới
            new_links = filter_new_links(links)
            logger.info(
                f"Trang {batch_start}-{batch_end} | Tổng link: {len(links)} | Mới: {len(new_links)}"
            )

            # 3️⃣ Crawl chi tiết
            if new_links:
                data = scrape_details(new_links, max_workers=5)

                for link in new_links:
                    save_new_url(link)

                save_to_csv(data, file_path="data/raw/gia_nha.csv")
            else:
                logger.info("Không có bài mới trong batch này")

            # 4️⃣ Nghỉ giữa các batch
            sleep_time = random.uniform(SLEEP_MIN, SLEEP_MAX)
            logger.info(f"Nghỉ {sleep_time:.0f} giây trước batch tiếp theo")
            time.sleep(sleep_time)

            page += BATCH_SIZE

        logger.info(f"Hoàn tất crawl loại: {loai}")


if __name__ == "__main__":
    extract()

