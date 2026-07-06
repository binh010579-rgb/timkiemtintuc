"""
Crawler GỘP VnExpress + Tuổi Trẻ — MỘT file, chạy MỘT lần ra raw_news.csv.

Thứ tự chạy khi gọi `python crawler_combined.py`:
    PHA 1: VnExpress  — dùng requests, phân trang dạng /{category}-p{n}
                          (không cần cài Chrome)
    PHA 2: Tuổi Trẻ   — dùng Selenium, click nút "Xem thêm"
                          (cần có Google Chrome đã cài trên máy)

Mỗi pha tự kiểm tra số bài đã có sẵn trong raw_news.csv để tính số bài
còn thiếu, pha nào đã đủ quota thì tự bỏ qua.

LƯU Ý: phần VnExpress (slug chuyên mục, cách phân trang, API đếm bình luận)
chưa được test trực tiếp với trang thật — nếu chạy ra 0 link/0 bình luận
liên tục thì gửi log lại để chỉnh.

Cài trước khi chạy:
    pip install selenium pandas beautifulsoup4 lxml requests
"""

import time
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

OUTPUT = "raw_news.csv"
TARGET_VNEXPRESS = 25000   # số bài VnExpress muốn đạt
TARGET_TUOITRE   = 25000   # số bài Tuổi Trẻ muốn đạt
SAVE_EVERY       = 50     # lưu tạm sau mỗi N bài

VNEXPRESS_CATEGORIES = [
    "thoi-su", "the-gioi", "kinh-doanh", "khoa-hoc", "so-hoa",
    "the-thao", "giao-duc", "suc-khoe", "doi-song", "du-lich",
    "giai-tri", "phap-luat", "oto-xe-may", "bat-dong-san", "goc-nhin",
]

TUOITRE_CATEGORIES = [
    "https://tuoitre.vn/thoi-su.htm", "https://tuoitre.vn/the-gioi.htm",
    "https://tuoitre.vn/kinh-doanh.htm", "https://tuoitre.vn/cong-nghe.htm",
    "https://tuoitre.vn/the-thao.htm", "https://tuoitre.vn/giao-duc.htm",
    "https://tuoitre.vn/suc-khoe.htm", "https://tuoitre.vn/van-hoa.htm",
    "https://tuoitre.vn/giai-tri.htm", "https://tuoitre.vn/phap-luat.htm",
    "https://tuoitre.vn/du-lich.htm", "https://tuoitre.vn/khoa-hoc.htm",
    "https://tuoitre.vn/xe.htm", "https://tuoitre.vn/nha-dat.htm",
    "https://tuoitre.vn/moi-truong.htm", "https://tuoitre.vn/goc-nhin.htm",
    "https://tuoitre.vn/nhip-song-tre.htm", "https://tuoitre.vn/gia-dinh.htm",
]

ARTICLE_URL_RE_VNE = re.compile(r"-\d{6,9}\.html(\?.*)?$")
ARTICLE_URL_RE_TT = re.compile(r"-\d{10,}\.htm(\?.*)?$")


def get_soup(url, retries=3):
    """Tải trang qua requests, dùng chung cho cả 2 nguồn."""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            r.raise_for_status()
            return BeautifulSoup(r.text, "lxml")
        except Exception as e:
            print(f"    Lỗi tải {url}: {e} (thử {attempt+1}/{retries})")
            time.sleep(2)
    return None


# ═══════════════════════════════════════════════════════════════
# PHA 1: VNEXPRESS (requests, phân trang -pN)
# ═══════════════════════════════════════════════════════════════

def get_links_from_category_vne(cat, quota, max_pages=200):
    links = []
    page = 1
    empty_streak = 0

    while len(links) < quota and page <= max_pages:
        url = f"https://vnexpress.net/{cat}" if page == 1 else f"https://vnexpress.net/{cat}-p{page}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            r.raise_for_status()
        except Exception as e:
            print(f"    Lỗi tải {url}: {e}")
            empty_streak += 1
            if empty_streak >= 3:
                break
            page += 1
            time.sleep(1)
            continue

        soup = BeautifulSoup(r.text, "lxml")
        before = len(links)
        for a in soup.select("a[href]"):
            href = a.get("href", "")
            if href.startswith("/"):
                href = "https://vnexpress.net" + href
            if (href.startswith("https://vnexpress.net")
                    and ARTICLE_URL_RE_VNE.search(href)
                    and href not in links):
                links.append(href)

        gained = len(links) - before
        print(f"    [{cat} trang {page}] +{gained} link, tổng {len(links)}")

        if gained == 0:
            empty_streak += 1
            if empty_streak >= 3:
                print(f"    Hết bài / hết trang ở chuyên mục {cat} (trang {page}).")
                break
        else:
            empty_streak = 0

        page += 1
        time.sleep(0.6)

    return links[:quota]


def collect_all_links_vne(existing_links, quota):
    all_links = []
    for cat in VNEXPRESS_CATEGORIES:
        if len(all_links) >= quota:
            break
        remaining = quota - len(all_links)
        print(f"\n  Chuyên mục: {cat} (cần thêm ~{remaining} bài)")
        cat_links = get_links_from_category_vne(cat, remaining + 20)
        added = 0
        for link in cat_links:
            if link not in existing_links and link not in all_links:
                all_links.append(link)
                added += 1
        print(f"  --> {added} bài mới, tổng tích lũy: {len(all_links)}/{quota}")
    return all_links


def get_comment_count_vne(url):
    try:
        match = re.search(r"-(\d{6,9})\.html", url)
        if not match:
            return "0"
        article_id = match.group(1)
        api = (f"https://usi-saas.vnexpress.net/index/get"
               f"?offset=0&limit=0&objectid={article_id}&objecttype=1&siteid=1000000")
        r = requests.get(api, headers={**HEADERS, "Referer": url}, timeout=8)
        data = r.json()
        d = data.get("data", data) if isinstance(data, dict) else None
        if isinstance(d, dict) and "total" in d:
            return str(d["total"])
    except Exception:
        pass
    return "0"


def extract_title_vne(soup):
    h1 = soup.select_one("h1.title-detail")
    if h1 is not None:
        text = h1.get_text(strip=True)
        if text:
            return text
    meta = soup.select_one("meta[property='og:title']")
    if meta is not None:
        content = meta.get("content", "")
        if content:
            return content.strip()
    return ""


def extract_date_vne(soup):
    span = soup.select_one("span.date")
    if span is not None:
        content = span.get("content", "")
        if content:
            return content.strip()
        text = span.get_text(strip=True)
        if text:
            return text
    meta = soup.select_one("meta[property='article:published_time']")
    if meta is not None:
        content = meta.get("content", "")
        if content:
            return content.strip()
    return ""


def extract_summary_vne(soup):
    for selector in ["p.description", "meta[name='description']", "meta[property='og:description']"]:
        tag = soup.select_one(selector)
        if tag is not None:
            value = tag.get("content") if tag.name == "meta" else tag.get_text(strip=True)
            if value and str(value).strip():
                return str(value).strip()
    for p in soup.select("article.fck_detail p.Normal"):
        text = p.get_text(" ", strip=True)
        if text and len(text) >= 40:
            return text
    return ""


def extract_author_vne(soup, paras):
    meta = soup.select_one("meta[name='author']")
    if meta is not None:
        content = meta.get("content", "")
        if content and content.strip():
            return content.strip()
    for p in reversed(paras[-3:]):
        strong = p.find("strong")
        text = p.get_text(strip=True)
        if strong is not None and 0 < len(text) < 40:
            return strong.get_text(strip=True)
    return ""


def parse_article_vne(url):
    soup = get_soup(url)
    if not soup:
        return None
    try:
        title = extract_title_vne(soup)
        date = extract_date_vne(soup)
        paras = soup.select("article.fck_detail p.Normal")
        content = " ".join(p.get_text(strip=True) for p in paras)
        author = extract_author_vne(soup, paras)
        num_comments = get_comment_count_vne(url)
        summary = extract_summary_vne(soup)
        return {
            "nguon": "VnExpress", "tieu_de": title, "ngay_dang": date,
            "tac_gia": author, "noi_dung": content, "summary": summary,
            "so_binh_luan": num_comments, "link": url,
        }
    except Exception as e:
        print(f"    Lỗi parse {url}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# PHA 2: TUỔI TRẺ (Selenium, click "Xem thêm")
# ═══════════════════════════════════════════════════════════════

def make_driver():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument(f"user-agent={HEADERS['User-Agent']}")
    opts.add_argument("--log-level=3")
    driver = webdriver.Chrome(options=opts)
    driver.set_page_load_timeout(30)
    return driver


def get_links_from_category_tt(driver, cat_url, quota, max_clicks=80):
    links = []
    print(f"    Mở: {cat_url}")
    try:
        driver.get(cat_url)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='.htm']"))
        )
    except Exception as e:
        print(f"    Timeout/lỗi khi tải trang: {e}")
        return links

    no_new_streak = 0
    for click_i in range(max_clicks + 1):
        before_count = len(links)
        soup = BeautifulSoup(driver.page_source, "lxml")
        for a in soup.select("a[href]"):
            href = a.get("href", "")
            if href.startswith("/"):
                href = "https://tuoitre.vn" + href
            if (href.startswith("https://tuoitre.vn")
                    and ARTICLE_URL_RE_TT.search(href)
                    and href not in links):
                links.append(href)
        print(f"      [click {click_i}] tổng tích lũy: {len(links)}")

        if len(links) >= quota:
            print(f"      Đủ {quota} bài, dừng click.")
            break

        clicked = False
        try:
            btns = driver.find_elements(By.XPATH,
                "//*[contains(text(),'Xem thêm') or contains(text(),'xem thêm')]"
                "[not(ancestor::*[@style='display:none'])]"
            )
            for btn in btns:
                if btn.is_displayed():
                    driver.execute_script(
                        "arguments[0].scrollIntoView({block:'center'});", btn)
                    time.sleep(0.4)
                    driver.execute_script("arguments[0].click();", btn)
                    clicked = True
                    print(f"      [click {click_i}] Đã click 'Xem thêm'")
                    break
        except Exception as e:
            print(f"      [click {click_i}] Lỗi click: {e}")

        time.sleep(1.5)

        if not clicked or len(links) == before_count:
            no_new_streak += 1
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            if no_new_streak >= 3:
                print("      Không có tiến triển 3 lần liên tiếp, dừng.")
                break
        else:
            no_new_streak = 0

    return links[:quota]


def collect_all_links_tt(existing_links, quota):
    all_links = []
    driver = make_driver()
    try:
        for cat_url in TUOITRE_CATEGORIES:
            if len(all_links) >= quota:
                break
            remaining = quota - len(all_links)
            print(f"\n  Chuyên mục: {cat_url} (cần thêm ~{remaining} bài)")
            cat_links = get_links_from_category_tt(driver, cat_url, remaining + 20)
            added = 0
            for link in cat_links:
                if link not in existing_links and link not in all_links:
                    all_links.append(link)
                    added += 1
            print(f"  --> {added} bài mới, tổng tích lũy: {len(all_links)}/{quota}")
    finally:
        driver.quit()
    return all_links


def get_comment_count_tt(url):
    try:
        match = re.search(r"-(\d{10,})\.htm$", url)
        if not match:
            return "0"
        article_id = match.group(1)
        api = (f"https://id.tuoitre.vn/api/getlist-comment.api"
               f"?pageindex=1&pagesize=500&objId={article_id}&objType=1&sort=2")
        r = requests.get(api, headers={**HEADERS, "Referer": url}, timeout=8)
        data = r.json()
        if isinstance(data, dict):
            raw = data.get("Data", "[]")
            if isinstance(raw, str):
                import json
                raw = json.loads(raw)
            if isinstance(raw, list):
                total = sum(1 + (c.get("child_count") or 0) for c in raw if isinstance(c, dict))
                return str(total)
    except Exception:
        pass
    return "0"


def extract_summary_tt(soup):
    if not soup:
        return ""
    for selector in [
        "meta[name='description']", "meta[property='og:description']",
        "meta[name='twitter:description']", "meta[itemprop='description']",
    ]:
        tag = soup.select_one(selector)
        if tag is not None:
            value = tag.get("content") if tag.name == "meta" else tag.get_text(strip=True)
            if value and str(value).strip():
                return str(value).strip()
    for selector in [
        "div.detail-sapo", "span.detail-sapo", "div.sapo", "p.lead",
        "div.article-sapo", "div.trichdan", "p.description", "div.description",
    ]:
        tag = soup.select_one(selector)
        if tag is not None:
            value = tag.get_text(" ", strip=True)
            if value and str(value).strip():
                return str(value).strip()
    for selector in [
        "div.detail-content.afcbc-body p", "div.detail-content p",
        "article p", "div.fck_detail p", "div.content p",
    ]:
        for p in soup.select(selector):
            text = p.get_text(" ", strip=True)
            if text and len(text) >= 40:
                return text
    for p in soup.select("p"):
        text = p.get_text(" ", strip=True)
        if text and len(text) >= 40:
            return text
    return ""


def parse_article_tt(url):
    soup = get_soup(url)
    if not soup:
        return None
    try:
        title_tag = soup.select_one("h1.detail-title")
        if title_tag is None:
            title_tag = soup.select_one("h1[data-role='title']")
        title = title_tag.get_text(strip=True) if title_tag is not None else ""

        date_tag = soup.select_one("time[data-role='publishdate']")
        date = date_tag.get_text(strip=True) if date_tag is not None else ""

        paras = soup.select("div.detail-content.afcbc-body p")
        content = " ".join(p.get_text(strip=True) for p in paras)

        author_tag = soup.select_one("div.detail-author-bot a.name")
        author = author_tag.get_text(strip=True) if author_tag is not None else ""

        num_comments = get_comment_count_tt(url)
        summary = extract_summary_tt(soup)

        return {
            "nguon": "Tuổi Trẻ", "tieu_de": title, "ngay_dang": date,
            "tac_gia": author, "noi_dung": content, "summary": summary,
            "so_binh_luan": num_comments, "link": url,
        }
    except Exception as e:
        print(f"    Lỗi parse {url}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# MAIN — chạy lần lượt PHA 1 (VnExpress) rồi PHA 2 (Tuổi Trẻ)
# ═══════════════════════════════════════════════════════════════

def save_progress(all_data, label=""):
    df_tmp = pd.DataFrame(all_data).drop_duplicates(subset=["link"])
    df_tmp.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"  --> Lưu tạm{label}: {len(df_tmp)} bài tổng cộng")
    return df_tmp


if __name__ == "__main__":
    try:
        df_old = pd.read_csv(OUTPUT, encoding="utf-8-sig")
        all_data = df_old.to_dict("records")
        existing_links = set(df_old["link"].tolist())
        vne_existing = len(df_old[df_old["nguon"] == "VnExpress"])
        tt_existing = len(df_old[df_old["nguon"] == "Tuổi Trẻ"])
        print(f"Load {len(all_data)} bài cũ ({vne_existing} VnExpress, {tt_existing} Tuổi Trẻ) từ {OUTPUT}")
    except FileNotFoundError:
        all_data = []
        existing_links = set()
        vne_existing = 0
        tt_existing = 0
        print("Chưa có file CSV, bắt đầu mới.")

    need_vne = TARGET_VNEXPRESS - vne_existing
    need_tt = TARGET_TUOITRE - tt_existing

    if need_vne <= 0 and need_tt <= 0:
        print(f"Đã đủ {TARGET_VNEXPRESS} VnExpress + {TARGET_TUOITRE} Tuổi Trẻ rồi, không cần chạy thêm.")
        exit()

    # ───── PHA 1: VNEXPRESS ─────
    if need_vne > 0:
        print(f"\n{'='*50}\nPHA 1: VNEXPRESS — cần thêm {need_vne} bài\n{'='*50}")
        new_links_vne = collect_all_links_vne(existing_links, need_vne)
        print(f"\nTổng link VnExpress mới gom được: {len(new_links_vne)}")
        for i, link in enumerate(new_links_vne, 1):
            print(f"  [VNE {i}/{len(new_links_vne)}] {link}")
            data = parse_article_vne(link)
            if data:
                all_data.append(data)
                existing_links.add(link)
            if i % SAVE_EVERY == 0:
                save_progress(all_data)
            time.sleep(0.4)
        save_progress(all_data, " (xong pha VnExpress)")
    else:
        print(f"\nĐã đủ {TARGET_VNEXPRESS} bài VnExpress, bỏ qua PHA 1.")

    # ───── PHA 2: TUỔI TRẺ ─────
    if need_tt > 0:
        print(f"\n{'='*50}\nPHA 2: TUỔI TRẺ — cần thêm {need_tt} bài\n{'='*50}")
        new_links_tt = collect_all_links_tt(existing_links, need_tt)
        print(f"\nTổng link Tuổi Trẻ mới gom được: {len(new_links_tt)}")
        for i, link in enumerate(new_links_tt, 1):
            print(f"  [TT {i}/{len(new_links_tt)}] {link}")
            data = parse_article_tt(link)
            if data:
                all_data.append(data)
                existing_links.add(link)
            if i % SAVE_EVERY == 0:
                save_progress(all_data)
            time.sleep(0.5)
        save_progress(all_data, " (xong pha Tuổi Trẻ)")
    else:
        print(f"\nĐã đủ {TARGET_TUOITRE} bài Tuổi Trẻ, bỏ qua PHA 2.")

    # ───── TỔNG KẾT ─────
    df_final = pd.DataFrame(all_data).drop_duplicates(subset=["link"])
    df_final.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    vne = len(df_final[df_final["nguon"] == "VnExpress"])
    tt = len(df_final[df_final["nguon"] == "Tuổi Trẻ"])
    print(f"\n{'='*50}")
    print(f"HOÀN TẤT! Tổng {len(df_final)} bài")
    print(f"  VnExpress : {vne}")
    print(f"  Tuổi Trẻ  : {tt}")
    if len(df_final) < 15000:
        print(f"  Thiếu {15000 - len(df_final)} bài để đạt 15000")
    else:
        print(f"  Đã đủ 15000+ bài!")
