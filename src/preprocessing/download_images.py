# download_images_async.py
import aiohttp
import asyncio
import aiofiles
import pandas as pd
import os
from tqdm import tqdm
from urllib.parse import urlparse
from pathlib import Path

# ==== CONFIG ====
CSV_PATH = r"D:\Amazon\dataset\raw\train.csv"
IMG_FOLDER = r"D:\Amazon\dataset\images\train_mapped"   # where images will be saved as <sample_id>.<ext>
LOG_FAILED_PATH = r"D:\Amazon\output\failed_downloads.txt"
MAX_CONCURRENCY = 100
TIMEOUT_SEC = 20
# ================

os.makedirs(IMG_FOLDER, exist_ok=True)
Path(LOG_FAILED_PATH).parent.mkdir(parents=True, exist_ok=True)

def get_ext_from_url(url):
    # fallback to .jpg if none
    try:
        path = urlparse(url).path
        ext = os.path.splitext(path)[1]
        if ext and len(ext) <= 5:
            return ext
    except Exception:
        pass
    return ".jpg"

async def download_image(session, sem, sid, url):
    ext = get_ext_from_url(url)
    outpath = os.path.join(IMG_FOLDER, f"{sid}{ext}")
    # If file already exists (maybe from previous run), skip
    if os.path.exists(outpath):
        return None
    async with sem:
        try:
            async with session.get(url, timeout=TIMEOUT_SEC) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    # small safety: skip empty content
                    if content:
                        async with aiofiles.open(outpath, 'wb') as f:
                            await f.write(content)
                        return None
                # non-200 or empty -> failure
                return (sid, url, resp.status)
        except Exception as e:
            return (sid, url, str(e))

async def main():
    df = pd.read_csv(CSV_PATH)
    pairs = df[['sample_id', 'image_link']].values.tolist()
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    connector = aiohttp.TCPConnector(limit_per_host=MAX_CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=None)
    failed = []

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [download_image(session, sem, sid, url) for sid, url in pairs]
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Downloading"):
            res = await fut
            if res is not None:
                failed.append(res)

    # Save failed log using UTF-8
    if failed:
        with open(LOG_FAILED_PATH, "w", encoding="utf-8") as f:
            for sid, url, err in failed:
                f.write(f"{sid},{url},{err}\n")
        print(f"WARNING: {len(failed)} downloads failed. Logged to {LOG_FAILED_PATH}")
    else:
        print("All images downloaded (or already present).")

if __name__ == "__main__":
    asyncio.run(main())
