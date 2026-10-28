"""流式拉取 TinyStoriesV2-GPT4-train.txt 的前 64MB 并按故事边界切分。

官方数据 2.23GB，HuggingFace 在本机不可达，改用 Gitee 镜像；
该镜像不支持 Range 请求，因此只读取前 64MB 后断开连接，避免下载全量。
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path

URL = "https://gitee.com/hf-datasets/TinyStories/raw/main/TinyStoriesV2-GPT4-train.txt"
RAW = Path("/tmp/ts_64m.bin")
MAX_BYTES = 64 * 1024 * 1024
TRAIN_BYTES = 50 * 1024 * 1024
SEP = "<|endoftext|>"
DATA_DIR = Path(__file__).parent / "data"


def download() -> None:
    if RAW.exists() and RAW.stat().st_size >= MAX_BYTES:
        print(f"[Download] 复用 {RAW}")
        return
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req)
    t0 = time.perf_counter()
    got = mark = 0
    with RAW.open("wb") as f:
        while got < MAX_BYTES:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if got - mark >= 8 << 20:
                print(f"[Download] {got / 2**20:5.0f} MB  {time.perf_counter() - t0:5.1f}s")
                mark = got
    resp.close()
    print(f"[Download] 完成 {got / 2**20:.1f} MB，用时 {time.perf_counter() - t0:.1f}s")


def split() -> None:
    raw = RAW.read_bytes()
    stories = [s.decode("utf-8", errors="ignore").strip() for s in raw.split(SEP.encode())]
    stories = [s for s in stories[:-1] if s]  # 丢弃末尾可能被截断的故事

    train, valid, acc = [], [], 0
    for story in stories:
        if acc < TRAIN_BYTES:
            train.append(story)
            acc += len(story) + 1
        else:
            valid.append(story)

    DATA_DIR.mkdir(exist_ok=True)
    for name, parts in [("tinystories_train_50M.txt", train), ("tinystories_valid_14M.txt", valid)]:
        path = DATA_DIR / name
        path.write_text("\n".join(parts), encoding="utf-8")
        print(f"[Split] {name}: {path.stat().st_size / 2**20:.1f} MB, {len(parts):,} 条故事")


if __name__ == "__main__":
    download()
    split()
