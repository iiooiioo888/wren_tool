"""簡單的 Load Test 腳本：
- 使用 multiprocessing 啟動多個 PoC 進程來模擬同時執行多個回測
- 這是輕量級的模擬，並非真實的高頻壓力測試工具
"""
from multiprocessing import Process
import os
import time

from scripts.poc_run import run_poc


def worker(id, csv_path):
    out = f"e:/Jerry_python/wren_tool/out/poc_worker_{id}.json"
    try:
        run_poc(csv_path, out_path=out)
    except Exception as e:
        print(f"worker {id} failed: {e}")


def run_load(n_workers=4, csv_path=None):
    csv_path = csv_path or os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "sample_ohlc.csv"))
    procs = []
    for i in range(n_workers):
        p = Process(target=worker, args=(i, csv_path))
        p.start()
        procs.append(p)
        time.sleep(0.1)  # staggered start
    for p in procs:
        p.join()


if __name__ == '__main__':
    run_load(n_workers=4)
