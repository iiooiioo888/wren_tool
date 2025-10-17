# PoC 與自測樣板

此目錄下包含最小可行 PoC 腳本與自動化測試樣板，方便您快速驗證回測流程。

快速開始：

1. 建議建立虛擬環境並安裝依賴：

```powershell
python -m venv .venv; .\.venv\Scripts\Activate; pip install -r requirements.txt
```

2. 執行 PoC：

```powershell
python scripts/poc_run.py --csv data/sample_ohlc.csv --out out/poc_results.json
```

3. 執行測試：

```powershell
pytest -q
```
