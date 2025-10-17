FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
CMD ["python", "scripts/poc_run.py", "--csv", "data/sample_ohlc.csv", "--out", "out/poc_results.json"]
