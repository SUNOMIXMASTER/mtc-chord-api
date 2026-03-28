FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install Cython==3.0.10 numpy==1.26.4
RUN pip install -r requirements.txt

COPY . .

CMD gunicorn app:app --workers 2 --timeout 120 --bind 0.0.0.0:$PORT
