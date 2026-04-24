FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /repo
COPY . .
RUN git lfs pull --include="Leiden/finetuned_sentiment_cardiff_xlmroberta/**,Leiden/pipeline_out/**"

WORKDIR /app
RUN cp -r /repo/Leiden/* /app/ && rm -rf /repo

COPY Leiden/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=8080
EXPOSE 8080

CMD ["python", "main_gui.py"]
