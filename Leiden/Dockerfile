FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/amaliap21/ECR-2.0.git /tmp/repo \
    && cd /tmp/repo && git lfs pull --include="Leiden/**" \
    && cp -r /tmp/repo/Leiden/* /app/ \
    && rm -rf /tmp/repo

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=8080
EXPOSE 8080

CMD ["python", "main_gui.py"]
