FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install cython \
    && pip install scikit-surprise --no-build-isolation || true

COPY src/       ./src/
COPY api/       ./api/
COPY models/    ./models/
COPY data/processed/ ./data/processed/
COPY data/raw/ml-100k/u.item ./data/raw/ml-100k/u.item

EXPOSE 5001
ENV PYTHONPATH=/app

CMD ["python", "api/app.py"]