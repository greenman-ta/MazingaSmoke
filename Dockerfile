FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 torchvision==0.19.1 \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "import api.src.inference" 
EXPOSE 8000
CMD ["uvicorn", "api.src.main:app", "--host", "0.0.0.0", "--port", "8000"]