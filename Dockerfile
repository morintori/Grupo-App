FROM python:3.10.9-alpine AS builder
COPY requirements.txt .
RUN apk --no-cache add musl-dev g++ && pip install --upgrade pip && pip install -r requirements.txt
COPY . .

EXPOSE 8050
ENV PORT 8050

CMD ["python", "app.py"]
