FROM python:3.10-slim

# Устанавливаем curl (нужен для скачивания модели)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Сначала зависимости (кэширование слоёв)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем проект
COPY . .

# Создаём папку для артефактов
RUN mkdir -p artifacts

# Скачиваем модель из GitHub Release
RUN curl -L -o artifacts/model.cbm \
    https://github.com/Vladanders7/recommendation_system/releases/download/v1.0.0/model.cbm

EXPOSE 8000

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]