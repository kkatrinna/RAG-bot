#  AI Platform: RAG Bot + ETL Pipeline

##  О проекте

Это полноценная AI-платформа, объединяющая два современных направления:

1. **RAG (Retrieval-Augmented Generation) Bot** — умный чат-бот, который отвечает на вопросы по вашим документам (PDF, TXT) с цитированием источников
2. **ETL Pipeline** — автоматический сбор данных о погоде и криптовалютах с их последующей визуализацией

Платформа демонстрирует навыки работы с:
-  LLM (Llama 3, Saiga)
-  Векторными базами данных (FAISS)
-  Оркестрацией пайплайнов (Prefect)
-  Визуализацией данных (Metabase, Plotly)
-  Контейнеризацией (Docker)

## Возможности
- Загрузка PDF и TXT файлов
- Разбивка на чанки с перекрытием
- Поиск по смыслу (векторный поиск)
- Генерация ответов с цитированием
- Локальная работа (без API-ключей)

## Пример работы
https://pix-up.ru/uploads/img_6a2aad637c9789.53561140_1781181795.jpg

## Быстрый старт

### Требования

- Python 3.11+
- Docker Desktop (опционально, для ETL)
- 8+ GB RAM
- 10+ GB свободного места

### Установка

```bash
# Клонирование репозитория
git clone https://github.com/yourusername/ai-etl-platform.git
cd ai-etl-platform

# Создание виртуального окружения
python -m venv .venv

# Активация (Windows)
.venv\Scripts\activate

# Активация (Linux/Mac)
source .venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt

### Установка
