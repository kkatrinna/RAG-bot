from prefect import flow, task, get_run_logger
from datetime import timedelta, datetime
import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from etl.extract import DataExtractor
from etl.transform import DataTransformer
from etl.load import DataLoader


@task(name="extract_weather", retries=3, retry_delay_seconds=10)
def extract_weather_task():
    logger = get_run_logger()
    logger.info(" Запуск задачи выгрузки погоды")

    extractor = DataExtractor()
    df = extractor.extract_weather()

    logger.info(f"Выгружено {len(df)} записей о погоде")
    return df


@task(name="extract_crypto", retries=3, retry_delay_seconds=10)
def extract_crypto_task():
    logger = get_run_logger()
    logger.info("Запуск задачи выгрузки криптовалют")

    extractor = DataExtractor()
    df = extractor.extract_crypto()

    if df.is_empty():
        logger.warning("Не удалось получить данные о криптовалютах")
        return df

    logger.info(f"Выгружено {len(df)} записей о криптовалютах")
    return df


@task(name="transform_weather")
def transform_weather_task(df):
    logger = get_run_logger()
    logger.info("Трансформация данных о погоде")

    if df.is_empty():
        logger.warning("Нет данных для трансформации")
        return df

    transformer = DataTransformer()
    transformed_df = transformer.transform_weather(df)

    logger.info(f"Трансформировано {len(transformed_df)} записей")
    return transformed_df


@task(name="transform_crypto")
def transform_crypto_task(df):
    logger = get_run_logger()
    logger.info("Трансформация данных о криптовалютах")

    if df.is_empty():
        logger.warning("Нет данных для трансформации")
        return df

    transformer = DataTransformer()
    transformed_df = transformer.transform_crypto(df)

    logger.info(f"Трансформировано {len(transformed_df)} записей")
    return transformed_df


@task(name="load_weather")
def load_weather_task(df):
    logger = get_run_logger()
    logger.info("Загрузка данных о погоде")

    if df.is_empty():
        logger.warning("Нет данных для загрузки")
        return 0

    loader = DataLoader()
    count = loader.load_weather(df)

    logger.info(f"Загружено {count} записей")
    return count


@task(name="load_crypto")
def load_crypto_task(df):
    logger = get_run_logger()
    logger.info("Загрузка данных о криптовалютах")

    if df.is_empty():
        logger.warning("Нет данных для загрузки")
        return 0

    loader = DataLoader()
    count = loader.load_crypto(df)

    logger.info(f"Загружено {count} записей")
    return count


@task(name="refresh_views")
def refresh_views_task():
    logger = get_run_logger()
    logger.info("Обновление представлений")

    loader = DataLoader()
    loader.refresh_materialized_views()

    logger.info("Представления обновлены")


@flow(name="ETL Pipeline")
def etl_pipeline():
    """
    Основной ETL пайплайн:
    1. Extract - выгрузка из API
    2. Transform - очистка и агрегация
    3. Load - загрузка в PostgreSQL
    """
    logger = get_run_logger()
    logger.info("Запуск ETL пайплайна")
    start_time = datetime.now()

    logger.info("Шаг 1: Extract - получение данных из API")
    weather_df = extract_weather_task()
    crypto_df = extract_crypto_task()

    logger.info("Шаг 2: Transform - очистка и агрегация данных")
    weather_transformed = transform_weather_task(weather_df)
    crypto_transformed = transform_crypto_task(crypto_df)

    logger.info("Шаг 3: Load - сохранение в PostgreSQL")
    weather_count = load_weather_task(weather_transformed)
    crypto_count = load_crypto_task(crypto_transformed)

    if weather_count > 0 or crypto_count > 0:
        refresh_views_task()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info(f"ETL пайплайн завершен за {duration:.2f} секунд")
    logger.info(f"Итоги: погода={weather_count}, крипто={crypto_count}")

    return {
        "weather_loaded": weather_count,
        "crypto_loaded": crypto_count,
        "duration_seconds": duration
    }


if __name__ == "__main__":
    result = etl_pipeline()
    print(f"\n📈 Результат: {result}")