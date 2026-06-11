import psycopg2
from psycopg2.extras import execute_values
import polars as pl
from loguru import logger
from datetime import datetime
import os


class DataLoader:

    def __init__(self):
        self.conn_params = {
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": os.getenv("POSTGRES_PORT", "5432"),
            "database": os.getenv("POSTGRES_DB", "etl_db"),
            "user": os.getenv("POSTGRES_USER", "etl_user"),
            "password": os.getenv("POSTGRES_PASSWORD", "etl_password")
        }

    def get_connection(self):
        return psycopg2.connect(**self.conn_params)

    def load_weather(self, df: pl.DataFrame) -> int:
        logger.info("Загрузка данных о погоде в PostgreSQL")

        if df.is_empty():
            logger.warning("Нет данных для загрузки")
            return 0

        conn = self.get_connection()
        cur = conn.cursor()

        try:
            cur.execute("SELECT MAX(recorded_at) FROM weather_metrics")
            last_record = cur.fetchone()[0]

            if last_record:
                df = df.filter(pl.col("recorded_at") > last_record)
                logger.info(f"  Загружается {len(df)} новых записей (после {last_record})")

            if df.is_empty():
                logger.info("  Нет новых данных для загрузки")
                return 0

            records = df.select([
                "city", "temperature", "feels_like", "humidity",
                "pressure", "wind_speed", "weather_description", "recorded_at"
            ]).to_numpy()

            insert_query = """
                INSERT INTO weather_metrics 
                (city, temperature, feels_like, humidity, pressure, wind_speed, weather_description, recorded_at)
                VALUES %s
                ON CONFLICT DO NOTHING
            """

            execute_values(cur, insert_query, records, page_size=100)
            conn.commit()

            logger.info(f"Загружено {len(records)} записей о погоде")
            return len(records)

        except Exception as e:
            logger.error(f"Ошибка при загрузке погоды: {e}")
            conn.rollback()
            return 0
        finally:
            cur.close()
            conn.close()

    def load_crypto(self, df: pl.DataFrame) -> int:
        logger.info("Загрузка данных о криптовалютах в PostgreSQL")

        if df.is_empty():
            logger.warning("Нет данных для загрузки")
            return 0

        conn = self.get_connection()
        cur = conn.cursor()

        try:
            records = df.select([
                "symbol", "name", "price_usd", "market_cap_usd",
                "volume_24h", "price_change_24h", "recorded_at"
            ]).to_numpy()

            insert_query = """
                INSERT INTO crypto_prices 
                (symbol, name, price_usd, market_cap_usd, volume_24h, price_change_24h, recorded_at)
                VALUES %s
            """

            execute_values(cur, insert_query, records, page_size=100)
            conn.commit()

            logger.info(f"Загружено {len(records)} записей о криптовалютах")
            return len(records)

        except Exception as e:
            logger.error(f"Ошибка при загрузке криптовалют: {e}")
            conn.rollback()
            return 0
        finally:
            cur.close()
            conn.close()

    def refresh_materialized_views(self):
        logger.info("Обновление материализованных представлений")

        conn = self.get_connection()
        cur = conn.cursor()

        try:
            cur.execute("SELECT refresh_etl_views()")
            conn.commit()
            logger.info("Материализованные представления обновлены")
        except Exception as e:
            logger.error(f"Ошибка при обновлении представлений: {e}")
        finally:
            cur.close()
            conn.close()