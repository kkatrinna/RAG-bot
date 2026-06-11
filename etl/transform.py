import polars as pl
from loguru import logger
from datetime import datetime


class DataTransformer:

    def __init__(self):
        self.required_weather_cols = ["city", "temperature", "humidity", "recorded_at"]
        self.required_crypto_cols = ["symbol", "price_usd", "recorded_at"]

    def transform_weather(self, df: pl.DataFrame) -> pl.DataFrame:

        logger.info("Трансформация данных о погоде")

        if df.is_empty():
            logger.warning("Нет данных для трансформации")
            return df

        initial_count = len(df)

        df = df.unique(subset=["city", "recorded_at"])

        df = df.with_columns([
            pl.col("temperature").fill_null(pl.col("temperature").mean()),
            pl.col("humidity").fill_null(60),
            pl.col("wind_speed").fill_null(5.0),
        ])

        df = df.with_columns([
            pl.col("temperature").cast(pl.Float64),
            pl.col("humidity").cast(pl.Int64),
            pl.col("pressure").cast(pl.Int64),
            pl.col("wind_speed").cast(pl.Float64),
        ])

        df = df.with_columns([
            pl.col("recorded_at").dt.hour().alias("hour"),
            pl.col("recorded_at").dt.date().alias("date"),
            (pl.col("temperature") * 9 / 5 + 32).alias("temperature_fahrenheit"),
        ])

        df = df.with_columns(
            pl.when(pl.col("temperature") < 0)
            .then(pl.lit("Холодно"))
            .when(pl.col("temperature") < 15)
            .then(pl.lit("Прохладно"))
            .when(pl.col("temperature") < 25)
            .then(pl.lit("Тепло"))
            .otherwise(pl.lit("Жарко"))
            .alias("temperature_category")
        )

        removed_count = initial_count - len(df)
        logger.info(f"Трансформировано {len(df)} записей (удалено дубликатов: {removed_count})")

        return df

    def transform_crypto(self, df: pl.DataFrame) -> pl.DataFrame:
        logger.info("Трансформация данных о криптовалютах")

        if df.is_empty():
            logger.warning("Нет данных для трансформации")
            return df

        initial_count = len(df)

        df = df.unique(subset=["symbol", "recorded_at"])

        df = df.with_columns([
            pl.col("price_usd").fill_null(0),
            pl.col("market_cap_usd").fill_null(0),
            pl.col("volume_24h").fill_null(0),
            pl.col("price_change_24h").fill_null(0),
        ])

        df = df.with_columns([
            pl.col("price_usd").cast(pl.Float64),
            pl.col("market_cap_usd").cast(pl.Float64),
            pl.col("volume_24h").cast(pl.Float64),
            pl.col("price_change_24h").cast(pl.Float64),
        ])

        df = df.with_columns([
            pl.col("recorded_at").dt.hour().alias("hour"),
            pl.col("recorded_at").dt.date().alias("date"),
        ])

        hourly_avg = df.group_by(["symbol", "hour"]).agg([
            pl.col("price_usd").mean().alias("avg_price_hourly"),
            pl.col("volume_24h").mean().alias("avg_volume_hourly"),
        ])

        logger.info(f"Трансформировано {len(df)} записей (удалено дубликатов: {initial_count - len(df)})")

        return df