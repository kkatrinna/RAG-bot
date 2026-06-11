import requests
import polars as pl
from loguru import logger
from datetime import datetime
from typing import Dict, List, Optional
import time
import os


class DataExtractor:

    def __init__(self):
        self.weather_api_key = os.getenv("WEATHER_API_KEY", "")
        self.rate_limit_delay = 0.5
        self.cities = ["Moscow", "London", "New York", "Tokyo", "Berlin", "Paris", "Rome", "Madrid"]

    def extract_weather(self) -> pl.DataFrame:
        logger.info("Извлечение данных о погоде")

        if not self.weather_api_key:
            logger.warning("API ключ для погоды не найден, использую тестовые данные")
            return self.extract_weather_mock()

        data = []
        for city in self.cities:
            try:
                url = f"http://api.openweathermap.org/data/2.5/weather"
                params = {
                    "q": city,
                    "appid": self.weather_api_key,
                    "units": "metric",
                    "lang": "ru"
                }

                response = requests.get(url, params=params, timeout=10)

                if response.status_code == 200:
                    result = response.json()
                    data.append({
                        "city": city,
                        "temperature": result["main"]["temp"],
                        "feels_like": result["main"]["feels_like"],
                        "humidity": result["main"]["humidity"],
                        "pressure": result["main"]["pressure"],
                        "wind_speed": result["wind"]["speed"],
                        "weather_description": result["weather"][0]["description"],
                        "recorded_at": datetime.now()
                    })
                    logger.debug(f"  ✓ {city}: {result['main']['temp']}°C")
                else:
                    logger.error(f"  ✗ Ошибка для {city}: {response.status_code}")

                time.sleep(self.rate_limit_delay)

            except Exception as e:
                logger.error(f"  ✗ Ошибка для {city}: {e}")

        if not data:
            logger.warning("Не удалось получить реальные данные, использую тестовые")
            return self.extract_weather_mock()

        df = pl.DataFrame(data)
        logger.info(f"Извлечено {len(df)} записей о погоде")
        return df

    def extract_weather_mock(self) -> pl.DataFrame:
        logger.info("  Использую тестовые данные для погоды")

        import random
        data = []

        for city in self.cities:
            data.append({
                "city": city,
                "temperature": round(15 + random.randint(-10, 10), 1),
                "feels_like": round(13 + random.randint(-10, 10), 1),
                "humidity": random.randint(40, 90),
                "pressure": random.randint(990, 1030),
                "wind_speed": round(random.uniform(0, 15), 1),
                "weather_description": random.choice(["Солнечно", "Облачно", "Дождь", "Ясно"]),
                "recorded_at": datetime.now()
            })

        return pl.DataFrame(data)

    def extract_crypto(self) -> pl.DataFrame:
        logger.info("Извлечение данных о криптовалютах")

        url = "https://api.coingecko.com/api/v3/simple/price"
        crypto_ids = ["bitcoin", "ethereum", "cardano", "dogecoin", "ripple", "solana", "polkadot"]
        crypto_names = {
            "bitcoin": "Bitcoin", "ethereum": "Ethereum", "cardano": "Cardano",
            "dogecoin": "Dogecoin", "ripple": "Ripple", "solana": "Solana", "polkadot": "Polkadot"
        }

        params = {
            "ids": ",".join(crypto_ids),
            "vs_currencies": "usd",
            "include_market_cap": "true",
            "include_24hr_change": "true",
            "include_24hr_vol": "true"
        }

        try:
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                result = response.json()
                data = []

                for crypto_id in crypto_ids:
                    if crypto_id in result:
                        data.append({
                            "symbol": crypto_id[:3] if crypto_id == "dogecoin" else crypto_id[:4],
                            "name": crypto_names.get(crypto_id, crypto_id),
                            "price_usd": result[crypto_id].get("usd", 0),
                            "market_cap_usd": result[crypto_id].get("usd_market_cap", 0),
                            "volume_24h": result[crypto_id].get("usd_24h_vol", 0),
                            "price_change_24h": result[crypto_id].get("usd_24h_change", 0),
                            "recorded_at": datetime.now()
                        })
                        logger.debug(f"  ✓ {crypto_names[crypto_id]}: ${result[crypto_id]['usd']:,.0f}")

                df = pl.DataFrame(data)
                logger.info(f"Извлечено {len(df)} записей о криптовалютах")
                return df
            else:
                logger.error(f"Ошибка API CoinGecko: {response.status_code}")

        except Exception as e:
            logger.error(f"Ошибка при выгрузке криптовалют: {e}")

        return pl.DataFrame()

    def extract_github_stats(self, repo_name: str = "langchain-ai/langchain") -> pl.DataFrame:
        logger.info(f"Извлечение статистики GitHub для {repo_name}")

        url = f"https://api.github.com/repos/{repo_name}"

        try:
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                result = response.json()
                data = [{
                    "repo_name": repo_name,
                    "stars": result.get("stargazers_count", 0),
                    "forks": result.get("forks_count", 0),
                    "open_issues": result.get("open_issues_count", 0),
                    "recorded_at": datetime.now()
                }]

                df = pl.DataFrame(data)
                logger.info(f"Извлечена статистика: {data[0]['stars']}, {data[0]['forks']}")
                return df
            else:
                logger.error(f"Ошибка GitHub API: {response.status_code}")

        except Exception as e:
            logger.error(f"Ошибка при выгрузке статистики GitHub: {e}")

        return pl.DataFrame()