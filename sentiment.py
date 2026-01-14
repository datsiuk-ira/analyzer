import requests
from logger import logger
import time
from typing import Dict, Any

class SentimentAnalyzer:
    """
    Fetches and caches market sentiment from Fear & Greed Index.
    """
    def __init__(self, cache_ttl: int = 43200): # 12 hours
        self.api_url = "https://api.alternative.me/fng/"
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Any] = {}
        self._last_update = 0

    def get_fng_index(self) -> int:
        """
        Returns the Fear & Greed Index (0-100).
        Defaults to 50 (Neutral) on failure.
        """
        now = time.time()
        if now - self._last_update < self.cache_ttl and self._cache:
            return self._cache.get('value', 50)

        try:
            response = requests.get(self.api_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                value = int(data['data'][0]['value'])
                self._cache = {
                    'value': value,
                    'classification': data['data'][0]['value_classification']
                }
                self._last_update = now
                logger.info(f"Sentiment Updated: {value} ({self._cache['classification']})")
                return value
            else:
                logger.warning(f"Sentiment API returned status {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to fetch sentiment: {e}")

        return self._cache.get('value', 50) # Return cached or neutral default
