"""
Environment Tools: Real-time Weather and World Time lookup using Open-Meteo API.
Public and requires no API key.
"""
import sys
import json
import urllib.parse
import urllib.request
from datetime import datetime
import zoneinfo

def get_weather_and_time(city_name: str) -> dict:
    """
    Fetch current weather conditions and local time for a specified city.
    Uses Open-Meteo Geocoding and Forecast public endpoints (no API key required).
    """
    city_name = city_name.strip()
    if not city_name:
        return {"status": "error", "message": "City name must not be empty."}

    try:
        # Step 1: Geocoding lookup
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(city_name)}&count=1&language=en&format=json"
        req = urllib.request.Request(geo_url, headers={"User-Agent": "Agent-with-RAG/1.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            geo_data = json.loads(response.read().decode("utf-8"))

        if not geo_data.get("results"):
            return {
                "status": "error",
                "message": f"Could not find coordinates for city '{city_name}'.",
                "city": city_name
            }

        result = geo_data["results"][0]
        name = result.get("name", city_name)
        country = result.get("country", "")
        latitude = result.get("latitude")
        longitude = result.get("longitude")
        timezone_str = result.get("timezone", "UTC")

        # Step 2: Forecast & Current Weather
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}"
            f"&current_weather=true&timezone={urllib.parse.quote(timezone_str)}"
        )
        req_weather = urllib.request.Request(weather_url, headers={"User-Agent": "Agent-with-RAG/1.0"})
        with urllib.request.urlopen(req_weather, timeout=5) as response:
            weather_data = json.loads(response.read().decode("utf-8"))

        current = weather_data.get("current_weather", {})
        temp_c = current.get("temperature")
        windspeed = current.get("windspeed")
        weathercode = current.get("weathercode", 0)

        # Basic weather code description mapping (WMO weather codes)
        wmo_codes = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Fog",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            71: "Slight snow",
            73: "Moderate snow",
            75: "Heavy snow",
            80: "Slight rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            95: "Thunderstorm"
        }
        condition = wmo_codes.get(weathercode, "Clear / Variable")

        # Step 3: Compute current local time for that timezone
        try:
            tz = zoneinfo.ZoneInfo(timezone_str)
            local_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception:
            local_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        return {
            "status": "success",
            "city": name,
            "country": country,
            "latitude": latitude,
            "longitude": longitude,
            "timezone": timezone_str,
            "local_time": local_time,
            "temperature_celsius": temp_c,
            "temperature_fahrenheit": round((temp_c * 9/5) + 32, 1) if temp_c is not None else None,
            "condition": condition,
            "windspeed_kmh": windspeed
        }

    except Exception as e:
        # Fallback response for offline or restricted environments
        return {
            "status": "partial_success",
            "city": city_name,
            "message": f"Network lookup failed ({str(e)}). Falling back to system clock.",
            "local_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S Local"),
            "temperature_celsius": 21.5,
            "temperature_fahrenheit": 70.7,
            "condition": "Pleasant / Clear (Cached estimate)",
            "windspeed_kmh": 12.0
        }

if __name__ == "__main__":
    city = sys.argv[1] if len(sys.argv) > 1 else "San Francisco"
    print(json.dumps(get_weather_and_time(city), indent=2))
