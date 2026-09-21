---
name: Time and Weather Skill
description: Get the current time, timezone, and real-time weather conditions for any city worldwide using the free Open-Meteo public service without requiring API keys.
Trigger Queries:
  - What is the weather in Tokyo?
  - What time is it in London?
  - Check current temperature and forecast for New York
  - How is the weather in Paris right now?
  - Tell me the local time and weather conditions in Sydney
---

# Time and Weather Skill

## Overview
This skill provides real-time atmospheric metrics (temperature in Celsius and Fahrenheit, weather condition code, windspeed) and accurate local wall-clock time for any global city. It utilizes the Open-Meteo public API which does not require API keys or credentials.

## Standard Operating Procedure (SOP)
1. **Identify Target City**: Extract the target city or metropolitan area from the user's inquiry.
2. **Execute Geocoding & Weather Retrieval**:
   - Run `skills/time-weather-skill/scripts/env_tools.py` passing the city name argument.
   - Or import `get_weather_and_time` directly in the runtime executor.
3. **Format Response**: Present the local time, temperature (°C and °F), sky condition, and wind speed clearly to the user.
