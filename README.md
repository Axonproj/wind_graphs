# Bramble Bank Wind Forecast vs Actual

Automated comparison of Met Office, AROME and ECMWF wind forecasts against actual measurements from the Bramble Bank weather station in the Solent.

## Main Scripts

- **run_daily_update.sh** - Main cron entry point (runs daily at 14:00)
- **fetch_and_graph_bramble.pl** - Orchestrates data fetching and graph generation
- **plot_wind_comparison.py** - Generates comparison graphs with enhanced visualizations
- **fetch_metoffice_forecast.sh** - Fetches Met Office forecast data
- **metoffice_api_client.pl** - Met Office API client library
- **fetch_openmeteo_forecast.pl** - Fetches AROME HD (1.5 km) and ECMWF IFS (9 km) forecasts from Open-Meteo

## Cron Job Setup

Add to crontab on Raspberry Pi:
```bash
0 14 * * * /home/pi/projects/wind_graphs/run_daily_update.sh
```

## Graph Features

- Faint raw actual data (30% opacity) with bold smoothed curves (2-hour rolling avg)
- Forecast lines in knots: Met Office green, AROME orange, ECMWF purple, each with its worst hour in the legend (how far out, too light or too strong, and when)
- Each forecast's error is also kept in data/forecast_scores.csv (worst-hour wind error and time, wind MAE and bias, gust MAE, knots; negative = forecast too light)
- Wind direction arrows at hourly intervals
- Day/night shading for context
- Beaufort scale bands and labels
