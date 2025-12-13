# Bramble Bank Wind Forecast vs Actual

Automated comparison of Met Office wind forecasts against actual measurements from the Bramble Bank weather station in the Solent.

## Main Scripts

- **run_daily_update.sh** - Main cron entry point (runs daily at 14:00)
- **fetch_and_graph_bramble.pl** - Orchestrates data fetching and graph generation
- **plot_wind_comparison.py** - Generates comparison graphs with enhanced visualizations
- **fetch_metoffice_forecast.sh** - Fetches Met Office forecast data
- **metoffice_api_client.pl** - Met Office API client library

## Cron Job Setup

Add to crontab on Raspberry Pi:
```bash
0 14 * * * /home/ray/projects/wind_graphs/run_daily_update.sh >> /tmp/run_daily_update_cron.log 2>&1
```

## Graph Features

- Faint raw actual data (30% opacity) with bold smoothed curves (2-hour rolling avg)
- Green forecast lines for comparison
- Wind direction arrows at hourly intervals
- Day/night shading for context
- Beaufort scale bands and labels
