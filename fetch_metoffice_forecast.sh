#!/usr/bin/env bash

# The key comes from the environment: run_daily_update.sh loads it from
# /etc/wind_graphs.env (METOFFICE_API_KEY=...), which is not in git.
: "${METOFFICE_API_KEY:?set METOFFICE_API_KEY in /etc/wind_graphs.env}"

if date -v+1d +"%Y%m%d" >/dev/null 2>&1; then
    # macOS / BSD date
    tomorrow=$(date -v+1d +"%Y%m%d")
else
    # Linux / GNU date
    tomorrow=$(date -d "tomorrow" +"%Y%m%d")
fi

perl ./metoffice_api_client.pl -k $METOFFICE_API_KEY -y 50.78 -x -1.28 -t hourly > data/${tomorrow}_bramble_forecast.json
