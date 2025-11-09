#/bin/bash

export METOFFICE_API_KEY="REDACTED_METOFFICE_KEY"

if date -v+1d +"%Y%m%d" >/dev/null 2>&1; then
    # macOS / BSD date
    tomorrow=$(date -v+1d +"%Y%m%d")
else
    # Linux / GNU date
    tomorrow=$(date -d "tomorrow" +"%Y%m%d")
fi

perl ./get_forecast.pl -k $METOFFICE_API_KEY -y 50.78 -x -1.28 -t hourly > data/${tomorrow}_bramble_forecast.json
