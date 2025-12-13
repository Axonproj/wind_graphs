#!/usr/bin/env python3
"""
plot_wind_comparison.py

Usage:
  python3 plot_wind_comparison.py yyyymmdd --no-show

Given a date string in yyyymmdd format, this script reads two files in ./data:

  1) {yyyymmdd}_bramble_forecast.json  -> forecast wind & gust for Bramble Bank
  2) {yyyymmdd}_bramble_actual.csv     -> actual wind speed & gust for that day

Plot features:
  - Wind:  solid line | Gusts: dashed line
  - Forecast: green lines
  - Actual: faint raw (30% opacity) + bold smoothed (2-hour rolling avg)
  - Wind direction arrows at hourly intervals (when data available)
  - Day/night shading for context
  - Beaufort scale bands and labels
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import signal



# --------------- Beaufort scale (in knots) ---------------
# (force, min_knots, max_knots)
BEAUFORT_BANDS = [
    (0, 0.0, 1.0),
    (1, 1.0, 3.0),
    (2, 4.0, 6.0),
    (3, 7.0, 10.0),
    (4, 11.0, 16.0),
    (5, 17.0, 21.0),
    (6, 22.0, 27.0),
    (7, 28.0, 33.0),
    (8, 34.0, 40.0),
    (9, 41.0, 47.0),
    (10, 48.0, 55.0),
    (11, 56.0, 63.0),
    (12, 64.0, 200.0),
]

def _beaufort_from_knots(kn):
    for force, lo, hi in BEAUFORT_BANDS:
        if lo <= kn <= hi:
            return force
    return BEAUFORT_BANDS[-1][0]
# --------------- configuration / aliases ---------------

# Time aliases to probe when auto-detecting
TIME_KEYS  = [
    "time","timestamp","date","datetime","TIME",
    "validTime","valid_time","validFrom","valid_from","timeISO","time_iso",
    "timeUtc","timeUTC","time_utc","issueTime","issue_time",
    "timeEpoch","time_epoch","epoch","unix","t","dt"
]
# Prefer 10 m products first
WIND_KEYS  = ["windSpeed10m","wind","wind_speed","speed","wspd","WSPD"]
GUST_KEYS  = ["windGustSpeed10m","gust","wind_gust","gst","GUST"]
DIR_KEYS   = ["windDirectionFrom10m","direction","wind_direction","dir","DIR","wd"]

LONDON_TZ = "Europe/London"



# --------------- interrupts ---------------

def handle_sigint(signum, frame):
    print("\nInterrupted by user (Ctrl-C).", file=sys.stderr)
    try:
        plt.close('all')
    except Exception:
        pass
    sys.exit(130)

signal.signal(signal.SIGINT, handle_sigint)
# --------------- helpers ---------------

def validate_date_str(s: str) -> None:
    """Ensure the argument is an 8-digit yyyymmdd string."""
    if len(s) != 8 or not s.isdigit():
        raise SystemExit("Error: date must be in yyyymmdd format, e.g., 20251101")


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Pick the first matching column, case-insensitive."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _parse_uk_datetime_strings(series: pd.Series) -> pd.Series:
    """Parse UK date/time strings with explicit formats, then fallback to dayfirst=True."""
    s = series.astype(str).str.strip()
    ts = pd.to_datetime(s, format="%d/%m/%Y %H:%M", errors="coerce")
    if ts.notna().any():
        return ts
    ts = pd.to_datetime(s, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    if ts.notna().any():
        return ts
    # Fallback – slower but robust
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def _coerce_time_series(df: pd.DataFrame, time_col: Optional[str]) -> Optional[pd.Series]:
    """
    Try to coerce timestamps from various shapes.
    Preference order:
      1) DATE + TIME columns (UK format)
      2) A single time-like column (try explicit UK formats, then dayfirst)
      3) Epoch seconds/milliseconds
    Naive times are localized to Europe/London; aware times are converted.
    """
    # 1) split DATE + TIME
    date_col = _pick_col(df, ["date","DATE"])
    time_part_col = _pick_col(df, ["time","TIME","hour","HOUR"])
    if date_col and time_part_col:
        dt_str = (df[date_col].astype(str) + " " + df[time_part_col].astype(str))
        ts = _parse_uk_datetime_strings(dt_str)
        # Localize to London if naive
        try:
            if ts.dt.tz is None:
                ts = ts.dt.tz_localize(LONDON_TZ, ambiguous="infer", nonexistent="shift_forward")
        except Exception:
            pass
        return ts

    # 2) direct single time column
    if time_col and time_col in df.columns:
        s = df[time_col].astype(str).str.strip()
        # Detect ISO-like timestamps (YYYY-MM-DDTHH:MM...); parse with utc=True (no dayfirst warning)
        is_iso_like = s.str.match(r'^\d{4}-\d{2}-\d{2}T').fillna(False)
        if is_iso_like.any():
            ts = pd.to_datetime(s, errors="coerce", utc=True)
            try:
                ts = ts.dt.tz_convert(LONDON_TZ)
            except Exception:
                pass
        else:
            # Try explicit UK formats first, then fallback
            ts = _parse_uk_datetime_strings(s)
            try:
                if ts.dt.tz is None:
                    ts = ts.dt.tz_localize(LONDON_TZ, ambiguous="infer", nonexistent="shift_forward")
            except Exception:
                pass
        return ts

    # 3) epoch seconds or milliseconds
    epoch_col = _pick_col(df, ["timeEpoch","time_epoch","epoch","unix","t","dt"])
    if epoch_col:
        s = pd.to_numeric(df[epoch_col], errors="coerce")
        if s.dropna().median() > 1e11:
            s = s / 1000.0  # likely ms
        ts = pd.to_datetime(s, unit="s", errors="coerce", utc=True)
        try:
            ts = ts.dt.tz_convert(LONDON_TZ)
        except Exception:
            pass
        return ts

    return None


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy dataframe with TIME (tz-aware), SPEED, GUST, and optionally DIRECTION columns."""
    tcol = _pick_col(df, TIME_KEYS)
    wcol = _pick_col(df, WIND_KEYS)
    gcol = _pick_col(df, GUST_KEYS)
    dcol = _pick_col(df, DIR_KEYS)

    tseries = _coerce_time_series(df, tcol)
    if tseries is None or tseries.isna().all():
        raise ValueError(f"Could not find/parse a timestamp column. Available columns: {list(df.columns)}")
    if wcol is None:
        raise ValueError(f"Could not find wind speed column. Looked for {WIND_KEYS}. Available columns: {list(df.columns)}")

    out = pd.DataFrame({
        "TIME": tseries,
        "SPEED": pd.to_numeric(df[wcol], errors="coerce"),
    })
    out["GUST"] = pd.to_numeric(df[gcol], errors="coerce") if gcol else np.nan
    out["DIRECTION"] = pd.to_numeric(df[dcol], errors="coerce") if dcol else np.nan
    out = out.dropna(subset=["TIME"])
    out = out.sort_values("TIME")
    return out


# --------------- forecast JSON loader ---------------

def read_forecast_json(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Case 1: GeoJSON FeatureCollection -> features[*].properties.timeSeries
    if isinstance(data, dict) and isinstance(data.get("features"), list):
        rows = []
        for feat in data["features"]:
            try:
                ts = feat.get("properties", {}).get("timeSeries", [])
                if isinstance(ts, list):
                    rows.extend(ts)
            except Exception:
                continue
        if rows:
            df = pd.DataFrame(rows)
            df = _normalize_df(df)
            # Forecasts are commonly UTC: convert/ensure London tz
            try:
                if df["TIME"].dt.tz is not None:
                    df["TIME"] = df["TIME"].dt.tz_convert(LONDON_TZ)
                else:
                    df["TIME"] = df["TIME"].dt.tz_localize(LONDON_TZ, ambiguous="infer", nonexistent="shift_forward")
            except Exception:
                pass
            return df

    # Case 2: dict with a list under common keys
    if isinstance(data, dict):
        for key in ["data","forecast","series","values","items","points","forecasts","list"]:
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

        # Case 2b: parallel arrays, e.g., {"times":[...], "windSpeed10m":[...]}
        if isinstance(data, dict):
            tkey = None
            for cand in ["times","time","timestamps","datetime","validTime","timeISO","timeUtc","timeUTC","time_epoch","epoch"]:
                if cand in data and isinstance(data[cand], list):
                    tkey = cand
                    break
            if tkey:
                length = len(data[tkey])
                rows = []
                for i in range(length):
                    row = {tkey: data[tkey][i]}
                    for k in ["windSpeed10m","windGustSpeed10m","wind","gust","wspd","gst"]:
                        if k in data and isinstance(data[k], list) and len(data[k]) == length:
                            row[k] = data[k][i]
                    rows.append(row)
                data = rows

    # Case 3: list or single dict
    if not isinstance(data, list):
        if isinstance(data, dict):
            data = [data]
        else:
            raise ValueError("Unrecognized JSON: expected FeatureCollection or list of records/dict with list(s).")

    df = pd.DataFrame(data)
    df = _normalize_df(df)
    # Convert to London tz if needed
    try:
        if df["TIME"].dt.tz is not None:
            df["TIME"] = df["TIME"].dt.tz_convert(LONDON_TZ)
        else:
            df["TIME"] = df["TIME"].dt.tz_localize(LONDON_TZ, ambiguous="infer", nonexistent="shift_forward")
    except Exception:
        pass
    return df


# --------------- actual CSV loader ---------------

def read_actual_csv(path: Path) -> pd.DataFrame:
    """
    Be lenient about delimiters:
      - Try sep=None (engine='python') to auto-detect commas vs. whitespace.
      - If that yields a single column (header like 'DATE TIME WSPD GST'), re-read with delim_whitespace=True.
    """
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, delim_whitespace=True)

    # If single squashed column, try whitespace parsing
    if df.shape[1] == 1:
        try:
            df2 = pd.read_csv(path, delim_whitespace=True)
            if df2.shape[1] > 1:
                df = df2
        except Exception:
            pass

    return _normalize_df(df)


# --------------- plotting ---------------

def _apply_time_axis_format(ax: plt.Axes, forecast: pd.DataFrame, actual: pd.DataFrame, start=None, end=None) -> None:
    """Choose sensible tick locators/formatters based on total span, and lock to 24h if start/end provided."""
    import matplotlib.ticker as mticker
    # If a 24h window is supplied, force ticks and labels
    if start is not None and end is not None:
        ticks = pd.date_range(start, end, freq="1h")
        ax.set_xlim(start, end)
        ax.set_xticks(ticks)
        end_num = mdates.date2num(pd.Timestamp(end).to_pydatetime())

        def fmt_func(x, pos=None):
            # Label the final tick as 24:00; others as HH:MM
            if abs(x - end_num) < 1e-9:
                return "24:00"
            return pd.to_datetime(mdates.num2date(x)).strftime("%H:%M")

        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_func))
        return

    # Fallback behaviour when no explicit window is provided
    times = []
    if forecast is not None and not forecast.empty:
        times.append(forecast["TIME"])
    if actual is not None and not actual.empty:
        times.append(actual["TIME"])
    if not times:
        return
    t = pd.concat(times).dropna().sort_values()
    if len(t) < 2:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        return

    span = (t.iloc[-1] - t.iloc[0]).to_pytimedelta()
    hours = span.total_seconds() / 3600.0

    if hours <= 12:
        locator  = mdates.HourLocator(interval=1)
        fmt      = mdates.DateFormatter("%H:%M")
    elif hours <= 48:
        locator  = mdates.HourLocator(interval=3)
        fmt      = mdates.DateFormatter("%H:%M")
    elif hours <= 7*24:
        locator  = mdates.DayLocator(interval=1)
        fmt      = mdates.DateFormatter("%d %b")
    else:
        locator  = mdates.DayLocator(interval=2)
        fmt      = mdates.DateFormatter("%d %b")

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt)
    pad = pd.Timedelta(minutes=30)
    ax.set_xlim(t.iloc[0] - pad, t.iloc[-1] + pad)


def plot_series(forecast: pd.DataFrame, actual: pd.DataFrame, title: str, start=None, end=None) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime, time

    # --- Smoothing function for rolling average ---
    def smooth_data(df, window_hours=2):
        """Apply rolling average to SPEED and GUST columns."""
        if df is None or df.empty:
            return df
        smoothed = df.copy()
        # Calculate window size based on time intervals
        if len(df) > 1:
            time_diff = (df["TIME"].iloc[1] - df["TIME"].iloc[0]).total_seconds() / 3600
            window_size = max(1, int(window_hours / time_diff))
        else:
            window_size = 1

        for col in ["SPEED", "GUST"]:
            if col in smoothed.columns:
                smoothed[f"{col}_SMOOTH"] = smoothed[col].rolling(
                    window=window_size, center=True, min_periods=1
                ).mean()
        return smoothed

    # --- Ensure gaps where values are 0 or near 0 ---
    def mask_zeros(df):
        if df is not None and not df.empty:
            for col in ["SPEED", "GUST"]:
                if col in df.columns:
                    df[col] = df[col].where(df[col].abs() > 0.1, np.nan)
            df.sort_values("TIME", inplace=True)
        return df

    forecast = mask_zeros(forecast)
    actual = mask_zeros(actual)

    # Apply smoothing to actual data
    actual = smooth_data(actual)
    # ---------------------------------------------------

    fig, ax = plt.subplots(figsize=(11, 6))

    # --- Day/night shading ---
    if start is not None and end is not None:
        # Approximate sunrise/sunset for UK in winter/summer
        # Simple heuristic: night is 22:00-06:00 winter, 21:00-05:00 summer
        current_time = start
        month = start.month
        if 4 <= month <= 9:  # Summer
            night_start_hour, night_end_hour = 21, 5
        else:  # Winter
            night_start_hour, night_end_hour = 22, 6

        while current_time < end:
            hour = current_time.hour
            next_hour = current_time + pd.Timedelta(hours=1)
            if hour >= night_start_hour or hour < night_end_hour:
                ax.axvspan(current_time, min(next_hour, end), alpha=0.05, color='gray', zorder=0)
            current_time = next_hour

    # Forecast: green (unchanged)
    if forecast is not None and not forecast.empty:
        ax.plot(forecast["TIME"], forecast["SPEED"], linestyle="-", label="Forecast wind", color="green", linewidth=1.5)
        if "GUST" in forecast.columns and forecast["GUST"].notna().any():
            ax.plot(forecast["TIME"], forecast["GUST"], linestyle="--", label="Forecast gust", color="green", linewidth=1.5)

    # Actual: faint raw data + bold smoothed curves
    if actual is not None and not actual.empty:
        # Faint raw data
        ax.plot(actual["TIME"], actual["SPEED"], linestyle="-", color="blue", alpha=0.3, linewidth=1, zorder=2)
        if "GUST" in actual.columns and actual["GUST"].notna().any():
            ax.plot(actual["TIME"], actual["GUST"], linestyle="--", color="blue", alpha=0.3, linewidth=1, zorder=2)

        # Bold smoothed curves
        if "SPEED_SMOOTH" in actual.columns:
            ax.plot(actual["TIME"], actual["SPEED_SMOOTH"], linestyle="-", label="Actual wind",
                   color="blue", linewidth=2.5, zorder=3)
        if "GUST_SMOOTH" in actual.columns and actual["GUST_SMOOTH"].notna().any():
            ax.plot(actual["TIME"], actual["GUST_SMOOTH"], linestyle="--", label="Actual gust",
                   color="blue", linewidth=2.5, zorder=3)

    # --- Wind direction arrows (if direction data available) ---
    # To enable: add 'windDirectionFrom10m' to get_forecast.pl data capture
    if forecast is not None and not forecast.empty:
        # Check for direction column (various possible names)
        dir_col = _pick_col(forecast, ["DIRECTION", "DIR", "windDirectionFrom10m", "direction", "wind_dir"])
        if dir_col and dir_col in forecast.columns:
            # Plot arrows at hourly intervals
            direction_data = forecast[forecast[dir_col].notna()].copy()
            # Sample every hour to avoid clutter
            direction_data = direction_data[direction_data["TIME"].dt.minute == 0]

            ymin, ymax = ax.get_ylim()
            arrow_y = ymax * 0.92  # Position arrows near top of plot

            for _, row in direction_data.iterrows():
                # Convert meteorological direction (from) to mathematical angle
                # Met direction: 0° = North (from), 90° = East (from)
                # We want arrow pointing TO where wind is going
                met_dir = row[dir_col]
                # Reverse direction (wind from N goes to S)
                arrow_angle = (met_dir + 180) % 360
                # Convert to radians and adjust for matplotlib (0° = East, CCW)
                math_angle = np.radians(90 - arrow_angle)

                # Arrow components - scale based on plot dimensions
                arrow_length = (ymax - ymin) * 0.04
                dx = np.cos(math_angle) * arrow_length
                dy = np.sin(math_angle) * arrow_length

                # Use annotate instead of arrow for better control with datetime axes
                ax.annotate('', xy=(mdates.date2num(row["TIME"]), arrow_y + dy),
                           xytext=(mdates.date2num(row["TIME"]), arrow_y),
                           arrowprops=dict(arrowstyle='->', color='darkgreen',
                                         lw=1.5, alpha=0.7),
                           zorder=10)

    # Format X axis (24h window and HH:MM labels)
    _apply_time_axis_format(ax, forecast, actual, start=start, end=end)

    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")

    ax.set_ylabel("Knots")
    ax.set_title((title or "Wind") + " Speed")

    ymin, ymax = ax.get_ylim()
    for i, (force, lo, hi) in enumerate(BEAUFORT_BANDS):
        if hi < ymin or lo > ymax:
            continue
        ax.axhspan(max(lo, ymin), min(hi, ymax), alpha=0.08 if force % 2 == 0 else 0.12)

    ax2 = ax.twinx()
    ticks, labels = [], []
    for (force, lo, hi) in BEAUFORT_BANDS:
        mid = (lo + hi) / 2.0
        if ymin <= mid <= ymax:
            ticks.append(mid)
            labels.append(str(force))
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(ticks)
    ax2.set_yticklabels(labels)
    ax2.tick_params(axis="y", length=0, labelright=True)
    ax2.set_ylabel("Beaufort")

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")
    fig.tight_layout()

    out_png = f"./graphs/{title.replace(' ', '_').lower()}.png" if title else "wind_speed.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_png}")

def main(argv: List[str]) -> int:
    if len(argv) < 2 or len(argv) > 3:
        print("Usage: python3 plot_wind_comparison.py yyyymmdd [--no-show]", file=sys.stderr)
        return 2

    date_str = argv[1]
    no_show = "--no-show" in argv
    validate_date_str(date_str)

    # Build 24h window in Europe/London for the specified date
    try:
        start_naive = pd.to_datetime(date_str, format="%Y%m%d")
    except Exception:
        print("Invalid date format; expected yyyymmdd.", file=sys.stderr)
        return 2
    date_window_start = start_naive.tz_localize(LONDON_TZ)
    date_window_end = date_window_start + pd.Timedelta(days=1)

    forecast_path = Path(f"data/{date_str}_bramble_forecast.json")
    actual_path   = Path(f"data/{date_str}_bramble_actual.csv")

    if not forecast_path.exists():
        print(f"Error: missing forecast file: {forecast_path}", file=sys.stderr)
        return 1
    if not actual_path.exists():
        print(f"Error: missing actual file:   {actual_path}", file=sys.stderr)
        return 1

    try:
        forecast_df = read_forecast_json(forecast_path)
    except Exception as e:
        print(f"Failed to read forecast JSON: {e}", file=sys.stderr)
        return 1

    try:
        actual_df = read_actual_csv(actual_path)
    except Exception as e:
        print(f"Failed to read actual CSV: {e}", file=sys.stderr)
        return 1

    # Restrict to the exact 24h London window
    def _clip(df):
        if df is None or df.empty:
            return df
        m = df["TIME"].between(date_window_start, date_window_end, inclusive="left")
        return df.loc[m].copy()

    forecast_df = _clip(forecast_df)
    actual_df = _clip(actual_df)

    title = f"{date_str} Bramble Bank wind"
    plot_series(forecast_df, actual_df, title, start=date_window_start, end=date_window_end)

    # --- Display interactively if no_show is False ---
    if not no_show:
        try:
            plt.show()
        except KeyboardInterrupt:
            print("Interrupted while showing plot.", file=sys.stderr)
            plt.close("all")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
