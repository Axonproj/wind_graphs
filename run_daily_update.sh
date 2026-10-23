#!/usr/bin/env bash
# ============================================
# run_daily_update.sh
#  - Runs fetch_and_graph_bramble.pl
#  - Commits new/modified files
#  - Pushes to GitHub
#  - Sends Telegram notification
#  - Works on macOS and Raspberry Pi
# ============================================

set -euo pipefail

# --- Configuration ---
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
PYTHON="$VENV_DIR/bin/python"
LOGFILE="$PROJECT_DIR/cron_weather.log"
#
cd "$PROJECT_DIR" || { echo "❌ ERROR: Cannot cd to $PROJECT_DIR"; exit 1; }

echo "===========================================" > "$LOGFILE"
echo "Run started: $(date)" >> "$LOGFILE"
echo "Host: $(hostname)" >> "$LOGFILE"
echo "===========================================" >> "$LOGFILE"

# --- Ensure virtual environment exists ---
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating Python venv at $VENV_DIR" >> "$LOGFILE"
  # System site-packages: on the Pi Zero (armv6) pandas/matplotlib come from apt,
  # since pip has no wheels for it and building from source takes hours.
  python3 -m venv --system-site-packages "$VENV_DIR"
fi

# --- Activate virtual environment ---
source "$VENV_DIR/bin/activate"

# --- Install dependencies only if missing ---
if ! python -c "import pandas, matplotlib, numpy, astral" 2>/dev/null; then
  pip install -q --upgrade pip
  pip install -q pandas matplotlib numpy astral
fi

# No display on a headless Pi
export MPLBACKEND=Agg

# --- Bring the checkout up to date (the Pi's read-only overlay reverts it at each reboot) ---
echo "➡️ Pulling latest from GitHub..." >> "$LOGFILE"
git pull --rebase origin main >> "$LOGFILE" 2>&1

# Optionally load env file if present
[ -f "/etc/wind_graphs.env" ] && set -a && . /etc/wind_graphs.env && set +a

: "${TELEGRAM_BOT_TOKEN:?set TELEGRAM_BOT_TOKEN}"
: "${CHAT_ID:?set CHAT_ID}"

echo "-------------------------------------------" >> "$LOGFILE"
echo "Run started: $(date)" >> "$LOGFILE"

# --- Run pipeline ---
echo "➡️ Running fetch_and_graph_bramble.pl..." >> "$LOGFILE"
perl fetch_and_graph_bramble.pl >> "$LOGFILE" 2>&1



# --- Commit and push only data + graphs ---
echo "➡️ Committing and pushing updates (data + graphs only)..." >> "$LOGFILE"

# Stage only relevant folders if they exist
if [ -d "data" ]; then
  git add data >> "$LOGFILE" 2>&1
fi

if [ -d "graphs" ]; then
  git add graphs >> "$LOGFILE" 2>&1
fi

git add "index.html"

# Commit only if there are changes
if ! git diff --cached --quiet; then
  git commit -m "Automated update (data + graphs) on $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOGFILE" 2>&1
else
  echo "No data/graph changes to commit." >> "$LOGFILE"
fi


# --- Push changes (fail fast on conflict) ---
#
# No, the push is too likely to fail.
#
echo "➡️ Pushing changes to GitHub..." >> "$LOGFILE"

    if ! git push origin main >> "$LOGFILE" 2>&1; then
      echo "❌ Push failed — remote has diverged. Please resolve manually." >> "$LOGFILE"
      echo "Stopping script to avoid overwriting remote history." >> "$LOGFILE"
      # --- Telegram failure alert ---
      ALERT_MSG="🚨 Git push failed on $(hostname) at $(date '+%H:%M:%S')%0A%0A\
The remote has diverged a    nd requires manual resolution.%0A\
Run:%0A%60cd ~/projects/wind_   graphs && git pull origin main%60%0A\
Then resolve conflicts and push        manually."
      
      curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
           -d chat_id="${CHAT_ID}" \
           -d text="${ALERT_MSG}" >> "$LOGFILE" 2>&1
      
      exit 1
    fi


# --- Determine modified files in this commit ---
LAST_COMMIT_HASH=$(git rev-parse HEAD)
LAST_COMMIT_AUTHOR=$(git log -1 --pretty=%an)
LAST_COMMIT_MSG=$(git log -1 --pretty=%s)

# Check if the last commit was made by this script (matches our automated message)
if git log -1 --pretty=%s | grep -q "Automated update (data + graphs)"; then
    MODIFIED_FILES=$(git diff-tree --no-commit-id --name-only -r "$LAST_COMMIT_HASH" | grep -E '^(data|graphs)/' || true)
else
    MODIFIED_FILES=""
fi

if [ -z "$MODIFIED_FILES" ]; then
    MODIFIED_FILES="(no data/graph changes)"
fi

echo "➡️ Modified files:" >> "$LOGFILE"
echo "$MODIFIED_FILES" >> "$LOGFILE"


# --- Telegram summary + log tail combined into one message ---
LOG_TAIL=$(cat "$LOGFILE" | tail -c 2000)  # include last ~2KB to stay under Telegram 4096 char limit
ENCODED_LOG=$(printf "%s" "$LOG_TAIL" | perl -pe 's/%/%25/g; s/\n/%0A/g; s/\r//g')

SUMMARY_MSG="✅ Weather update completed on $(hostname) at $(date '+%H:%M:%S')%0A%0A\
Files changed:%0A$(printf '%s' "$MODIFIED_FILES" | perl -pe 's/\n/%0A/g')%0A%0A\
📝 Last part of log:%0A%0A$ENCODED_LOG"

curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
     -d chat_id="${CHAT_ID}" \
     -d text="${SUMMARY_MSG}" >> "$LOGFILE" 2>&1

echo "Run completed: $(date)" >> "$LOGFILE"
echo "-------------------------------------------" >> "$LOGFILE"

deactivate
exit 0
