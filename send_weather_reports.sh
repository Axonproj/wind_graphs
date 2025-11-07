#!/usr/bin/env bash
# ============================================
# send_weather_reports.sh
#  - Runs bramble_pipeline.pl
#  - Commits new/modified files
#  - Pushes to GitHub
#  - Sends Telegram notification
#  - Works on macOS and Raspberry Pi
# ============================================

set -euo pipefail

# --- Configuration ---
PROJECT_DIR="$HOME/projects/wind_graphs"
VENV_DIR="$PROJECT_DIR/venv"
PYTHON="$VENV_DIR/bin/python"
LOGFILE="$PROJECT_DIR/cron_weather.log"
TELEGRAM_BOT_TOKEN=8341214958:AAGjCWs9gcZAiyCh9nZBMvfJcfHw-W9R6PQ
CHAT_ID=8282574082
#
cd "$PROJECT_DIR" || { echo "❌ ERROR: Cannot cd to $PROJECT_DIR"; exit 1; }

# --- Ensure virtual environment exists ---
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating Python venv at $VENV_DIR" >> "$LOGFILE"
  python3 -m venv "$VENV_DIR"
fi

# --- Activate virtual environment ---
source "$VENV_DIR/bin/activate"

# --- Install dependencies if missing ---
pip install -q --upgrade pip
pip install -q pandas matplotlib numpy

echo "-------------------------------------------" >> "$LOGFILE"
echo "Run started: $(date)" >> "$LOGFILE"

# --- Run pipeline ---
echo "➡️ Running bramble_pipeline.pl..." >> "$LOGFILE"
perl bramble_pipeline.pl >> "$LOGFILE" 2>&1

# --- Commit and push changes ---
echo "➡️ Committing and pushing updates..." >> "$LOGFILE"
git add -A >> "$LOGFILE" 2>&1
git commit -m "Automated update on $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOGFILE" 2>&1 || echo "No changes to commit" >> "$LOGFILE"
git push origin main >> "$LOGFILE" 2>&1 || echo "Git push failed" >> "$LOGFILE"

# --- Determine modified files in this commit ---
MODIFIED_FILES=$(git diff-tree --no-commit-id --name-only -r HEAD)
if [ -z "$MODIFIED_FILES" ]; then
  MODIFIED_FILES="(no file changes)"
fi

echo "➡️ Modified files:" >> "$LOGFILE"
echo "$MODIFIED_FILES" >> "$LOGFILE"

# --- Telegram notification with file list ---
ENCODED_FILES=$(printf "%s" "$MODIFIED_FILES" | perl -pe 's/\n/%0A/g')
MESSAGE="✅ Weather update completed on $(hostname) at $(date '+%H:%M:%S')%0A%0AFiles changed:%0A$ENCODED_FILES"

curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
     -d chat_id="${CHAT_ID}" \
     -d text="${MESSAGE}" >> "$LOGFILE" 2>&1

echo "Run completed: $(date)" >> "$LOGFILE"
echo "-------------------------------------------" >> "$LOGFILE"

deactivate
exit 0
