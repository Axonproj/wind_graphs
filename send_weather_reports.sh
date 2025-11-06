#!/bin/bash
# --- Bramble Pipeline + Git Auto-Commit + Telegram Notifier ---

# === Telegram setup ===
TOKEN="YOUR_BOT_TOKEN"
CHAT_ID="YOUR_CHAT_ID"

# === Paths ===
PROJECT="/home/ray/projects/wind_graphs"
PIPELINE="$PROJECT/bramble_pipeline.pl"
OUTFILE="/tmp/bramble_output.txt"
GIT_LOG="/tmp/git_push_log.txt"

# === Header ===
{
  echo "📅 $(date '+%Y-%m-%d %H:%M:%S')"
  echo "🏗️  Running Bramble pipeline..."
  echo "-----------------------------------------------"
} > "$OUTFILE"

# === Run the pipeline ===
cd "$PROJECT"
/usr/bin/perl "$PIPELINE" >> "$OUTFILE" 2>&1

echo "-----------------------------------------------" >> "$OUTFILE"
echo "🗂️  Running Git commit and push..." >> "$OUTFILE"

# === Git commit & push ===
git add -A >> "$GIT_LOG" 2>&1
git commit -m "Automated update from Raspberry Pi on $(date '+%Y-%m-%d %H:%M:%S')" >> "$GIT_LOG" 2>&1
git push origin main >> "$GIT_LOG" 2>&1

# Append Git log to report
{
  echo "-----------------------------------------------"
  echo "✅ Git Push Log:"
  cat "$GIT_LOG"
  echo "-----------------------------------------------"
  echo "✅ All tasks completed successfully."
} >> "$OUTFILE"

# === Send Telegram message ===
curl -s -X POST "https://api.telegram.org/bot${TOKEN}/sendMessage" \
  -d chat_id="$CHAT_ID" \
  --data-urlencode text="$(cat "$OUTFILE")" > /dev/null
