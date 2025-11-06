#!/usr/bin/bash
# --- Bramble Pipeline + Git Auto-Commit + Telegram Notifier ---

TOKEN="REDACTED_TELEGRAM_TOKEN"
CHAT_ID="CHAT_ID"

PROJECT="/home/ray/projects/wind_graphs"
PIPELINE="$PROJECT/bramble_pipeline.pl"
OUTFILE="/tmp/bramble_full_output.txt"

cd "$PROJECT"

# Start log fresh
{
  echo "📅 $(date '+%Y-%m-%d %H:%M:%S')"
  echo "🏗️  Running Bramble pipeline..."
  echo "-----------------------------------------------"
} > "$OUTFILE"

# Run the pipeline and capture *all* output
/usr/bin/perl "$PIPELINE" >> "$OUTFILE" 2>&1
echo "-----------------------------------------------" >> "$OUTFILE"
echo "🗂️  Running Git commit and push..." >> "$OUTFILE"

# Commit and push any changes
if ! git diff-index --quiet HEAD --; then
  git add -A >> "$OUTFILE" 2>&1
  git commit -m "Automated update from Raspberry Pi on $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUTFILE" 2>&1
  git push origin main >> "$OUTFILE" 2>&1
  echo "✅ Changes committed and pushed to GitHub." >> "$OUTFILE"
else
  echo "ℹ️  No changes detected. Nothing to commit." >> "$OUTFILE"
fi

echo "-----------------------------------------------" >> "$OUTFILE"
echo "✅ All tasks completed successfully." >> "$OUTFILE"

# Send the *entire* output file to Telegram
curl -s -X POST "https://api.telegram.org/bot${TOKEN}/sendMessage" \
  -d chat_id="$CHAT_ID" \
  --data-urlencode text="$(cat "$OUTFILE")" > /dev/null
