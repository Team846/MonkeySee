#!/bin/bash
#   ./deploy.sh            dry run
#   ./deploy.sh --go       actually deploy

set -euo pipefail

TARGET_USER=orangepi
TARGET_HOST=10.8.46.204
TARGET_DIR=/home/orangepi/MonkeySee
SOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"

GO=0
for arg in "$@"; do
  case "$arg" in
    --go) GO=1 ;;
    1|2|3|4) TARGET_HOST="funkyvision$arg" ;;
  esac
done

STAMP=$(date +%Y%m%d_%H%M%S)

EXCLUDES=(
  --exclude 'cal.json'
  --exclude 'config.json'
  --exclude 'venv/'
  --exclude 'captures/'
  --exclude '__pycache__/'
  --exclude '*.py[cod]'
  --exclude '.git/'
  --exclude '*.log'
  --exclude '.DS_Store'
  --exclude 'config_dev.json'
  --exclude 'deploy.sh'
  --exclude 'deploy.bat'
  --exclude 'actuallyonopi/'
)

RSYNC=(rsync -avz --backup "--backup-dir=${TARGET_DIR}_backup_${STAMP}"
       "${EXCLUDES[@]}" "$SOURCE_DIR/" "$TARGET_USER@$TARGET_HOST:$TARGET_DIR/")

if [ "$GO" -eq 0 ]; then
  echo "DRY RUN -> $TARGET_HOST  (nothing will be written; add --go to deploy)"
  echo
  "${RSYNC[@]}" --dry-run
  echo
  echo "Re-run with --go to apply."
else
  echo "Deploying to $TARGET_HOST ..."
  "${RSYNC[@]}"
  echo
  echo "Done. Overwritten files backed up to ${TARGET_DIR}_backup_${STAMP}/"
  echo "Restart:  ssh $TARGET_USER@$TARGET_HOST 'sudo systemctl restart monkeysee'"
fi
