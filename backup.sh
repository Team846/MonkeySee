#!/bin/bash
#   ./pull_configs.sh        pull from 10.8.46.204
#   ./pull_configs.sh 2      pull from funkyvision2

set -euo pipefail

TARGET_USER=orangepi
TARGET_HOST=10.8.46.204
TARGET_DIR=/home/orangepi/MonkeySee
SOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"

for arg in "$@"; do
  case "$arg" in
    1|2|3|4) TARGET_HOST="funkyvision$arg" ;;
  esac
done

DEST="$SOURCE_DIR/pi_configs/${TARGET_HOST}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SOURCE_DIR/pi_configs"

echo "Pulling configs from $TARGET_HOST ..."
rsync -avzm \
  --exclude 'venv/' --exclude '.git/' --exclude 'captures/' --exclude '__pycache__/' \
  --include '*/' \
  --include 'config.json' --include 'cal.json' --include 'config_dev.json' \
  --exclude '*' \
  "$TARGET_USER@$TARGET_HOST:$TARGET_DIR/" "$DEST/"

echo
echo "Saved to $DEST"
