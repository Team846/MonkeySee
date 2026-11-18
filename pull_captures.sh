#!/bin/bash
#   ./pull_captures.sh              pull captures/ and cal.json from 10.8.46.204
#   ./pull_captures.sh 2            from funkyvision2
#   ./pull_captures.sh 10.8.46.205  from that address

set -euo pipefail

TARGET_USER=orangepi
TARGET_HOST=10.8.46.204
TARGET_DIR=/home/orangepi/MonkeySee
SOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"

for arg in "$@"; do
  case "$arg" in
    1|2|3|4) TARGET_HOST="funkyvision$arg" ;;
    *.*) TARGET_HOST="$arg" ;;
  esac
done

echo "Pulling captures/ from $TARGET_HOST ..."
mkdir -p "$SOURCE_DIR/captures"
rsync -av --exclude '.DS_Store' "$TARGET_USER@$TARGET_HOST:$TARGET_DIR/captures/" "$SOURCE_DIR/captures/"

NEW=$(mktemp)
trap 'rm -f "$NEW"' EXIT
scp -q "$TARGET_USER@$TARGET_HOST:$TARGET_DIR/cal.json" "$NEW"
echo
python3 - "$SOURCE_DIR/cal.json" "$NEW" <<'PY'
import json, os, sys
old = json.load(open(sys.argv[1])) if os.path.exists(sys.argv[1]) else {}
new = json.load(open(sys.argv[2]))
def info(entry):
    m = entry.get("meta", {})
    k = m.get("camera_matrix")
    if not k:
        return "no calibration"
    return f"rpe {m.get('reprojection_error', float('nan')):.3f} px, fx {k[0][0]:.1f}, cx {k[0][2]:.1f}, cy {k[1][2]:.1f}"
for key in sorted(set(old) | set(new)):
    if key not in old:
        print(f"  {key:12} new        {info(new[key])}")
    elif key not in new:
        print(f"  {key:12} only on this Mac, dropped: {info(old[key])}")
    elif old[key] != new[key]:
        print(f"  {key:12} changed    {info(old[key])}  ->  {info(new[key])}")
    else:
        print(f"  {key:12} same       {info(new[key])}")
PY
cp "$NEW" "$SOURCE_DIR/cal.json"
cp "$NEW" "$SOURCE_DIR/pi_cal.json"
echo
echo "cal.json and pi_cal.json now match $TARGET_HOST (git diff cal.json to review)"
