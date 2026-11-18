#!/bin/bash
#   ./install_napa.sh download   fetch the Pi wheel into wheels/
#   ./install_napa.sh            install on 10.8.46.204, turn NAPA on for every apriltag camera, restart
#   ./install_napa.sh 2          funkyviz2

set -euo pipefail

TARGET_USER=orangepi
TARGET_HOST=10.8.46.204
TARGET_DIR=/home/orangepi/MonkeySee
VERSION=0.5.1
MIN_TAG_PX=16
WHEELS="$(cd "$(dirname "$0")" && pwd)/wheels"

for arg in "$@"; do
  case "$arg" in
    1|2|3|4) TARGET_HOST="funkyvision$arg" ;;
  esac
done

if ! ls "$WHEELS"/napa_vision-$VERSION-*.whl >/dev/null 2>&1; then
  echo "Downloading napa-vision $VERSION for the Pi (aarch64, Python 3.11) ..."
  python3 -m pip download "napa-vision==$VERSION" --no-deps --only-binary=:all: \
    --platform manylinux2014_aarch64 --python-version 3.11 --implementation cp -d "$WHEELS"
fi
[ "${1:-}" = download ] && { echo "Wheel ready in $WHEELS"; exit 0; }

WHEEL=$(basename "$(ls "$WHEELS"/napa_vision-$VERSION-*.whl | head -1)")
STAMP=$(date +%Y%m%d_%H%M%S)
REMOTE=$(mktemp)
trap 'rm -f "$REMOTE"' EXIT
cat > "$REMOTE" <<EOF
set -euo pipefail
cd $TARGET_DIR
grep -q MAKE_APRILTAG_DETECTOR localization/detection.py || { echo "This Pi has no NAPA code yet, run ./deploy.sh --go first"; exit 1; }
venv/bin/python -m pip install --no-index /tmp/$WHEEL
venv/bin/python -c "import numpy as np; from napa_vision import NapaDetector; NapaDetector(min_tag_px=$MIN_TAG_PX, max_hamming=2, min_contrast=12, sharpening=0.5).detect(np.zeros((600, 800), np.uint8)); print('napa_vision works')"
cp config.json config.json.before_napa_$STAMP
venv/bin/python - <<'PY'
import json
d = json.load(open("config.json"))
for cam in d["cameras"]:
    if cam["pipeline"] == "apriltag":
        section = cam.get("apriltag_config", f"AprilTag{cam['id']}")
        d.setdefault(section, {})["napa_min_tag_px"] = {"value": "$MIN_TAG_PX", "type": 1}
        print(f"camera {cam['id']}: {section}.napa_min_tag_px = $MIN_TAG_PX")
with open("config.json", "w") as f:
    json.dump(d, f, indent=4)
    f.write("\n")
PY
if systemctl list-unit-files monkeysee.service 2>/dev/null | grep -q monkeysee; then
  sudo systemctl restart monkeysee
else
  pkill -f "python main.py" || true
fi
echo "Restarted. Old config saved as config.json.before_napa_$STAMP"
EOF

echo "Installing $WHEEL on $TARGET_HOST ..."
scp "$WHEELS/$WHEEL" "$REMOTE" "$TARGET_USER@$TARGET_HOST:/tmp/"
ssh -t "$TARGET_USER@$TARGET_HOST" "bash /tmp/$(basename "$REMOTE") && rm -f /tmp/$(basename "$REMOTE")"
echo
echo "Check the log for 'NAPA detector, min_tag_px $MIN_TAG_PX' on each camera."
