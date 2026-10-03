import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import localization.undistort as undistort
from calibration.checkerboard_calibrator import CheckerboardCalibrator

parser = argparse.ArgumentParser(description="Calibrate a camera from sample images saved by the dashboard")
parser.add_argument("folder", help="captures/calibration/cam<id>_<w>x<h>_<time> folder")
parser.add_argument("--cal", default="cal.json", help="cal.json to add the calibration to")
args = parser.parse_args()

match = re.match(r"cam(\d+)_", os.path.basename(os.path.normpath(args.folder)))
if not match:
    sys.exit(f"Folder name should start with cam<id>_ : {args.folder}")
camera_id = int(match.group(1))

undistort.CAL_PATH = args.cal
calibrator = CheckerboardCalibrator(camera_id)
paths = sorted(glob.glob(os.path.join(args.folder, "*.png")))
for path in paths:
    frame = cv2.imread(path)
    found, corners, _ = calibrator.find_corners(frame)
    if not found or corners is None or len(corners) < calibrator.min_corners_capture:
        print(f"skipped {os.path.basename(path)}: checkerboard not found")
        continue
    calibrator.all_corners.append(corners)
    calibrator.image_size = (frame.shape[1], frame.shape[0])

print(f"Using {len(calibrator.all_corners)} of {len(paths)} images")
result = calibrator.calibrate()
if result is None:
    sys.exit(calibrator.status_message)

meta = result["meta"]
k = meta["camera_matrix"]
print(f"Reprojection error {meta['reprojection_error']:.3f} px, fx {k[0][0]:.1f} fy {k[1][1]:.1f} cx {k[0][2]:.1f} cy {k[1][2]:.1f}")
if not calibrator.save_calibration():
    sys.exit("Failed to save calibration")
print(f"Saved camera {camera_id} {meta['resolution']['width']}x{meta['resolution']['height']} calibration to {args.cal}")
