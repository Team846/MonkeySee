import json
import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
try:
    import pyudev
except ImportError:
    pyudev = None

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import platform
import subprocess
import plistlib

import cv2
import numpy as np
from sklearn.linear_model import LinearRegression

from preprocess import (
    PROCESS_FRAME,
    GET_TARGET_BRIGHTNESS,
    SET_TARGET_BRIGHTNESS,
    GET_DIVERGENCE_GAIN,
    SET_DIVERGENCE_GAIN,
    GET_MIN_CORR_STRENGTH,
    SET_MIN_CORR_STRENGTH,
)

from camera.camerareader import CameraReader

Z_MM = 155.5
TICK_SPACING_MM = 107.5 / 7.0
CAM_INDEX = 1
MIN_TICKS_TO_FIT = 7
SAVE_COEFFS_PATH = "cal.json"
DIVERGENCE_GAIN = 0.004

def merge_lines(frame, lines, axis="vertical", threshold=10):
    if not lines:
        return []

    h, w = frame.shape[:2]
    if axis == "vertical":
        coords = sorted([int((x1 + x2) / 2) for x1, y1, x2, y2 in lines])
    else:
        coords = sorted([int((y1 + y2) / 2) for x1, y1, x2, y2 in lines])

    merged, group = [], [coords[0]]
    for c in coords[1:]:
        if abs(c - group[-1]) <= threshold:
            group.append(c)
        else:
            avg_coord = int(np.mean(group))
            if axis == "vertical":
                merged.append((avg_coord, 0, avg_coord, h))
            else:
                merged.append((0, avg_coord, w, avg_coord))
            group = [c]

    avg_coord = int(np.mean(group))
    if axis == "vertical":
        merged.append((avg_coord, 0, avg_coord, h))
    else:
        merged.append((0, avg_coord, w, avg_coord))
    return merged


def fit_px_to_angle(x_px: np.ndarray, theta_rad: np.ndarray):
    x = x_px.reshape(-1, 1)
    X_poly = np.hstack([x, x**3, x**5])
    model = LinearRegression(fit_intercept=False).fit(X_poly, theta_rad)
    a, b, c = model.coef_
    return (float(a), float(b), float(c)), model


def px_to_angle_fn(a: float, b: float, c: float):
    def f(x_px: float) -> float:
        return a * x_px + b * (x_px**3) + c * (x_px**5)

    return f


def save_coeffs(path: str, a: float, b: float, c: float, meta: dict):
    data = {"a": a, "b": b, "c": c, "meta": meta}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_len=6, gap_len=6):
    x1, y1 = pt1
    x2, y2 = pt2
    length = int(np.hypot(x2 - x1, y2 - y1))
    for i in range(0, length, dash_len + gap_len):
        t1 = i / length
        t2 = min(i + dash_len, length) / length
        px1 = int(x1 + (x2 - x1) * t1)
        py1 = int(y1 + (y2 - y1) * t1)
        px2 = int(x1 + (x2 - x1) * t2)
        py2 = int(y1 + (y2 - y1) * t2)
        cv2.line(img, (px1, py1), (px2, py2), color, thickness)


@dataclass
class CalibrationResult:
    a: float
    b: float
    c: float
    model: LinearRegression
    used_ticks: int
    rmse_deg: float


def detect_vertical_ticks(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0, 20, minLineLength=20, maxLineGap=10
    )
    vertical_lines = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            if abs(x2 - x1) <= 3:
                vertical_lines.append((x1, y1, x2, y2))
    return merge_lines(frame, vertical_lines, axis="vertical", threshold=30)


def detect_horizontal_ticks(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0, 20, minLineLength=10, maxLineGap=10
    )
    horiz_lines = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            if abs(y2 - y1) <= 3:
                horiz_lines.append((x1, y1, x2, y2))
    return merge_lines(frame, horiz_lines, axis="horizontal", threshold=30)


def build_pairs(frame, ticks, Z_mm, tick_spacing_mm, axis="x"):
    h, w = frame.shape[:2]
    c = w // 2 if axis == "x" else h // 2
    coords = sorted({x1 if axis == "x" else y1 for (x1, y1, x2, y2) in ticks})
    if len(coords) < 2:
        return np.array([]), np.array([]), []
    center_idx = int(np.argmin([abs(v - c) for v in coords]))

    px, theta = [], []
    for i, v in enumerate(coords):
        offset_px = v - c
        mm = (i - center_idx) * tick_spacing_mm
        theta_true = math.atan2(mm, Z_mm)
        px.append(offset_px)
        theta.append(theta_true)
    return np.array(px, float), np.array(theta, float), coords


def calibrate_axis(x_px, theta_rad) -> Optional[CalibrationResult]:
    if len(x_px) < MIN_TICKS_TO_FIT:
        return None
    (a, b, c), model = fit_px_to_angle(x_px, theta_rad)
    X_poly = np.hstack(
        [x_px.reshape(-1, 1), x_px.reshape(-1, 1) ** 3, x_px.reshape(-1, 1) ** 5]
    )
    pred = model.predict(X_poly)
    rmse_rad = float(np.sqrt(np.mean((pred - theta_rad) ** 2)))
    rmse_deg = math.degrees(rmse_rad)
    return CalibrationResult(a, b, c, model, len(x_px), rmse_deg)


def calibrate_from_frame_xy(frame: np.ndarray, Z_mm: float, tick_spacing_mm: float):
    merged_v = detect_vertical_ticks(frame)
    x_px, theta_x, _ = build_pairs(frame, merged_v, Z_mm, tick_spacing_mm, axis="x")
    fit_x = calibrate_axis(x_px, theta_x)

    merged_h = detect_horizontal_ticks(frame)
    y_px, theta_y, _ = build_pairs(frame, merged_h, Z_mm, tick_spacing_mm, axis="y")
    fit_y = calibrate_axis(y_px, theta_y)

    return fit_x, fit_y, merged_v, merged_h


def save_full_calibration(path, ax, bx, cx_fit, ay, by, cy_fit, w, h, name):
    meta = {
        "resolution": 
        {
            "width": 800,
            "height": 600,
        },
    }
    data = {
        name[5:6]:{
            "x": {"a": ax, "b": bx, "c": cx_fit},
            "y": {"a": ay, "b": by, "c": cy_fit},
        },
        "meta": meta,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[SAVED] Calibration written to {path}")


def get_camera_serial(index: int) -> str:
    system = platform.system()
    
    if system == "Linux" and pyudev is not None:
        try:
            context = pyudev.Context()
            device_file = f"/dev/video{index}"
            device = pyudev.Devices.from_device_file(context, device_file)
            return device.get("ID_SERIAL_SHORT", "N/A - No Serial")
        except Exception as e:
            print(f"Error getting serial via pyudev: {e}")
            return "N/A - Error"
            
    elif system == "Darwin":
        try:
            output = subprocess.check_output(
                ['system_profiler', 'SPUSBDataType', '-xml'], 
                stderr=subprocess.DEVNULL
            )
            plist_data = plistlib.loads(output)
            
            def find_camera(items):
                for item in items:
                    if 'ATCam4' in item.get('_name', ''):
                        return item.get('serial_num', 'N/A')
                    elif 'ATCam3' in item.get('_name', ''):
                        return item.get('serial_num', 'N/A')
                    elif 'ATCam2' in item.get('_name', ''):
                        return item.get('serial_num', 'N/A')
                    elif 'ATCam1' in item.get('_name', ''):
                        return item.get('serial_num', 'N/A')
                    elif 'ATCam0' in item.get('_name', ''):
                        return item.get('serial_num', 'N/A')
                    elif 'ATCam5' in item.get('_name', ''):
                        return item.get('serial_num', 'N/A')
                    if '_items' in item:
                        res = find_camera(item['_items'])
                        if res:
                            return res
                return None
                
            return find_camera(plist_data) or "N/A - Not Found"
            
        except Exception as e:
            print(f"Error getting serial on macOS: {e}")
            return "N/A - Error"
            
    return "N/A"


def main():
    cam = CameraReader(CAM_INDEX)
    fit_history_x: List[Tuple[float, CalibrationResult]] = []
    fit_history_y: List[Tuple[float, CalibrationResult]] = []
    last_report_time = 0.0
    AVG_WINDOW = 10

    latest_coeffs = None

    print(f"Camera Serial: {get_camera_serial(CAM_INDEX)}")

    while True:
        frame = cam.get_frame()[0]
        if frame is None:
            continue

        processed_frame = PROCESS_FRAME(frame)

        disp = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
        h, w = disp.shape[:2]
        cx, cy = w // 2, h // 2

        cv2.circle(disp, (cx, cy), 5, (0, 0, 255), -1)
        cv2.line(disp, (0, cy), (w, cy), (0, 0, 255), 1)
        cv2.line(disp, (cx, 0), (cx, h), (0, 0, 255), 1)

        cur_bright = GET_TARGET_BRIGHTNESS()
        cur_div = GET_DIVERGENCE_GAIN()
        cur_min_corr = GET_MIN_CORR_STRENGTH()

        texts = [
            f"{cur_bright}",
            f"{cur_div:.2f}",
            f"{cur_min_corr:.2f}",
        ]
        for i, text in enumerate(texts):
            cv2.putText(
                disp,
                text,
                (10, 20 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        now = time.time()



        fit_x, fit_y, _, _ = calibrate_from_frame_xy(
            processed_frame, Z_MM, TICK_SPACING_MM
        )
        if fit_x:
            fit_history_x.append((now, fit_x))
        if fit_y:
            fit_history_y.append((now, fit_y))

        fit_history_x = [(t, f) for t, f in fit_history_x if now - t <= AVG_WINDOW]
        fit_history_y = [(t, f) for t, f in fit_history_y if now - t <= AVG_WINDOW]

        if now - last_report_time > AVG_WINDOW:
            last_report_time = now

            if fit_history_x and fit_history_y:
                # average coefficients
                ax = np.mean([f.a for _, f in fit_history_x])
                bx = np.mean([f.b for _, f in fit_history_x])
                cx_fit = np.mean([f.c for _, f in fit_history_x])
                ay = np.mean([f.a for _, f in fit_history_y])
                by = np.mean([f.b for _, f in fit_history_y])
                cy_fit = np.mean([f.c for _, f in fit_history_y])

                latest_coeffs = (ax, bx, cx_fit, ay, by, cy_fit)

                fx = px_to_angle_fn(ax, bx, cx_fit)
                fy = px_to_angle_fn(ay, by, cy_fit)

                theta_left = fx(-w // 2)
                theta_right = fx(w // 2)
                fov_x_deg = math.degrees(theta_right - theta_left)

                theta_top = fy(-h // 2)
                theta_bottom = fy(h // 2)
                fov_y_deg = math.degrees(theta_bottom - theta_top)

                print("[CALIBRATION UPDATED]")
                print(f"Estimated FOV: H={fov_x_deg:.2f}°, V={fov_y_deg:.2f}°")

        if fit_history_x:
            last_fit_x = fit_history_x[-1][1]
            for x1, _, _, _ in detect_vertical_ticks(processed_frame):
                draw_dashed_line(disp, (x1, 0), (x1, h), (0, 255, 0), thickness=1)
            cv2.putText(
                disp,
                f"X RMSE: {last_fit_x.rmse_deg:.3f} deg",
                (10, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (50, 220, 50),
                1,
                cv2.LINE_AA,
            )

        if fit_history_y:
            last_fit_y = fit_history_y[-1][1]
            for _, y1, _, _ in detect_horizontal_ticks(processed_frame):
                draw_dashed_line(disp, (0, y1), (w, y1), (0, 200, 200), thickness=1)
            cv2.putText(
                disp,
                f"Y RMSE: {last_fit_y.rmse_deg:.3f} deg",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 220, 50),
                1,
                cv2.LINE_AA,
            )

        cv2.imshow("Calibration View", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s") and latest_coeffs:
            ax, bx, cx_fit, ay, by, cy_fit = latest_coeffs
            save_full_calibration(
                SAVE_COEFFS_PATH, ax, bx, cx_fit, ay, by, cy_fit, w, h, get_camera_serial(CAM_INDEX)
            )
        elif key == ord("["):
            SET_TARGET_BRIGHTNESS(max(0, cur_bright - 5))
        elif key == ord("]"):
            SET_TARGET_BRIGHTNESS(min(255, cur_bright + 5))
        elif key == ord("-"):
            SET_DIVERGENCE_GAIN(max(0.0, cur_div - 0.1))
        elif key == ord("="):
            SET_DIVERGENCE_GAIN(cur_div + 0.1)
        elif key == ord(","):
            SET_MIN_CORR_STRENGTH(max(0.0, cur_min_corr - 0.05))
        elif key == ord("."):
            SET_MIN_CORR_STRENGTH(min(1.0, cur_min_corr + 0.05))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
