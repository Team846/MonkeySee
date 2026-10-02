from typing import Tuple, Dict, Optional
import cv2
import numpy as np
import math
import json
from util.logger import Logger

logger = Logger("Calibration")

CAL_PATH = "cal.json"

calibrations: Dict[int, dict] = {}
_calibrations_loaded = False

_camera_matrices: Dict[Tuple[int, int, int], np.ndarray] = {}
_dist_coeffs: Dict[Tuple[int, int, int], np.ndarray] = {}
_reported_missing = set()

def calibration_key(camera_id: int, width: int, height: int) -> str:
    return f"{camera_id}@{width}x{height}"

def load_all_calibrations(force: bool = False):
    global calibrations, _calibrations_loaded, _camera_matrices, _dist_coeffs
    
    if _calibrations_loaded and not force:
        return
    
    try:
        with open(CAL_PATH, "r") as f:
            all_cals = json.load(f)
        
        cals, matrices, dists = {}, {}, {}
        for key, cal_data in all_cals.items():
            cam_id = int(key.split("@")[0])
            cals.setdefault(cam_id, cal_data)
            
            meta = cal_data.get("meta", {})
            res = meta.get("resolution", {})
            size = (cam_id, res.get("width"), res.get("height"))
            if "camera_matrix" in meta and "dist_coeffs" in meta:
                matrices[size] = np.array(meta["camera_matrix"], dtype=np.float64)
                dists[size] = np.array(meta["dist_coeffs"], dtype=np.float64)
                logger.Log(f"Loaded OpenCV calibration for camera {cam_id} at {size[1]}x{size[2]}")
            else:
                logger.Warn(f"Camera {cam_id} missing OpenCV calibration data. Please recalibrate using /calibration tab.")
        
        calibrations, _camera_matrices, _dist_coeffs = cals, matrices, dists
        logger.Log(f"Loaded calibrations for cameras: {list(calibrations.keys())}")
        _calibrations_loaded = True
    except FileNotFoundError:
        logger.Error("cal.json not found. Please calibrate cameras using the /calibration tab.")
        _calibrations_loaded = True
    except Exception as e:
        logger.Error(f"Failed to load cal.json: {e}")
        _calibrations_loaded = True

def store_calibration(camera_id: int, result: dict) -> None:
    res = result["meta"]["resolution"]
    try:
        with open(CAL_PATH, "r") as f:
            all_cals = json.load(f)
    except FileNotFoundError:
        all_cals = {}
    if all_cals.get(str(camera_id), {}).get("meta", {}).get("resolution") == res:
        del all_cals[str(camera_id)]
    all_cals[calibration_key(camera_id, res["width"], res["height"])] = result
    with open(CAL_PATH, "w") as f:
        json.dump(all_cals, f, indent=2)
    load_all_calibrations(force=True)

def load_calibration(camera_id: int) -> dict:
    global calibrations
    
    if not _calibrations_loaded:
        load_all_calibrations()
    
    if camera_id in calibrations:
        return calibrations[camera_id]
    
    logger.Warn(f"No calibration found for camera {camera_id}. Please calibrate using /calibration tab.")
    return {}

def has_opencv_calibration(camera_id: int, width: int, height: int) -> bool:
    if not _calibrations_loaded:
        load_all_calibrations()
    
    return (camera_id, width, height) in _camera_matrices

def calibration_for(camera_id: int, width: int, height: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not _calibrations_loaded:
        load_all_calibrations()

    key = (camera_id, width, height)
    camera_matrix = _camera_matrices.get(key)
    dist_coeffs = _dist_coeffs.get(key)
    if camera_matrix is None or dist_coeffs is None:
        if key not in _reported_missing:
            _reported_missing.add(key)
            logger.Error(f"No OpenCV calibration found for camera {camera_id} at {width}x{height}. Please calibrate using the /calibration tab.")
        return None
    return camera_matrix, dist_coeffs

def GET_CAMERA_ANGLES(
    x_img: float,
    y_img: float,
    frame: cv2.typing.MatLike,
    camera_id: int,
    unused_param: np.ndarray = None,
) -> Tuple[float, float]:
    h, w = frame.shape[:2]
    calibration = calibration_for(camera_id, w, h)
    if calibration is None:
        return (0.0, 0.0)
    camera_matrix, dist_coeffs = calibration
    
    points = np.array([[[x_img, y_img]]], dtype=np.float64)
    undistorted = cv2.undistortPoints(points, camera_matrix, dist_coeffs, P=None)
    
    x_norm = undistorted[0, 0, 0]
    y_norm = undistorted[0, 0, 1]
    
    angle_x = math.degrees(math.atan(x_norm))
    angle_y = -math.degrees(math.atan(y_norm))
    
    return angle_x, angle_y

def undistort_frame(frame: cv2.typing.MatLike, camera_id: int) -> Optional[cv2.typing.MatLike]:
    global _camera_matrices, _dist_coeffs
    
    if not _calibrations_loaded:
        load_all_calibrations()
    
    h, w = frame.shape[:2]
    camera_matrix = _camera_matrices.get((camera_id, w, h))
    dist_coeffs = _dist_coeffs.get((camera_id, w, h))
    if camera_matrix is None or dist_coeffs is None:
        return None
    
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    x, y, w_roi, h_roi = roi
    if w_roi > 0 and h_roi > 0:
        undistorted = undistorted[y:y+h_roi, x:x+w_roi]
    
    return undistorted
