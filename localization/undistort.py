from typing import Tuple, Dict, Optional
import cv2
import numpy as np
import math
import json
from util.logger import Logger

logger = Logger("Calibration")

calibrations: Dict[int, dict] = {}
_calibrations_loaded = False

_camera_matrices: Dict[int, np.ndarray] = {}
_dist_coeffs: Dict[int, np.ndarray] = {}

def load_all_calibrations():
    global calibrations, _calibrations_loaded, _camera_matrices, _dist_coeffs
    
    if _calibrations_loaded:
        return
    
    try:
        with open("cal.json", "r") as f:
            all_cals = json.load(f)
        
        for cam_id_str, cal_data in all_cals.items():
            cam_id = int(cam_id_str)
            calibrations[cam_id] = cal_data
            
            meta = cal_data.get("meta", {})
            if "camera_matrix" in meta and "dist_coeffs" in meta:
                _camera_matrices[cam_id] = np.array(meta["camera_matrix"], dtype=np.float64)
                _dist_coeffs[cam_id] = np.array(meta["dist_coeffs"], dtype=np.float64)
                logger.Log(f"Loaded OpenCV calibration for camera {cam_id}")
            else:
                logger.Warn(f"Camera {cam_id} missing OpenCV calibration data. Please recalibrate using /calibration tab.")
        
        logger.Log(f"Loaded calibrations for cameras: {list(calibrations.keys())}")
        _calibrations_loaded = True
    except FileNotFoundError:
        logger.Error("cal.json not found. Please calibrate cameras using the /calibration tab.")
        _calibrations_loaded = True
    except Exception as e:
        logger.Error(f"Failed to load cal.json: {e}")
        _calibrations_loaded = True

def load_calibration(camera_id: int) -> dict:
    
    global calibrations
    
    if not _calibrations_loaded:
        load_all_calibrations()
    
    if camera_id in calibrations:
        return calibrations[camera_id]
    
    return {}
    
    logger.Warn(f"No calibration found for camera {camera_id}. Please calibrate using /calibration tab.")
    return {}

def has_opencv_calibration(camera_id: int) -> bool:
    if not _calibrations_loaded:
        load_all_calibrations()
    
    return camera_id in _camera_matrices and camera_id in _dist_coeffs

def GET_CAMERA_ANGLES(
    x_img: float,
    y_img: float,
    frame: cv2.typing.MatLike,
    camera_id: int,
    unused_param: np.ndarray = None,
) -> Tuple[float, float]:
    
    global _camera_matrices, _dist_coeffs
    
    if not _calibrations_loaded:
        load_all_calibrations()
    
    if camera_id not in _camera_matrices or camera_id not in _dist_coeffs:
        # Only log once per camera to avoid spamming
        if not hasattr(GET_CAMERA_ANGLES, "_warned"): GET_CAMERA_ANGLES._warned = set()
        if camera_id not in GET_CAMERA_ANGLES._warned:
            logger.Warn(f"No OpenCV calibration found for camera {camera_id}. Using default angles (0.0, 0.0). Please calibrate using the /calibration tab.")
            GET_CAMERA_ANGLES._warned.add(camera_id)
        return (0.0, 0.0)
    
    camera_matrix = _camera_matrices[camera_id]
    dist_coeffs = _dist_coeffs[camera_id]
    
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
    
    if camera_id not in _camera_matrices or camera_id not in _dist_coeffs:
        return None
    
    camera_matrix = _camera_matrices[camera_id]
    dist_coeffs = _dist_coeffs[camera_id]
    
    h, w = frame.shape[:2]
    
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    x, y, w_roi, h_roi = roi
    if w_roi > 0 and h_roi > 0:
        undistorted = undistorted[y:y+h_roi, x:x+w_roi]
    
    return undistorted
