from typing import Tuple, Dict
import cv2
import numpy as np
import math
import json
from util.logger import Logger

logger = Logger("Calibration")

calibrations: Dict[int, dict] = {}
_calibrations_loaded = False

def load_all_calibrations():
    global calibrations, _calibrations_loaded
    
    if _calibrations_loaded:
        return
    
    try:
        with open("cal.json", "r") as f:
            all_cals = json.load(f)
        
        for cam_id_str, cal_data in all_cals.items():
            calibrations[int(cam_id_str)] = cal_data
        
        logger.Log(f"Loaded calibrations for cameras: {list(calibrations.keys())}")
        _calibrations_loaded = True
    except Exception as e:
        logger.Error(f"Failed to load cal.json: {e}. Using default calibration.")
        default_cal = {
            "x": {"a": 0.0018803150360736042, "b": -1.4941116971361661e-09, "c": -4.5622203794325465e-15},
            "y": {"a": 0.0017131708207895275, "b": -1.8890823506313983e-09, "c": 7.353164417891053e-15},
            "meta": {"resolution": {"width": 800, "height": 600}}
        }
        calibrations[0] = default_cal
        _calibrations_loaded = True

def load_calibration(camera_id: int) -> dict:
    global calibrations
    
    if not _calibrations_loaded:
        load_all_calibrations()
    
    if camera_id in calibrations:
        return calibrations[camera_id]
    
    logger.Warn(f"No calibration found for camera {camera_id}, using default")
    default_cal = {
        "x": {"a": 0.0018803150360736042, "b": -1.4941116971361661e-09, "c": -4.5622203794325465e-15},
        "y": {"a": 0.0017131708207895275, "b": -1.8890823506313983e-09, "c": 7.353164417891053e-15},
        "meta": {"resolution": {"width": 800, "height": 600}}
    }
    calibrations[camera_id] = default_cal
    return default_cal


def GET_CAMERA_ANGLES(
    x_img: float,
    y_img: float,
    frame: cv2.typing.MatLike,
    camera_id: int,
    unused_param: np.ndarray = None,
) -> Tuple[float, float]:
    global calibrations
    
    cal = calibrations.get(camera_id, {})
    
    q_x = cal.get("x", {"a": 0, "b": 0, "c": 0})
    q_y = cal.get("y", {"a": 0, "b": 0, "c": 0})
    
    meta = cal.get("meta", {})
    cal_width = meta.get("resolution", {}).get("width", 800)
    cal_height = meta.get("resolution", {}).get("height", 600)
    
    h, w = frame.shape[:2]
    
    x_scaled = x_img * (cal_width / w)
    y_scaled = y_img * (cal_height / h)
    
    x_centered = x_scaled - cal_width / 2.0
    y_centered = cal_height / 2.0 - y_scaled
    
    angle_x = math.degrees(q_x["a"] * x_centered + q_x["b"] * x_centered**3 + q_x["c"] * x_centered**5)
    angle_y = math.degrees(q_y["a"] * y_centered + q_y["b"] * y_centered**3 + q_y["c"] * y_centered**5)
    
    return angle_x, angle_y
