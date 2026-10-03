import cv2
import numpy as np
from localization.undistort import GET_CAMERA_ANGLES, load_calibration, calibration_for
from util.config import ConfigCategory, Config
from util.logger import Logger
from cv2.typing import MatLike
from typing import List, Set, Dict
import math

logger = Logger("AprilTagSolution")

valid_tag_ids: Set[int] = set(range(1, 33))

CAM_PARAMS: Dict[int, Dict[str, Config]] = {}

def SET_VALID_TAG_IDS(tag_ids: List[int]):
    global valid_tag_ids
    valid_tag_ids = set(tag_ids)
    logger.Log(f"Valid AprilTag IDs: {sorted(valid_tag_ids)}")

def SET_CAM(pipeline: int, config_key: str = None):
    section = config_key if config_key is not None else f"AprilTag{pipeline}"
    pref_category = ConfigCategory(section)
    CAM_PARAMS[pipeline] = {
        "CAM_MOUNT_H_deg": pref_category.getFloatConfig("CAM_MOUNT_H_deg", 0.0),
        "CAM_MOUNT_V_deg": pref_category.getFloatConfig("CAM_MOUNT_V_deg", 0.0),
        "TAG_H_in": pref_category.getFloatConfig("TAG_H_in", 6.5),
    }
    logger.Log(f"Camera {pipeline} using config section {section}: "
               f"H={CAM_PARAMS[pipeline]['CAM_MOUNT_H_deg'].valueFloat()}, "
               f"V={CAM_PARAMS[pipeline]['CAM_MOUNT_V_deg'].valueFloat()}, "
               f"TAG_H={CAM_PARAMS[pipeline]['TAG_H_in'].valueFloat()}")
    load_calibration(pipeline)

class Detection:
    def __init__(self, r_ground: float, theta_h: float, tag: int):
        self.r = r_ground
        self.theta = theta_h
        self.tag = tag

    def getTag(self) -> int:
        return self.tag

    def getR(self) -> float:
        return self.r

    def getTheta(self) -> float:
        return self.theta

    def __str__(self) -> str:
        return f"Tag {self.tag}: ({self.r:.2f}, {self.theta:.2f} deg)"

    def __repr__(self) -> str:
        return self.__str__()

def CALCULATE_PARTIAL_SOLUTION(camera_id: int, image: MatLike, all_corners, all_IDs) -> List[Detection]:
    result = []

    if all_IDs is None:
        return result

    params = CAM_PARAMS.get(camera_id)
    if params is None:
        logger.Warn(f"No AprilTag params for camera {camera_id}; call SET_CAM first")
        return result

    height, width = image.shape[:2]
    if calibration_for(camera_id, width, height) is None:
        return result

    cam_angle_h = params["CAM_MOUNT_H_deg"].valueFloat()
    cam_angle_v = params["CAM_MOUNT_V_deg"].valueFloat()
    tag_h = params["TAG_H_in"].valueFloat()

    for corners, tID in zip(all_corners, all_IDs):
        if tID not in valid_tag_ids:
            continue

        corners = corners.flatten()
        x_center: float = (corners[0] + corners[2] + corners[4] + corners[6]) / 4.0
        y_center: float = (corners[1] + corners[3] + corners[5] + corners[7]) / 4.0
        y_top: float = (corners[1] + corners[3]) / 2.0
        y_bottom: float = (corners[5] + corners[7]) / 2.0

        tx_c, ty_c = GET_CAMERA_ANGLES(
            x_center,
            y_center,
            image,
            camera_id,
            None,
        )
        _, ty_t = GET_CAMERA_ANGLES(
            x_center,
            y_top,
            image,
            camera_id,
            None,
        )
        _, ty_b = GET_CAMERA_ANGLES(
            x_center,
            y_bottom,
            image,
            camera_id,
            None,
        )

        subtense = math.tan(math.radians(ty_t + cam_angle_v)) - math.tan(
            math.radians(ty_b + cam_angle_v)
        )
        if abs(subtense) < 1e-9:
            logger.Warn(
                f"Camera {camera_id}: divide by zero, check if camera has calibration"
            )
            continue
        d_forward: float = tag_h / abs(subtense)

        pitch = math.radians(cam_angle_v)
        x_n = math.tan(math.radians(tx_c))
        y_n_up = math.tan(math.radians(ty_c))
        bearing = math.atan2(x_n, math.cos(pitch) - y_n_up * math.sin(pitch))

        r_ground = d_forward / math.cos(bearing)
        theta = math.degrees(bearing) + cam_angle_h
        result.append(Detection(r_ground, theta, tID))

    return result
