import cv2
import numpy as np
from localization.undistort import GET_CAMERA_ANGLES, load_calibration
from util.config import ConfigCategory, Config
from util.logger import Logger
from cv2.typing import MatLike
from typing import List, Set
import math

logger = Logger("AprilTagSolution")

valid_tag_ids: Set[int] = set(range(1, 33))

CAM_ANGLE_H = 0
CAM_ANGLE_V = 0
TAG_H = 0

def SET_VALID_TAG_IDS(tag_ids: List[int]):
    global valid_tag_ids
    valid_tag_ids = set(tag_ids)
    logger.Log(f"Valid AprilTag IDs: {sorted(valid_tag_ids)}")

def SET_CAM(pipeline: int):
    pref_category = ConfigCategory(f"AprilTag{pipeline}")
    global CAM_ANGLE_H, CAM_ANGLE_V, TAG_H
    CAM_ANGLE_H = pref_category.getFloatConfig("CAM_MOUNT_H_deg", 0.0)
    CAM_ANGLE_V = pref_category.getFloatConfig("CAM_MOUNT_V_deg", 0.0)
    TAG_H = pref_category.getFloatConfig("TAG_H_in", 6.5)
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
    global CAM_ANGLE_H, TAG_H, CAM_ANGLE_V

    result = []

    if all_IDs is None:
        return result

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
        tx_c += CAM_ANGLE_H.valueFloat()

        r_cam: float = TAG_H.valueFloat() / abs(
            math.tan(math.radians(ty_t + CAM_ANGLE_V.valueFloat()))
            - math.tan(math.radians(ty_b + CAM_ANGLE_V.valueFloat()))
        )
        r_ground = r_cam / math.cos(math.radians(tx_c - CAM_ANGLE_H.valueFloat()))
        result.append(Detection(r_ground, tx_c, tID))

    return result

