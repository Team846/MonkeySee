from localization.undistort import GET_CAMERA_ANGLES, load_calibration
from util.config import ConfigCategory, Config
from util.logger import Logger
from cv2.typing import MatLike
from typing import Dict, List, Set, Tuple
import math

logger = Logger("AprilTagSolution")

valid_tag_ids: Set[int] = set(range(1, 33))

_cam_prefs: Dict[int, Tuple[Config, Config, Config]] = {}


def SET_VALID_TAG_IDS(tag_ids: List[int]):
    global valid_tag_ids
    valid_tag_ids = set(tag_ids)
    logger.Log(f"Valid AprilTag IDs: {sorted(valid_tag_ids)}")


def SET_CAM(pipeline: int):
    pref_category = ConfigCategory(f"AprilTag{pipeline}")
    cam_h = pref_category.getFloatConfig("CAM_MOUNT_H_deg", 0.0)
    cam_v = pref_category.getFloatConfig("CAM_MOUNT_V_deg", 0.0)
    tag_h = pref_category.getFloatConfig("TAG_H_in", 6.5)
    _cam_prefs[pipeline] = (cam_h, cam_v, tag_h)
    load_calibration(pipeline)
    logger.Log(
        f"AprilTag cam {pipeline}: H={cam_h.valueFloat()}, "
        f"V={cam_v.valueFloat()}, TAG_H={tag_h.valueFloat()}"
    )


def _get_cam_prefs(camera_id: int) -> Tuple[float, float, float]:
    prefs = _cam_prefs.get(camera_id)
    if prefs is None:
        SET_CAM(camera_id)
        prefs = _cam_prefs[camera_id]
    cam_h, cam_v, tag_h = prefs
    return cam_h.valueFloat(), cam_v.valueFloat(), tag_h.valueFloat()


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

    cam_h, cam_v, tag_h = _get_cam_prefs(camera_id)

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
        tx_c += cam_h

        r_cam: float = tag_h / abs(
            math.tan(math.radians(ty_t + cam_v))
            - math.tan(math.radians(ty_b + cam_v))
        )
        r_ground = r_cam / math.cos(math.radians(tx_c - cam_h))
        result.append(Detection(r_ground, tx_c, tID))

    return result
