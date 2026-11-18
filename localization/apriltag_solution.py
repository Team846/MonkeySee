import cv2
import numpy as np
from localization.undistort import load_calibration, calibration_for
from util.config import ConfigCategory, Config
from util.logger import Logger
from cv2.typing import MatLike
from typing import List, Set, Dict
import math

logger = Logger("AprilTagSolution")

valid_tag_ids: Set[int] = set(range(1, 33))

CAM_PARAMS: Dict[int, Dict[str, Config]] = {}

UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 1e-10)

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
        "CAM_MOUNT_ROLL_deg": pref_category.getFloatConfig("CAM_MOUNT_ROLL_deg", 0.0),
        "TAG_H_in": pref_category.getFloatConfig("TAG_H_in", 6.5),
    }
    logger.Log(f"Camera {pipeline} using config section {section}: "
               f"H={CAM_PARAMS[pipeline]['CAM_MOUNT_H_deg'].valueFloat()}, "
               f"V={CAM_PARAMS[pipeline]['CAM_MOUNT_V_deg'].valueFloat()}, "
               f"roll={CAM_PARAMS[pipeline]['CAM_MOUNT_ROLL_deg'].valueFloat()}, "
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
    calibration = calibration_for(camera_id, width, height)
    if calibration is None:
        return result
    camera_matrix, dist_coeffs = calibration

    cam_angle_h = params["CAM_MOUNT_H_deg"].valueFloat()
    pitch = math.radians(params["CAM_MOUNT_V_deg"].valueFloat())
    roll = math.radians(params["CAM_MOUNT_ROLL_deg"].valueFloat())
    tag_h = params["TAG_H_in"].valueFloat()

    for corners, tID in zip(all_corners, all_IDs):
        if tID not in valid_tag_ids:
            continue

        points = np.asarray(corners, dtype=np.float64).reshape(4, 1, 2)
        x, y = cv2.undistortPointsIter(points, camera_matrix, dist_coeffs, None, None, UNDISTORT_CRITERIA).reshape(4, 2).T
        x, y = x * math.cos(roll) - y * math.sin(roll), x * math.sin(roll) + y * math.cos(roll)
        forward = math.cos(pitch) + y * math.sin(pitch)
        up = (math.sin(pitch) - y * math.cos(pitch)) / forward
        side = x / forward

        left, right = abs(up[0] - up[3]), abs(up[1] - up[2])
        if min(left, right) < 1e-9:
            logger.Warn(f"Camera {camera_id}: tag {tID} has no height, check corner order and calibration")
            continue
        z_left, z_right = tag_h / left, tag_h / right
        x_center = (z_left * (side[0] + side[3]) + z_right * (side[1] + side[2])) / 4.0
        z_center = (z_left + z_right) / 2.0

        r_ground = math.hypot(x_center, z_center)
        theta = math.degrees(math.atan2(x_center, z_center)) + cam_angle_h
        result.append(Detection(r_ground, theta, tID))

    return result

if __name__ == "__main__":
    import localization.undistort as undistort

    class _Setting:
        def __init__(self, value):
            self.value = value
        def valueFloat(self):
            return self.value

    K = np.array([[910.3, 0.0, 630.6], [0.0, 914.6, 379.8], [0.0, 0.0, 1.0]])
    D = np.array([0.0282, 0.0837, -0.0035, 0.0006, -0.2264])
    undistort._calibrations_loaded = True
    undistort._camera_matrices[(99, 1280, 800)], undistort._dist_coeffs[(99, 1280, 800)] = K, D
    image = np.zeros((800, 1280), np.uint8)
    rng = np.random.default_rng(0)
    for pitch_deg, roll_deg in ((0.0, 0.0), (21.0, 0.0), (-10.0, 0.0), (23.0, -1.5), (0.0, 3.0)):
        CAM_PARAMS[99] = {"CAM_MOUNT_H_deg": _Setting(30.0), "CAM_MOUNT_V_deg": _Setting(pitch_deg), "CAM_MOUNT_ROLL_deg": _Setting(roll_deg), "TAG_H_in": _Setting(6.5)}
        p, q = math.radians(pitch_deg), math.radians(roll_deg)
        level = np.array([[1, 0, 0], [0, -math.cos(p), math.sin(p)], [0, math.sin(p), math.cos(p)]])
        to_cam = np.array([[math.cos(q), math.sin(q), 0], [-math.sin(q), math.cos(q), 0], [0, 0, 1]]) @ level
        checked = 0
        while checked < 300:
            r, bearing, yaw = rng.uniform(20, 300), rng.uniform(-33, 33), rng.uniform(-65, 65)
            b, a = math.radians(bearing), math.radians(bearing + yaw)
            center = np.array([r * math.sin(b), rng.uniform(-40, 80), r * math.cos(b)])
            across, up = 3.25 * np.array([math.cos(a), 0.0, -math.sin(a)]), np.array([0.0, 3.25, 0.0])
            cam = np.array([center - across + up, center + across + up, center + across - up, center - across - up]) @ to_cam.T
            if cam[:, 2].min() < 6:
                continue
            px = cv2.projectPoints(cam, np.zeros(3), np.zeros(3), K, D)[0].reshape(1, 4, 2)
            if (px < 0).any() or (px[..., 0] > 1279).any() or (px[..., 1] > 799).any():
                continue
            for order in ([0, 1, 2, 3], [2, 3, 0, 1]):
                found = CALCULATE_PARTIAL_SOLUTION(99, image, [px[:, order]], np.array([7]))
                assert len(found) == 1, found
                assert abs(found[0].getR() / r - 1) < 2e-4 and abs(found[0].getTheta() - bearing - 30.0) < 2e-3, (pitch_deg, roll_deg, r, bearing, yaw, found)
            checked += 1
    print("apriltag solution ok")
