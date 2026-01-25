from util.config import ConfigCategory, Config
from util.logger import Logger
from cv2.typing import MatLike
from typing import List, Tuple
import math
from localization.undistort import load_calibration, GET_CAMERA_ANGLES

logger = Logger("GamePieceSolution")

pref_category = ConfigCategory(f"GamePieceSolution")

WD = pref_category.getFloatConfig("WD", 15.5)
ONT = pref_category.getFloatConfig("ONT", 10.0)

def SET_CAM(pipeline: int):
    load_calibration(pipeline)

class Detection:
    def __init__(self, r_ground: float, theta_h: float, is_on_top: bool, h: float = 0.0):
        self.r = r_ground
        self.theta = theta_h
        self.is_on_top = is_on_top
        self.height = h

    def getR(self) -> float:
        return self.r

    def getTheta(self) -> float:
        return self.theta

    def isOnTop(self) -> bool:
        return self.is_on_top

    def __str__(self) -> str:
        return f"Detection: ({self.r:.2f}, {self.theta:.2f} deg). Top: {self.isOnTop()}"

    def __repr__(self) -> str:
        return self.__str__()

def pixel_to_angles(x_pos: float, y_pos: float, frame: MatLike, camera_id: int) -> Tuple[float, float]:
    cal = load_calibration(camera_id)
    meta = cal.get("meta", {})
    w = meta.get("resolution", {}).get("width", 640)
    h = meta.get("resolution", {}).get("height", 480)
    
    x_img = x_pos * (w / 256.0)
    y_img = y_pos * (h / 256.0)
    
    angle_x, angle_y = GET_CAMERA_ANGLES(x_img, y_img, frame, camera_id)
    
    return angle_x, angle_y

def CALCULATE_PARTIAL_SOLUTION(camera_id: int, image: MatLike, objs: List[Tuple[float, float]]) -> List[Detection]:
    global WD, ONT
    
    result = []
    
    for obj in objs:
        try:
            objr = [0.0, 0.0, 0.0, 0.0]
            objr[0] = -256 / 2 + obj[0]
            objr[2] = -256 / 2 + obj[2]
            objr[1] = 256 / 2 - obj[1]
            objr[3] = 256 / 2 - obj[3]
            
            l_h, _ = pixel_to_angles(objr[0], 0.0, image, camera_id)
            r_h, _ = pixel_to_angles(objr[2], 0.0, image, camera_id)
            
            _, d_v = pixel_to_angles(0.0, objr[1], image, camera_id)
            _, u_v = pixel_to_angles(0.0, objr[3], image, camera_id)
            
            l_h = math.radians(l_h)
            r_h = math.radians(r_h)
            d_v = math.radians(d_v)
            u_v = math.radians(u_v)
            
            r_ground: float = WD.valueFloat() / (math.tan(r_h) - math.tan(l_h))
            r_ground += WD.valueFloat() / (math.tan(d_v) - math.tan(u_v))
            r_ground /= 2.0
            
            height: float = r_ground * math.tan((d_v + u_v) / 2.0)
            
            is_on_top = height > ONT.valueFloat()
            
            center_angle = math.degrees(l_h + r_h) / 2.0
            
            result.append(Detection(r_ground, center_angle, is_on_top, height))
        except Exception as e:
            logger.Error(f"Failed to process object {obj}: {e}")
    
    return result

