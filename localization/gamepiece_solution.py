from util.config import ConfigCategory, Config
from util.logger import Logger
from cv2.typing import MatLike
from typing import List, Tuple
import math
from localization.undistort import load_calibration, GET_CAMERA_ANGLES, calibrations

logger = Logger("GamePieceSolution")

pref_category = ConfigCategory(f"GamePieceSolution")
WD = pref_category.getFloatConfig("WD", 7)
ONT = pref_category.getFloatConfig("ONT", 10.0)
CAM_HEIGHT = pref_category.getFloatConfig("CAM_HEIGHT", 14.5) 
MIN_DISTANCE = pref_category.getFloatConfig("MIN_DISTANCE", 2.0) 
MAX_DISTANCE = pref_category.getFloatConfig("MAX_DISTANCE", 200.0)  
SMOOTHING = pref_category.getFloatConfig("SMOOTHING", 0.2)  

_prev_optimal_dist: float = 0.0

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

def horizontal_angle(x_pos: float, frame: MatLike, camera_id: int) -> float:
    global calibrations
    
    cal = calibrations.get(camera_id, {})
    q = cal.get("x", {"a": 0, "b": 0, "c": 0})
    
    meta = cal.get("meta", {})
    w = meta.get("resolution", {}).get("width", 800)
    
    x_pos *= w / 256.0
    
    return math.degrees(q["a"] * x_pos + q["b"] * x_pos**3 + q["c"] * x_pos**5)

def vertical_angle(y_pos: float, frame: MatLike, camera_id: int) -> float:
    global calibrations
    
    cal = calibrations.get(camera_id, {})
    q = cal.get("y", {"a": 0, "b": 0, "c": 0})
    
    meta = cal.get("meta", {})

    h = 256 #fix later
    
    y_pos *= h / 256.0
    
    return math.degrees(q["a"] * y_pos + q["b"] * y_pos**3 + q["c"] * y_pos**5)


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


#detections: List[Detection]
def CALCULATE_OPTIMAL_POINT_SOLUTION(camera_id: int, image: MatLike, optimal_point: Tuple[float, float], raw_boxes: List[Tuple[float, float, float, float]]) -> Tuple[float, float]:

    global _prev_optimal_dist, calibrations
    
    try:
        x, y = optimal_point


        # x_centered = x - 640 / 2
        # x_in_256 = x_centered * 256 / 640
        # theta_h = horizontal_angle(x_in_256, image, camera_id)
        # y_centered = 640 / 2 - y
        # y_in_256 = y_centered * 256 / 640
        # CAMERA_PITCH = 0#32.93 
        # camera_pitch_rad = math.radians(CAMERA_PITCH)
        # elevation_angle = math.radians(y_in_256 / 128.0 * 46.5 / 2.0)
        # print(f"elevation angle: {elevation_angle}")
        # depression_angle = -(elevation_angle - camera_pitch_rad)
        # raw_dist = abs(CAM_HEIGHT.valueFloat() / math.tan(depression_angle))
        # return (raw_dist, theta_h)

        angle_x, angle_y = pixel_to_angles(x, y, image, camera_id)

        angle_x = (x-128)/(256.0) * 60.5
        CAMERA_PITCH = 0  
        camera_pitch_rad = math.radians(CAMERA_PITCH)
        depression_angle = -(math.radians(angle_y) - camera_pitch_rad)
        try:
            raw_dist = abs(CAM_HEIGHT.valueFloat() / math.tan(depression_angle))
        except Exception as e:
            logger.Error(f"Failed to calculate distance: {e}")
            raw_dist = 0.0
        return (raw_dist,angle_x)
        
    except Exception as e:
        logger.Error(f"Failed to calculate optimal point solution: {e}")
        return (0.0, 0.0)

