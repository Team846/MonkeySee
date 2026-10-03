import cv2
import numpy as np
from util.config import ConfigCategory, Config
from util.logger import Logger
from cv2.typing import MatLike

logger = Logger("Detection")

pref_category = ConfigCategory("Detection")

min_thresh_win = pref_category.getIntConfig("min_thresh_win", 3)
max_thresh_win = pref_category.getIntConfig("max_thresh_win", 23)
thresh_step = pref_category.getIntConfig("thresh_step", 17)
persp_px_per_cell = pref_category.getIntConfig("persp_px_per_cell", 10)

corner_refine = pref_category.getIntConfig("corner_refine", 1)
corner_refine_win = pref_category.getIntConfig("corner_refine_win", 5)
corner_refine_iters = pref_category.getIntConfig("corner_refine_iters", 30)
corner_refine_eps = pref_category.getFloatConfig("corner_refine_eps", 0.01)

def GET_THRESH_STEP() -> int:
    return thresh_step.valueFloat()

def SET_THRESH_STEP(value: int):
    thresh_step.setInt(value)

def GET_THRESH_WIN() -> int:
    return max_thresh_win.valueFloat()

def SET_THRESH_WIN(value: int):
    max_thresh_win.setInt(value)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

parameters = cv2.aruco.DetectorParameters()

parameters.adaptiveThreshWinSizeMin = min_thresh_win.valueInt()
parameters.adaptiveThreshWinSizeMax = max_thresh_win.valueInt()
parameters.adaptiveThreshWinSizeStep = thresh_step.valueInt()
parameters.perspectiveRemovePixelPerCell = persp_px_per_cell.valueInt()

_detector = cv2.aruco.ArucoDetector(dictionary, parameters) if hasattr(cv2.aruco, "ArucoDetector") else None

def GET_CORNER_REFINE() -> int:
    return corner_refine.valueInt()

def SET_CORNER_REFINE(value: int):
    corner_refine.setInt(value)

def _refine_corners(corners, refine_image: MatLike):

    if refine_image is None or not corners:
        return corners

    if refine_image.ndim == 3:
        refine_image = cv2.cvtColor(refine_image, cv2.COLOR_BGR2GRAY)

    base_win = max(2, corner_refine_win.valueInt())
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        max(1, corner_refine_iters.valueInt()),
        max(1e-4, corner_refine_eps.valueFloat()),
    )

    refined = []
    for quad in corners:
        pts = np.asarray(quad, dtype=np.float32).reshape(-1, 1, 2)

        sides = [
            float(np.linalg.norm(pts[i, 0] - pts[(i + 1) % 4, 0]))
            for i in range(4)
        ]
        win = int(min(base_win, max(2, min(sides) / 8.0)))
        if win < 2:
            refined.append(quad)
            continue

        try:
            out = cv2.cornerSubPix(refine_image, pts.copy(), (win, win), (-1, -1), criteria)
            refined.append(out.reshape(np.asarray(quad).shape))
        except cv2.error as e:
            logger.Warn(f"cornerSubPix failed, using unrefined corners: {e}")
            refined.append(quad)

    return refined

def DETECT_TAGS(image: MatLike, refine_image: MatLike = None):
    global _detector
    if _detector is not None:
        corners, IDs, _ = _detector.detectMarkers(image)
    else:
        corners, IDs, _ = cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)

    if IDs is None:
        return corners, None

    if corner_refine.valueInt() > 0:
        corners = _refine_corners(corners, refine_image if refine_image is not None else image)

    corners = [np.roll(q, 2, axis=1) for q in corners]

    return corners, IDs.flatten()

def ANNOTATE_TAGS(image: MatLike, corners, IDs) -> MatLike:
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if IDs is not None:
        image = cv2.aruco.drawDetectedMarkers(image, corners, IDs)
    return image