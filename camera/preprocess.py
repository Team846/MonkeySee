import cv2
import numpy as np
from cv2.typing import MatLike
from util.config import ConfigCategory, Config
from typing import Tuple
from util.numba_utils import njit_cached

pref_category = ConfigCategory("Preprocessing")

acc_num_bins = pref_category.getIntConfig("acc_num_bins", 1200)
target_brightness = pref_category.getIntConfig("target_brightness", 155)
min_corr_strength = pref_category.getFloatConfig("min_corr_strength", 0.1)
corr_divisor = pref_category.getFloatConfig("corr_divisor", 400.0)
divergence_gain = pref_category.getFloatConfig("divergence_gain", 1.5)

@njit_cached(nogil=True)
def COMPUTE_CORRECTION_MATRIX(image: np.ndarray, bins_per_side: int, target_brightness: int) -> np.ndarray:
    height, width = image.shape
    bin_height = height // bins_per_side
    bin_width = width // bins_per_side

    bin_means = np.array([
        np.mean(image[i * bin_height:(i + 1) * bin_height, j * bin_width:(j + 1) * bin_width])
        for i in range(bins_per_side) for j in range(bins_per_side)
    ]).reshape(bins_per_side, bins_per_side)

    correction_matrix = target_brightness - bin_means

    return correction_matrix

def BIN_BASED_CORRECTION(image: np.ndarray, acc_num_bins: int, target_brightness: int, min_corr_strength: float, corr_divisor: float) -> Tuple[np.ndarray, float, float]:
    height, width = image.shape
    num_bins = acc_num_bins
    bins_per_side = int(np.sqrt(num_bins))

    mean = np.mean(image)
    corr_strength = abs(target_brightness - mean) / corr_divisor + min_corr_strength

    correction_matrix = COMPUTE_CORRECTION_MATRIX(image, bins_per_side, target_brightness)

    correction_matrix = cv2.blur(correction_matrix, (3, 3), 0)

    correction_matrix = cv2.resize(correction_matrix.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)

    corrected_mean = mean + corr_strength * cv2.mean(correction_matrix)[0]

    return correction_matrix, corr_strength, corrected_mean

@njit_cached(nogil=True)
def CORRECT_AND_DIVERGE(image: np.ndarray, correction_matrix: np.ndarray, corr_strength: float, divergence_gain: float, mean: float) -> np.ndarray:
    height, width = image.shape

    corrected_image = np.empty((height, width), np.uint8)
    for y in range(height):
        for x in range(width):
            v = image[y, x] + correction_matrix[y, x] * corr_strength
            v = v * ((divergence_gain * (v - mean) + mean) / mean)
            corrected_image[y, x] = min(max(v, 0.0), 255.0)

    return corrected_image

def PROCESS_FRAME(image: MatLike) -> MatLike:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    correction_matrix, corr_strength, mean = BIN_BASED_CORRECTION(image, acc_num_bins.valueInt(), target_brightness.valueInt(), min_corr_strength.valueFloat(), corr_divisor.valueFloat())
    image = CORRECT_AND_DIVERGE(image, correction_matrix, corr_strength, divergence_gain.valueFloat(), mean)

    return image

def GET_DIVERGENCE_GAIN() -> float:
    return divergence_gain.valueFloat()

def SET_DIVERGENCE_GAIN(value: float) -> None:
    divergence_gain.setFloat(value)

def GET_TARGET_BRIGHTNESS() -> int:
    return target_brightness.valueInt()

def SET_TARGET_BRIGHTNESS(value: int) -> None:
    target_brightness.setInt(value)

def GET_NUM_BINS() -> int:
    return acc_num_bins.valueInt()

def SET_NUM_BINS(value: int) -> None:
    acc_num_bins.setInt(value)

def GET_MIN_CORR_STRENGTH() -> float:
    return min_corr_strength.valueFloat()

def SET_MIN_CORR_STRENGTH(value: float) -> None:
    min_corr_strength.setFloat(value)