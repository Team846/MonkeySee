import cv2
import numpy as np
from numba import njit
from typing import Tuple, List, Optional
from util.config import ConfigCategory, Config
from util.logger import Logger

logger = Logger("Clustering")

pref_category = ConfigCategory("Clustering")

EPS_MULTIPLIER = pref_category.getFloatConfig("EPS_MULTIPLIER", 1.5)
MIN_SAMPLES = pref_category.getIntConfig("MIN_SAMPLES", 2)
DEFAULT_EPS = pref_category.getFloatConfig("DEFAULT_EPS", 30.0)


@njit(cache=True, fastmath=True)
def fast_dbscan(centers, eps_sq, min_samples):
    n = len(centers)
    if n == 0:
        return np.empty(0, dtype=np.int32)
    
    labels = np.full(n, -1, dtype=np.int32)
    cluster_id = 0
    
    for i in range(n):
        if labels[i] != -1:
            continue
        
        neighbors_i = []
        for j in range(n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            if dx * dx + dy * dy <= eps_sq:
                neighbors_i.append(j)
        
        if len(neighbors_i) < min_samples:
            continue
        
        labels[i] = cluster_id
        stack = neighbors_i.copy()
        
        while stack:
            p = stack.pop()
            if labels[p] != -1 and labels[p] != cluster_id:
                continue
            if labels[p] == cluster_id:
                continue
            labels[p] = cluster_id
            
            neighbors_p = []
            for j in range(n):
                dx = centers[p, 0] - centers[j, 0]
                dy = centers[p, 1] - centers[j, 1]
                if dx * dx + dy * dy <= eps_sq:
                    neighbors_p.append(j)
            
            if len(neighbors_p) >= min_samples:
                stack.extend(neighbors_p)
        
        cluster_id += 1
    
    return labels

@njit(cache=True, fastmath=True)
def find_optimal_cluster_target(centers, labels, camera_x, camera_y):

    n = len(centers)
    if n == 0:
        return 0.0, 0.0
    
    if n == 1:
        return centers[0, 0], centers[0, 1]
    
    all_noise = True
    for i in range(n):
        if labels[i] != -1:
            all_noise = False
            break

    if all_noise:
        best_idx = 0
        best_dist_sq = 1e9
        for i in range(n):
            dx = centers[i, 0] - camera_x
            dy = centers[i, 1] - camera_y
            dist_sq = dx * dx + dy * dy
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_idx = i
        return centers[best_idx, 0], centers[best_idx, 1]
    
    # best cluster
    label_set = set()
    for i in range(n):
        if labels[i] != -1:
            label_set.add(labels[i])
    
    best_cluster = -1
    best_cluster_score = -1.0
    
    for label in label_set:
        count = 0
        sum_dist_sq = 0.0
        
        for i in range(n):
            if labels[i] == label:
                dx = centers[i, 0] - camera_x
                dy = centers[i, 1] - camera_y
                sum_dist_sq += dx * dx + dy * dy
                count += 1
        
        if count > 0:
            avg_dist_sq = sum_dist_sq / count
            avg_dist = np.sqrt(avg_dist_sq) 
            
            score = (count * count * 100.0) / (avg_dist + 10.0)
            
            if score > best_cluster_score:
                best_cluster_score = score
                best_cluster = label
    
    if best_cluster == -1:
        return camera_x, camera_y
    
    total_weight = 0.0
    weighted_x = 0.0
    weighted_y = 0.0
    
    for i in range(n):
        if labels[i] != best_cluster:
            continue
        
        dx = centers[i, 0] - camera_x
        dy = centers[i, 1] - camera_y
        dist_sq = dx * dx + dy * dy
        
        # weight = 1 / distance squared 
        weight = 1.0 / (dist_sq + 1000.0)
        
        total_weight += weight
        weighted_x += centers[i, 0] * weight
        weighted_y += centers[i, 1] * weight
    
    if total_weight > 0:
        return weighted_x / total_weight, weighted_y / total_weight
    return camera_x, camera_y


def get_centers_from_boxes(boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    if not boxes:
        return np.empty((0, 2), dtype=np.float32)
    
    centers = np.empty((len(boxes), 2), dtype=np.float32)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        centers[i, 0] = (x1 + x2) * 0.5
        centers[i, 1] = (y1 + y2) * 0.5
    return centers


def get_avg_diameter_from_boxes(boxes: List[Tuple[int, int, int, int]]) -> float:
    if not boxes:
        return DEFAULT_EPS.valueFloat()
    
    widths = [box[2] - box[0] for box in boxes]
    return float(np.mean(widths))


class ClusteringProcessor:
    
    def __init__(self, frame_width: int = 256, frame_height: int = 256):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.camera_x = frame_width / 2.0
        self.camera_y = float(frame_height)
        
        self.last_target: Optional[Tuple[float, float]] = None
        self.last_cluster_count = 0
        self.last_detection_count = 0
        
        self._warmup_jit()
    
    def _warmup_jit(self):
        dummy_centers = np.zeros((2, 2), dtype=np.float32)
        _ = fast_dbscan(dummy_centers, 100.0 ** 2, 2)
        _ = find_optimal_cluster_target(dummy_centers, np.zeros(2, dtype=np.int32), 0.0, 0.0)
    
    def process(self, boxes: List[Tuple[int, int, int, int]]) -> Optional[Tuple[float, float]]:

        if not boxes:
            self.last_target = None
            self.last_cluster_count = 0
            self.last_detection_count = 0
            return None
        
        centers = get_centers_from_boxes(boxes)
        self.last_detection_count = len(centers)
        
        avg_diameter = get_avg_diameter_from_boxes(boxes)
        eps = avg_diameter * EPS_MULTIPLIER.valueFloat()
        min_samples = MIN_SAMPLES.valueInt()
        
        labels = fast_dbscan(centers, eps ** 2, min_samples)
        
        unique_labels = set(labels)
        self.last_cluster_count = len(unique_labels) - (1 if -1 in labels else 0)
        
        opt_x, opt_y = find_optimal_cluster_target(centers, labels, self.camera_x, self.camera_y)
        
        self.last_target = (opt_x, opt_y)
        return self.last_target
    
    def annotate_frame(self, frame: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        annotated = frame.copy()
        
        
        prev_target = self.last_target
        
        target = self.process(boxes)
        if target is not None and prev_target is not None:
            dx = target[0] - prev_target[0]
            dy = target[1] - prev_target[1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist <= 0.5*DEFAULT_EPS.valueFloat():
                target = prev_target
                self.last_target = prev_target  
        if target is not None:
            opt_x, opt_y = target
            center = (int(opt_x), int(opt_y))
            
            cv2.drawMarker(
                annotated, center, 
                color=(0, 255, 0),  
                markerType=cv2.MARKER_CROSS,
                markerSize=20,
                thickness=3
            )
            
            cv2.circle(annotated, center, 12, (0, 255, 0), 2)
        
        return annotated
    
    def get_stats(self) -> dict:
        return {
            "detection_count": self.last_detection_count,
            "cluster_count": self.last_cluster_count,
            "target": self.last_target
        }


_processor: Optional[ClusteringProcessor] = None


def get_processor(frame_width: int = 640, frame_height: int = 640) -> ClusteringProcessor:
    global _processor
    if _processor is None:
        _processor = ClusteringProcessor(frame_width, frame_height)
    return _processor


def process_detections(boxes: List[Tuple[int, int, int, int]], 
                       frame_width: int =640, 
                       frame_height: int = 640) -> Optional[Tuple[float, float]]:

    processor = get_processor(frame_width, frame_height)
    return processor.process(boxes)


def annotate_with_target(frame: np.ndarray, 
                         boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:

    processor = get_processor(frame.shape[1], frame.shape[0])
    return processor.annotate_frame(frame, boxes)