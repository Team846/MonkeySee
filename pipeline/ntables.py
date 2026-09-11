from typing import List
import ntcore
from typing import Optional, Tuple

_nt_instance = None
_initialized = False
_server_address = "10.8.46.2"

def set_server(server: str):
    global _server_address
    _server_address = server

def reset_instance():
    global _nt_instance, _initialized
    if _nt_instance is not None:
        try:
            if _initialized:
                _nt_instance.stopClient()
        except AttributeError:
            pass
    _initialized = False

def get_nt_instance():
    global _nt_instance, _initialized
    if _nt_instance is None:
        _nt_instance = ntcore.NetworkTableInstance.getDefault()
    if not _initialized:
        _nt_instance.setServer(_server_address)
        _nt_instance.startClient4("MonkeySee")
        _initialized = True
    return _nt_instance

class AprilTagNTables:
    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        inst = get_nt_instance()
        self.table = inst.getTable(f"AprilTagsCam{camera_id}")
        
        self.latency_pub = self.table.getDoubleTopic("tl").publish()
        self.angles_pub = self.table.getDoubleArrayTopic("tx").publish()
        self.distances_pub = self.table.getDoubleArrayTopic("distances").publish()
        self.tags_pub = self.table.getDoubleArrayTopic("tags").publish()
        self.frame_num_pub = self.table.getDoubleTopic("curFrameNum").publish()
    
    def execute(self, detections, latency):
        self.latency_pub.set(latency)
        
        angles: List[float] = []
        distances: List[float] = []
        tags: List[float] = []
        
        for detection in detections:
            angles.append(detection.getTheta())
            distances.append(detection.getR())
            tags.append(float(detection.getTag()))
        
        self.angles_pub.set(angles)
        self.distances_pub.set(distances)
        self.tags_pub.set(tags)
    
    def updateFrameNum(self, frame_num: int):
        self.frame_num_pub.set(float(frame_num))

class GamePieceNTables:
    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        inst = get_nt_instance()
        self.table = inst.getTable(f"GPDCam{camera_id}")
        
        self.angles_pub = self.table.getDoubleArrayTopic("tx").publish()
        self.distances_pub = self.table.getDoubleArrayTopic("distances").publish()
        self.tops_pub = self.table.getBooleanArrayTopic("on_tops").publish()
        self.latency_pub = self.table.getDoubleTopic("tl").publish()
        self.heights_pub = self.table.getDoubleArrayTopic("heights").publish()
        self.optimal_target_pub = self.table.getDoubleArrayTopic("optimal_target").publish()
    
    def execute(self, detections, latency, optimal_solution: Optional[Tuple[float, float]] = None):
        angles: List[float] = []
        distances: List[float] = []
        tops: List[bool] = []
        heights: List[float] = []
        
        for detection in detections:
            angles.append(detection.getTheta())
            distances.append(detection.getR())
            tops.append(detection.isOnTop())
            heights.append(detection.height)
        
        self.angles_pub.set(angles)
        self.distances_pub.set(distances)
        self.tops_pub.set(tops)
        self.latency_pub.set(latency)
        self.heights_pub.set(heights)
        if optimal_solution is not None:
            self.optimal_target_pub.set([optimal_solution[0], optimal_solution[1]])
        else:
            self.optimal_target_pub.set([])