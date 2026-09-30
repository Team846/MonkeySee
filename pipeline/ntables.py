from typing import List
import ntcore
from time import monotonic_ns
from typing import Optional, Tuple

_nt_instance = None
_initialized = False
_server_address = "10.8.46.2"

_PUB_OPTIONS = ntcore.PubSubOptions(periodic=0.01, sendAll=True)

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

def capture_stamp(capture_ns: int) -> Tuple[float, float]:
    age_s = (monotonic_ns() - capture_ns) / 1e9
    offset_us = get_nt_instance().getServerTimeOffset()
    if offset_us is None:
        return -1.0, age_s
    return (ntcore._now() + offset_us) / 1e6 - age_s, age_s

class AprilTagNTables:
    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        inst = get_nt_instance()
        self.table = inst.getTable(f"AprilTagsCam{camera_id}")

        self.latency_pub = self.table.getDoubleTopic("tl").publish(_PUB_OPTIONS)
        self.angles_pub = self.table.getDoubleArrayTopic("tx").publish(_PUB_OPTIONS)
        self.distances_pub = self.table.getDoubleArrayTopic("distances").publish(_PUB_OPTIONS)
        self.tags_pub = self.table.getDoubleArrayTopic("tags").publish(_PUB_OPTIONS)
        self.frame_num_pub = self.table.getDoubleTopic("curFrameNum").publish(_PUB_OPTIONS)
        self.result_pub = self.table.getDoubleArrayTopic("result").publish(_PUB_OPTIONS)

    def execute(self, detections, latency, capture_ns):
        self.latency_pub.set(latency)

        angles: List[float] = []
        distances: List[float] = []
        tags: List[float] = []
        result: List[float] = list(capture_stamp(capture_ns))

        for detection in detections:
            angles.append(detection.getTheta())
            distances.append(detection.getR())
            tags.append(float(detection.getTag()))
            result += [float(detection.getTag()), detection.getTheta(), detection.getR()]

        self.angles_pub.set(angles)
        self.distances_pub.set(distances)
        self.tags_pub.set(tags)
        self.result_pub.set(result)
        self.table.getInstance().flush()

    def updateFrameNum(self, frame_num: int):
        self.frame_num_pub.set(float(frame_num))

class GamePieceNTables:
    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        inst = get_nt_instance()
        self.table = inst.getTable(f"GPDCam{camera_id}")

        self.angles_pub = self.table.getDoubleArrayTopic("tx").publish(_PUB_OPTIONS)
        self.distances_pub = self.table.getDoubleArrayTopic("distances").publish(_PUB_OPTIONS)
        self.tops_pub = self.table.getBooleanArrayTopic("on_tops").publish(_PUB_OPTIONS)
        self.latency_pub = self.table.getDoubleTopic("tl").publish(_PUB_OPTIONS)
        self.heights_pub = self.table.getDoubleArrayTopic("heights").publish(_PUB_OPTIONS)
        self.optimal_target_pub = self.table.getDoubleArrayTopic("optimal_target").publish(_PUB_OPTIONS)
        # [capture_time_s, capture_latency_s, tx, distance, height, on_top, ...]
        self.result_pub = self.table.getDoubleArrayTopic("result").publish(_PUB_OPTIONS)

    def execute(self, detections, latency, capture_ns, optimal_solution: Optional[Tuple[float, float]] = None):
        angles: List[float] = []
        distances: List[float] = []
        tops: List[bool] = []
        heights: List[float] = []
        result: List[float] = list(capture_stamp(capture_ns))

        for detection in detections:
            angles.append(detection.getTheta())
            distances.append(detection.getR())
            tops.append(detection.isOnTop())
            heights.append(detection.height)
            result += [detection.getTheta(), detection.getR(), detection.height, float(detection.isOnTop())]

        self.angles_pub.set(angles)
        self.distances_pub.set(distances)
        self.tops_pub.set(tops)
        self.latency_pub.set(latency)
        self.heights_pub.set(heights)
        if optimal_solution is not None:
            self.optimal_target_pub.set([optimal_solution[0], optimal_solution[1]])
        else:
            self.optimal_target_pub.set([])
        self.result_pub.set(result)
        self.table.getInstance().flush()
