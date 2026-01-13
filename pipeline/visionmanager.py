from camera.camerareader import CameraReader
import cv2
import time
import localization.detection
import localization.apriltag_solution
import localization.visiony
import localization.gamepiece_solution
import pipeline.ntables
from time import time_ns
from typing import List, Dict
import platform
import json
from threading import Thread, Lock
from camera.preprocess import PROCESS_FRAME
from util.logger import Logger

logger = Logger("VisionManager")

class CameraPipeline:
    def __init__(self, camera_config: dict):
        self.camera_id = camera_config["id"]
        self.camera_name = camera_config["name"]
        self.device_id = camera_config["device_id"]
        self.pipeline_type = camera_config["pipeline"]
        self.enabled = camera_config["enabled"]
        
        use_preprocessing = self.pipeline_type == "apriltag"
        
        if platform.system() == "Windows" or platform.system() == "Darwin":
            self.cam = CameraReader(self.device_id if self.camera_id == 0 else self.camera_id - 1, use_preprocessing)
        else:
            self.cam = CameraReader(self.device_id, use_preprocessing)
        
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_log_time = time.time()
        self.fps_log_count = 0
        self.framerate = 30.0
        self.backend_fps = 0.0
        self.processing_latency = 0.0
        self.frame: cv2.typing.MatLike = None
        self.detections: List = []
        self.frame_num = 0
        self.lock = Lock()
        self.running = True
        
        if self.pipeline_type == "apriltag":
            localization.apriltag_solution.SET_CAM(self.camera_id)
            self.ntables = pipeline.ntables.AprilTagNTables(self.camera_id)
        else:
            localization.gamepiece_solution.SET_CAM(self.camera_id)
            self.ntables = pipeline.ntables.GamePieceNTables(self.camera_id)
    
    def process_frame(self):
        if not self.enabled:
            time.sleep(0.05)
            return
        
        try:
            frame, timestamp = self.cam.get_frame()
            
            if frame is None:
                time.sleep(0.1)
                return
            
            self.frame_num += 1
            self.frame_num %= 500
            
            if self.pipeline_type == "apriltag":
                frame = PROCESS_FRAME(frame)
                corners, ids = localization.detection.DETECT_TAGS(frame)
                annotated_frame = localization.detection.ANNOTATE_TAGS(frame, corners, ids)
                detections = localization.apriltag_solution.CALCULATE_PARTIAL_SOLUTION(
                    self.camera_id, frame, corners, ids
                )
            else:
                annotated_frame, rawDets = localization.visiony.runPipeline(frame)
                detections = localization.gamepiece_solution.CALCULATE_PARTIAL_SOLUTION(self.camera_id, frame, rawDets)
            
            processing_latency = (time_ns() - timestamp) / 1e9
            
            with self.lock:
                self.frame = annotated_frame
                self.detections = detections
                self.processing_latency = processing_latency
            
            self.frame_count += 1
            self.fps_log_count += 1
            
            if self.frame_count % 30 == 0:
                end_time = time.time()
                self.framerate = 30 / (end_time - self.start_time)
                self.start_time = end_time
            
            if self.fps_log_count >= 100:
                current_time = time.time()
                elapsed = current_time - self.fps_log_time
                self.backend_fps = self.fps_log_count / elapsed
                logger.Log(f"Camera {self.camera_id} ({self.pipeline_type}): {self.backend_fps:.2f} FPS | Latency: {processing_latency*1000:.2f}ms | Detections: {len(detections)}")
                self.fps_log_time = current_time
                self.fps_log_count = 0
            
            if self.pipeline_type == "apriltag":
                self.ntables.updateFrameNum(self.frame_num)
            
            self.ntables.execute(detections, processing_latency)
            
        except Exception as e:
            logger.Warn(f"Error in camera {self.camera_id}: {e}")
    
    def get_frame(self):
        with self.lock:
            if self.frame is not None:
                return self.frame
            return None
    
    def get_detections(self):
        with self.lock:
            return list(self.detections)
    
    def get_framerate(self):
        return self.framerate
    
    def get_backend_fps(self):
        return self.backend_fps

    def get_processing_latency(self):
        return self.processing_latency
    
    def get_camera_id(self):
        return self.camera_id
    
    def get_camera_name(self):
        return self.camera_name
    
    def get_pipeline_type(self):
        return self.pipeline_type
    
    def set_enabled(self, enabled: bool):
        self.enabled = enabled
    
    def is_enabled(self):
        return self.enabled
    
    def stop(self):
        self.running = False
        self.cam.release()

class VisionManager:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        if "network_tables" in self.config and "server" in self.config["network_tables"]:
            server = self.config["network_tables"]["server"]
            pipeline.ntables.set_server(server)
            logger.Log(f"NetworkTables server set to: {server}")
        
        if "valid_apriltag_ids" in self.config:
            localization.apriltag_solution.SET_VALID_TAG_IDS(self.config["valid_apriltag_ids"])
        
        self.pipelines: Dict[int, CameraPipeline] = {}
        
        for camera_config in self.config["cameras"]:
            pipeline_instance = CameraPipeline(camera_config)
            self.pipelines[camera_config["id"]] = pipeline_instance
        
        self.threads: List[Thread] = []
    
    def execute(self):
        for camera_id, pipeline in self.pipelines.items():
            thread = Thread(target=self._run_pipeline, args=(pipeline,), daemon=True)
            thread.start()
            self.threads.append(thread)
        
        for thread in self.threads:
            thread.join()
    
    def _run_pipeline(self, pipeline: CameraPipeline):
        while pipeline.running:
            try:
                pipeline.process_frame()
            except Exception as e:
                print(f"Error in pipeline {pipeline.camera_id}: {e}")
                time.sleep(0.1)
    
    def get_pipeline(self, camera_id: int) -> CameraPipeline:
        return self.pipelines.get(camera_id)
    
    def get_all_pipelines(self) -> Dict[int, CameraPipeline]:
        return self.pipelines
    
    def update_camera_pipeline(self, camera_id: int, pipeline_type: str):
        if camera_id in self.pipelines:
            old_pipeline = self.pipelines[camera_id]
            old_pipeline.stop()
            
            camera_config = None
            for cam in self.config["cameras"]:
                if cam["id"] == camera_id:
                    cam["pipeline"] = pipeline_type
                    camera_config = cam
                    break
            
            if camera_config:
                new_pipeline = CameraPipeline(camera_config)
                self.pipelines[camera_id] = new_pipeline
                thread = Thread(target=self._run_pipeline, args=(new_pipeline,), daemon=True)
                thread.start()
                self.threads.append(thread)
    
    def toggle_camera(self, camera_id: int, enabled: bool):
        if camera_id in self.pipelines:
            self.pipelines[camera_id].set_enabled(enabled)

