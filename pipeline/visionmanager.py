from camera.camerareader import CameraReader
import cv2
import time
import localization.detection
import localization.apriltag_solution
import pipeline.ntables
from time import monotonic_ns
from typing import List, Dict, Optional
import platform
import json
import os
import struct
from datetime import datetime
from threading import Thread, Lock
from camera.preprocess import PROCESS_FRAME
from util.logger import Logger

logger = Logger("VisionManager")

CAPTURES_DIR = "captures"


def _open_writer(path: str, size, is_color: bool, raw: bool):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    if raw:
        writer = cv2.VideoWriter(path, cv2.CAP_FFMPEG, fourcc, 30.0, size, [cv2.VIDEOWRITER_PROP_RAW_VIDEO, 1])
        if writer.isOpened():
            return writer, True
    return cv2.VideoWriter(path, fourcc, 30.0, size, isColor=is_color), False


def _set_avi_fps(path: str, fps: float) -> None:
    with open(path, "r+b") as f:
        head = f.read(8192)
        avih, strh = head.find(b"avih"), head.find(b"strh")
        if avih < 0 or strh < 0:
            return
        f.seek(avih + 8)
        f.write(struct.pack("<I", round(1e6 / fps)))
        f.seek(strh + 28)
        f.write(struct.pack("<II", 1000, round(fps * 1000)))


class CameraPipeline:
    def __init__(self, camera_config: dict):
        self.camera_id = camera_config["id"]
        self.camera_name = camera_config["name"]
        self.device_id = camera_config["device_id"]
        self.pipeline_type = camera_config["pipeline"]
        self.enabled = camera_config["enabled"]

        use_high_res = self.pipeline_type == "apriltag"

        settings_category = f"Camera{self.camera_id}"
        if platform.system() == "Windows" or platform.system() == "Darwin":
            self.cam = CameraReader(self.device_id if self.camera_id == 0 else self.camera_id - 1, use_high_res, settings_category)
        else:
            self.cam = CameraReader(self.device_id, use_high_res, settings_category)

        self.frame_count = 0
        self.start_time = time.time()
        self.fps_log_time = time.time()
        self.fps_log_count = 0
        self.framerate = 30.0
        self.backend_fps = 0.0
        self.processing_latency = 0.0
        self.frame: cv2.typing.MatLike = None
        self._tags = None
        self._overlay_src = None
        self._overlay = None
        self._preview = (None, 0, b"")
        self.detections: List = []
        self.frame_num = 0
        self.lock = Lock()
        self.running = True

        self.focus_mode = False
        self.focus_score = 0.0

        self._writer: Optional[cv2.VideoWriter] = None
        self._recording_path: Optional[str] = None
        self._recording_pending = False
        self._last_capture_message = ""

        if self.pipeline_type == "apriltag":
            apriltag_section = camera_config.get("apriltag_config", f"AprilTag{self.camera_id}")
            localization.apriltag_solution.SET_CAM(self.camera_id, apriltag_section)
            self._apriltag = localization.detection.MAKE_APRILTAG_DETECTOR(apriltag_section)
            self.ntables = pipeline.ntables.AprilTagNTables(self.camera_id)
        else:
            from localization import gamepiece_solution
            from localization import visiony
            gamepiece_solution.SET_CAM(self.camera_id)
            self.ntables = pipeline.ntables.GamePieceNTables(self.camera_id)
            self._visiony = visiony
            self._gamepiece_solution = gamepiece_solution

    def process_frame(self):
        if not self.enabled or not self.running:
            time.sleep(0.05)
            return

        try:
            if not self.running:
                return
            frame, capture_ns = self.cam.get_frame()
            cam_frame, jpeg = frame, self.cam.last_jpeg
            read_ns = monotonic_ns()

            if frame is None:
                time.sleep(0.1)
                return

            self.frame_num += 1
            self.frame_num %= 500

            if self.focus_mode:
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame
                focus_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                with self.lock:
                    self.focus_score = focus_score
                    self.frame = frame
                    self._tags = None
                time.sleep(0.03)
                return

            tags = None
            if self.pipeline_type == "apriltag":
                raw_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                if self._apriltag is not None:
                    frame = raw_gray
                    corners, ids = self._apriltag(raw_gray)
                else:
                    frame = PROCESS_FRAME(raw_gray)
                    corners, ids = localization.detection.DETECT_TAGS(frame, refine_image=raw_gray)
                if ids is not None:
                    self.cam.note_tags(raw_gray, corners)
                display_frame = frame
                tags = (corners, ids)
                detections = localization.apriltag_solution.CALCULATE_PARTIAL_SOLUTION(
                    self.camera_id, raw_gray, corners, ids
                )
            else:
                display_frame, rawDets = self._visiony.runPipeline(frame)
                detections = self._gamepiece_solution.CALCULATE_PARTIAL_SOLUTION(self.camera_id, frame, rawDets)

            done_ns = monotonic_ns()
            processing_latency = (done_ns - read_ns) / 1e9
            capture_latency = (done_ns - capture_ns) / 1e9

            if self.pipeline_type == "apriltag":
                self.ntables.updateFrameNum(self.frame_num)
            self.ntables.execute(detections, processing_latency, capture_ns)

            with self.lock:
                if self._recording_pending and self._writer is None:
                    self._start_writer_locked(cam_frame, jpeg is not None)
                if self._writer is not None:
                    data = jpeg if self._writer_raw else cam_frame
                    if data is not None:
                        self._writer.write(data)
                        self._record_frames += 1
                self.frame = display_frame
                self._tags = tags
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
                logger.Log(f"Camera {self.camera_id} ({self.pipeline_type}): {self.backend_fps:.2f} FPS | Latency: {processing_latency*1000:.2f}ms (since capture {capture_latency*1000:.2f}ms) | Detections: {len(detections)}")
                self.fps_log_time = current_time
                self.fps_log_count = 0

        except Exception as e:
            logger.Warn(f"Error in camera {self.camera_id}: {e}")

    def _capture_stamp(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    def _start_writer_locked(self, frame, raw: bool = False) -> None:
        height, width = frame.shape[:2]
        path = os.path.join(
            CAPTURES_DIR,
            f"cam{self.camera_id}_{self._capture_stamp()}.avi",
        )
        writer, self._writer_raw = _open_writer(path, (width, height), frame.ndim == 3, raw)
        if not writer.isOpened():
            self._last_capture_message = f"Failed to open video writer: {path}"
            logger.Warn(self._last_capture_message)
            self._recording_pending = False
            return
        self._writer = writer
        self._record_frames = 0
        self._record_start = time.monotonic()
        self._recording_path = path
        self._recording_pending = False
        self._last_capture_message = f"Recording: {path}"
        logger.Log(self._last_capture_message)

    def save_snapshot(self) -> Optional[str]:
        frame = self.get_frame()
        if frame is None:
            self._last_capture_message = "No frame available"
            return None
        os.makedirs(CAPTURES_DIR, exist_ok=True)
        path = os.path.join(
            CAPTURES_DIR,
            f"cam{self.camera_id}_{self._capture_stamp()}.png",
        )
        if not cv2.imwrite(path, frame):
            self._last_capture_message = f"Failed to save snapshot: {path}"
            logger.Warn(self._last_capture_message)
            return None
        self._last_capture_message = f"Saved snapshot: {path}"
        logger.Log(self._last_capture_message)
        return path

    def start_recording(self) -> str:
        with self.lock:
            if self._writer is not None or self._recording_pending:
                return self._last_capture_message or "Already recording"
            os.makedirs(CAPTURES_DIR, exist_ok=True)
            self._recording_pending = True
            self._last_capture_message = "Recording starting..."
            return self._last_capture_message

    def stop_recording(self) -> str:
        with self.lock:
            if self._writer is None and not self._recording_pending:
                self._last_capture_message = "Not recording"
                return self._last_capture_message
            path = self._recording_path
            if self._writer is not None:
                self._writer.release()
                seconds = time.monotonic() - self._record_start
                fps = self._record_frames / seconds if seconds > 0 else 0.0
                if fps > 0:
                    _set_avi_fps(path, fps)
                path = f"{path} ({self._record_frames} frames, {seconds:.0f} s, {fps:.1f} fps)"
            self._writer = None
            self._recording_path = None
            self._recording_pending = False
            self._last_capture_message = f"Saved recording: {path}" if path else "Recording cancelled"
            logger.Log(self._last_capture_message)
            return self._last_capture_message

    def is_recording(self) -> bool:
        with self.lock:
            return self._writer is not None or self._recording_pending

    def get_capture_status(self) -> dict:
        with self.lock:
            return {
                "recording": self._writer is not None or self._recording_pending,
                "path": self._recording_path,
                "message": self._last_capture_message,
            }

    def get_frame(self):
        with self.lock:
            if self.frame is None or self._tags is None:
                return self.frame
            if self._overlay_src is not self.frame:
                self._overlay = localization.detection.ANNOTATE_TAGS(self.frame, *self._tags)
                self._overlay_src = self.frame
            return self._overlay

    def get_preview_jpeg(self, quality: int, max_width: int):
        with self.lock:
            frame, tags = self.frame, self._tags
        if frame is None:
            return None, None
        source, cached_quality, jpeg = self._preview
        if source is frame and cached_quality == quality:
            return frame, jpeg
        scale = min(1.0, max_width / frame.shape[1])
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
        if tags is not None:
            corners, ids = tags
            small = localization.detection.ANNOTATE_TAGS(small, [c * scale for c in corners], ids)
        jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
        self._preview = (frame, quality, jpeg)
        return frame, jpeg

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
        try:
            with self.lock:
                if self._writer is not None:
                    self._writer.release()
                    self._writer = None
                    self._recording_pending = False
                if self.cam:
                    self.cam.release()
        except Exception as e:
            logger.Warn(f"Error releasing camera {self.camera_id}: {e}")

class VisionManager:
    def __init__(self, config_path: str):
        self.config_path = config_path
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

    def reload_all_pipelines(self):
        try:
            logger.Log("Reloading all pipelines...")

            for cam_pipeline in list(self.pipelines.values()):
                try:
                    cam_pipeline.running = False
                except:
                    pass

            time.sleep(1.5)

            for cam_pipeline in list(self.pipelines.values()):
                try:
                    if hasattr(cam_pipeline, 'cam') and cam_pipeline.cam:
                        cam_pipeline.cam.release()
                except:
                    pass

            self.threads = []

            import platform
            if platform.system() == "Windows":
                time.sleep(1.5)
            else:
                time.sleep(0.5)

            with open(self.config_path, 'r') as f:
                self.config = json.load(f)

            try:
                pipeline.ntables.reset_instance()
            except:
                pass

            if "network_tables" in self.config and "server" in self.config["network_tables"]:
                try:
                    pipeline.ntables.set_server(self.config["network_tables"]["server"])
                except:
                    pass

            if "valid_apriltag_ids" in self.config:
                try:
                    localization.apriltag_solution.SET_VALID_TAG_IDS(self.config["valid_apriltag_ids"])
                except:
                    pass

            self.pipelines = {}
            for camera_config in self.config["cameras"]:
                try:
                    self.pipelines[camera_config["id"]] = CameraPipeline(camera_config)
                except Exception as e:
                    logger.Warn(f"Error creating pipeline for camera {camera_config.get('id', 'unknown')}: {e}")

            for cam_pipeline in self.pipelines.values():
                try:
                    thread = Thread(target=self._run_pipeline, args=(cam_pipeline,), daemon=True)
                    thread.start()
                    self.threads.append(thread)
                except:
                    pass

            logger.Log("All pipelines reloaded successfully")

        except Exception as e:
            logger.Warn(f"Error during pipeline reload: {e}")
            import traceback
            logger.Warn(traceback.format_exc())


if __name__ == "__main__":
    import tempfile
    import numpy as np
    gray = np.random.randint(0, 255, (800, 1280), np.uint8)
    for raw in (True, False):
        path = os.path.join(tempfile.mkdtemp(), "check.avi")
        writer, used_raw = _open_writer(path, (1280, 800), False, raw)
        assert used_raw == raw, f"raw JPEG passthrough unavailable in OpenCV {cv2.__version__}"
        for _ in range(10):
            writer.write(cv2.imencode(".jpg", gray)[1].reshape(1, -1) if raw else gray)
        writer.release()
        _set_avi_fps(path, 57.5)
        cap = cv2.VideoCapture(path)
        frames, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        assert frames == 10 and abs(fps - 57.5) < 0.01, (raw, frames, fps)
    print("recorder ok")
