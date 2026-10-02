import cv2
from cv2.typing import MatLike
from util.config import ConfigCategory
from util.logger import Logger
from time import sleep, monotonic, monotonic_ns
from typing import Tuple, Optional
from threading import Lock
import platform
import os
import shutil
import subprocess

logger = Logger("Camera")

# v4l2 auto_exposure menu values
V4L2_EXPOSURE_MANUAL = 1
V4L2_EXPOSURE_APERTURE_PRIORITY = 3

RESOLUTIONS = ((800, 600), (1280, 800))

class CameraReader:
    MAX_READ_RETRIES = 200
    REOPEN_SLEEP_S = 0.5
    STALE_FRAME_S = 0.5
    MAX_FRAME_AGE_NS = 1_000_000_000

    def __init__(self, camera_id: str, use_preprocessing: bool = False, settings_category: Optional[str] = None):
        self.camera_id = camera_id
        self.use_preprocessing = use_preprocessing

        # exposure is in v4l2 units of 0.1 ms
        settings = ConfigCategory(settings_category or f"Camera_{camera_id}")
        self._auto_exposure = settings.getIntConfig("auto_exposure", 1)
        self._exposure = settings.getIntConfig("exposure", 20)
        self._gain = settings.getIntConfig("gain", 50)
        self._width = settings.getIntConfig("width", 800)
        self._height = settings.getIntConfig("height", 600)
        self._exposure_status = "Camera not opened yet"

        if platform.system() in ["Windows", "Darwin"]:
            self.camera_path = ""
            print("using mac/win")
        else:
            self.camera_path = (
                f"/dev/v4l/by-id/"
                f"usb-Arducam_Technology_Co.__Ltd._{camera_id}_{camera_id}-video-index0"
            )

            print("camera path: ", self.camera_path)
            try:
                print("camera paths: ", os.listdir("/dev/v4l/by-id"))
            except FileNotFoundError:
                print("/dev/v4l/by-id not found")

        self.cap: Optional[cv2.VideoCapture] = None
        self.fail_count = 0
        self._opened = False

        self._lock = Lock()
        self._last_frame: Optional[MatLike] = None
        self._last_frame_at = 0.0

    def _open_camera_locked(self):
        logger.Log("Opening camera...")

        if self.cap:
            self.cap.release()
            sleep(self.REOPEN_SLEEP_S)

        if platform.system() in ["Windows", "Darwin"]:
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture(self.camera_path, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            logger.Warn("Failed to open camera")
            return

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self.use_preprocessing:
            width, height = self.get_resolution()
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, 120)
            if platform.system() == "Linux":
                self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            self.cap.set(cv2.CAP_PROP_FPS, 60)

        self._apply_exposure_locked()

        self.fail_count = 0
        try:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(self.cap.get(cv2.CAP_PROP_FPS))
            fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            backend = int(self.cap.get(cv2.CAP_PROP_BACKEND))
            logger.Log(f"Camera opened successfully: {w}x{h} @ {fps:.1f}fps fourcc={fourcc_str} backend={backend}")
        except Exception:
            logger.Log("Camera opened successfully")

    def _apply_exposure_locked(self) -> None:
        if platform.system() != "Linux":
            self._exposure_status = "Exposure control requires Linux (V4L2)"
            return
        if not self.cap or not self.cap.isOpened():
            return

        auto = self._auto_exposure.valueInt() != 0
        exposure = self._exposure.valueInt()
        gain = self._gain.valueInt()

        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, V4L2_EXPOSURE_APERTURE_PRIORITY if auto else V4L2_EXPOSURE_MANUAL)
        if not auto:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            self.cap.set(cv2.CAP_PROP_GAIN, gain)
            if self._read_exposure_locked()[1:] != (exposure, gain):
                self._apply_with_v4l2_ctl(exposure, gain)

        reported_auto, reported_exposure, reported_gain = self._read_exposure_locked()
        if auto:
            self._exposure_status = f"Auto exposure (camera reports auto_exposure={reported_auto})"
        elif (reported_exposure, reported_gain) == (exposure, gain):
            ms = exposure / 10.0
            mode_fps = 120 if self.use_preprocessing else 60
            fps_note = f" (too long for {mode_fps} FPS, limits to ~{1000.0 / ms:.0f})" if ms * mode_fps > 1000.0 else ""
            self._exposure_status = f"Manual: {ms:.1f} ms, gain {gain}{fps_note}"
        else:
            self._exposure_status = (
                f"Camera did not accept settings: requested exposure={exposure} gain={gain}, "
                f"camera reports exposure={reported_exposure} gain={reported_gain}"
            )
            logger.Warn(f"{self.camera_id}: {self._exposure_status}")
            return
        logger.Log(f"{self.camera_id}: {self._exposure_status}")

    def _read_exposure_locked(self) -> Tuple[int, int, int]:
        return (
            int(round(self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))),
            int(round(self.cap.get(cv2.CAP_PROP_EXPOSURE))),
            int(round(self.cap.get(cv2.CAP_PROP_GAIN))),
        )

    def _apply_with_v4l2_ctl(self, exposure: int, gain: int) -> None:
        if shutil.which("v4l2-ctl") is None:
            return
        for auto_name, exposure_name in (("auto_exposure", "exposure_time_absolute"), ("exposure_auto", "exposure_absolute")):
            controls = f"{auto_name}={V4L2_EXPOSURE_MANUAL},{exposure_name}={exposure},gain={gain}"
            try:
                result = subprocess.run(
                    ["v4l2-ctl", "-d", self.camera_path, "-c", controls],
                    capture_output=True, text=True, timeout=2,
                )
            except (OSError, subprocess.TimeoutExpired) as e:
                logger.Warn(f"v4l2-ctl failed: {e}")
                return
            if result.returncode == 0:
                return
        logger.Warn(f"v4l2-ctl failed: {result.stderr.strip()}")

    def set_exposure_settings(self, auto_exposure: bool, exposure: int, gain: int) -> None:
        self._auto_exposure.setInt(1 if auto_exposure else 0)
        self._exposure.setInt(exposure)
        self._gain.setInt(gain)
        with self._lock:
            self._apply_exposure_locked()

    def get_exposure_settings(self) -> dict:
        return {
            "auto_exposure": self._auto_exposure.valueInt() != 0,
            "exposure": self._exposure.valueInt(),
            "gain": self._gain.valueInt(),
            "status": self._exposure_status,
        }

    def get_resolution(self) -> Tuple[int, int]:
        return self._width.valueInt(), self._height.valueInt()

    def set_resolution(self, width: int, height: int) -> None:
        self._width.setInt(width)
        self._height.setInt(height)
        with self._lock:
            if self._opened:
                self._open_camera_locked()

    def _capture_time_ns_locked(self) -> int:
        now = monotonic_ns()
        if platform.system() != "Linux":
            return now
        captured = int(self.cap.get(cv2.CAP_PROP_POS_MSEC) * 1e6)
        if 0 <= now - captured < self.MAX_FRAME_AGE_NS:
            return captured
        return now

    def get_frame(self) -> Tuple[Optional[MatLike], int]:
        """Returns (frame, capture time in monotonic_ns())."""
        with self._lock:
            if not self._opened:
                self._open_camera_locked()
                self._opened = True
            if not self.cap or not self.cap.isOpened():
                logger.Warn("Camera not open, reopening...")
                self._open_camera_locked()
                return None, monotonic_ns()

            ret, frame = self.cap.read()
            if ret and frame is not None and frame.ndim == 2 and min(frame.shape) == 1:
                frame = cv2.imdecode(frame, cv2.IMREAD_GRAYSCALE)

            if ret and frame is not None:
                self.fail_count = 0
                self._last_frame = frame
                self._last_frame_at = monotonic()
                return frame, self._capture_time_ns_locked()

            self.fail_count += 1

            if self.fail_count % 20 == 0:
                logger.Warn(f"Camera read failed ({self.fail_count}x)")

            if self.fail_count >= self.MAX_READ_RETRIES:
                logger.Warn("Camera stalled, reopening...")
                self._open_camera_locked()

            return None, monotonic_ns()

    def get_raw_frame(self) -> Optional[MatLike]:
        with self._lock:
            if (
                self._last_frame is not None
                and (monotonic() - self._last_frame_at) < self.STALE_FRAME_S
            ):
                frame = self._last_frame.copy()
            else:
                frame = None

        if frame is None:
            frame, _ = self.get_frame()
        if frame is not None and frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame

    def release(self):
        with self._lock:
            if self.cap:
                self.cap.release()
                self.cap = None
            self._opened = False
            self._last_frame = None
