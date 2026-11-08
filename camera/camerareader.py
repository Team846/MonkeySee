import cv2
import numpy as np
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

RESOLUTIONS = ((800, 600), (1280, 800))
GAIN_MAX = 255
AUTO_GAIN_EVERY = 10
AUTO_GAIN_STUCK_STEPS = 40
AUTO_GAIN_WINDOW = (0.5, 1.7)
AUTO_GAIN_RETUNE_STEPS = 30
TAG_WHITE_CLIPPED = 250

def tag_white(frame: MatLike, corners) -> Optional[float]:
    whites = []
    for quad in corners:
        quad = np.asarray(quad, np.float32).reshape(4, 2)
        x, y, w, h = cv2.boundingRect(quad)
        x, y = max(x, 0), max(y, 0)
        crop = frame[y:y + h, x:x + w]
        mask = np.zeros(crop.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, np.round(quad - (x, y)).astype(np.int32), 1)
        pixels = crop[mask > 0]
        if pixels.size >= 16:
            whites.append(np.percentile(pixels, 90))
    return float(np.median(whites)) if whites else None

def next_gain(gain: int, mean: float, target: float) -> Tuple[int, int]:
    ratio = target / max(mean, 1.0)
    if 0.9 < ratio < 1.1:
        return gain, 0
    push = 1 if ratio > 1 else -1
    new = round(gain * min(max(ratio, 0.8), 1.25))
    if new == gain:
        new += push
    return min(max(new, 0), GAIN_MAX), push

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
        self._auto_gain = settings.getIntConfig("auto_exposure", 1)
        self._exposure = settings.getIntConfig("exposure", 20)
        self._gain = settings.getIntConfig("gain", 50)
        self._gain_target = settings.getIntConfig("gain_target", 100)
        self._tag_white_target = settings.getIntConfig("tag_white_target", 190)
        self._width = settings.getIntConfig("width", 800)
        self._height = settings.getIntConfig("height", 600)
        self._exposure_status = "Camera not opened yet"
        self._live_gain = self._gain.valueInt()
        self._frames_since_gain = 0
        self._gain_push = 0
        self._gain_push_steps = 0
        self._gain_push_start = self._live_gain
        self._gain_stuck = False
        self._gain_locked = True
        self._gain_outside = 0
        self._gain_save_on_lock = False
        self._gain_source = "frame average"
        self._tag_frame: Optional[MatLike] = None
        self._tag_corners = None
        self._frames_since_tags = AUTO_GAIN_EVERY + 1

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

        auto = self._auto_gain.valueInt() != 0
        exposure = self._exposure.valueInt()
        gain = self._live_gain if auto else self._gain.valueInt()
        self._live_gain = gain

        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, V4L2_EXPOSURE_MANUAL)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        self.cap.set(cv2.CAP_PROP_GAIN, gain)
        if self._read_exposure_locked()[1:] != (exposure, gain):
            self._apply_with_v4l2_ctl(exposure, gain)

        _, reported_exposure, reported_gain = self._read_exposure_locked()
        if (reported_exposure, reported_gain) == (exposure, gain):
            ms = exposure / 10.0
            mode_fps = 120 if self.use_preprocessing else 60
            fps_note = f" (too long for {mode_fps} FPS, limits to ~{1000.0 / ms:.0f})" if ms * mode_fps > 1000.0 else ""
            mode = "Auto gain" if auto else "Manual"
            self._exposure_status = f"{mode}: {ms:.1f} ms, gain {gain}{fps_note}"
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

    def _auto_gain_locked(self, frame: MatLike) -> None:
        if not self._auto_gain.valueInt():
            return
        self._frames_since_tags += 1
        self._frames_since_gain += 1
        if self._frames_since_gain < AUTO_GAIN_EVERY:
            return
        self._frames_since_gain = 0
        white = tag_white(self._tag_frame, self._tag_corners) if self._frames_since_tags <= AUTO_GAIN_EVERY else None
        if white is not None:
            measured, target, source = white, self._tag_white_target.valueInt(), "tag white"
        else:
            measured, target, source = float(frame[::8, ::8].mean()), self._gain_target.valueInt(), "frame average"
        if self._gain_locked:
            if white is None:
                return
            low, high = AUTO_GAIN_WINDOW[0] * target, min(AUTO_GAIN_WINDOW[1] * target, TAG_WHITE_CLIPPED)
            self._gain_outside = 0 if low <= measured < high else self._gain_outside + 1
            if self._gain_outside < AUTO_GAIN_RETUNE_STEPS:
                return
            self._gain_locked = False
            self._gain_save_on_lock = False
            logger.Log(f"{self.camera_id}: tag white {measured:.0f} stayed outside {low:.0f}-{high:.0f}, re-tuning gain until restart")
        gain, push = next_gain(self._live_gain, measured, target)
        if push == 0 or push != self._gain_push:
            self._gain_push_steps = 0
            self._gain_push_start = self._live_gain
        else:
            self._gain_push_steps += 1
        self._gain_push = push

        if push == 0:
            self._gain_locked = True
            self._gain_outside = 0
            self._gain_stuck = False
            if self._gain_save_on_lock:
                self._gain.setInt(self._live_gain)
            saved = "saved" if self._gain_save_on_lock else "not saved"
            self._gain_save_on_lock = False
            self._gain_source = source
            logger.Log(f"{self.camera_id}: auto gain locked at gain {self._live_gain} on {source} {measured:.0f}, {saved}")
        elif gain != self._live_gain:
            self.cap.set(cv2.CAP_PROP_GAIN, gain)
            self._live_gain = int(round(self.cap.get(cv2.CAP_PROP_GAIN)))

        if self._gain_push_steps >= AUTO_GAIN_STUCK_STEPS and not self._gain_stuck:
            self._gain_stuck = True
            if self._live_gain == self._gain_push_start:
                why = f"camera stayed at gain {self._live_gain} when asked for {gain}, so it may not accept gain changes"
            else:
                why = f"gain went {self._gain_push_start} -> {self._live_gain} and can't go further (asked for {gain}), so change the exposure slider"
            self._exposure_status = f"Auto gain can't reach {source} {target} (now {measured:.0f}): {why}"
            logger.Warn(f"{self.camera_id}: {self._exposure_status}")
        elif not self._gain_stuck:
            state = f"locked on {self._gain_source}" if self._gain_locked else f"tuning on {source}"
            self._exposure_status = f"Auto gain: {self._exposure.valueInt() / 10.0:.1f} ms, gain {self._live_gain}, {state}"

    def note_tags(self, frame: MatLike, corners) -> None:
        with self._lock:
            self._tag_frame, self._tag_corners, self._frames_since_tags = frame, corners, 0

    def retune_gain(self) -> None:
        with self._lock:
            self._gain_locked = False
            self._gain_push_steps = 0
            self._gain_stuck = False
            self._gain_save_on_lock = True

    def wait_for_gain(self, timeout_s: float = 10.0) -> None:
        deadline = monotonic() + timeout_s
        while (
            platform.system() == "Linux"
            and self._auto_gain.valueInt()
            and not (self._gain_locked or self._gain_stuck)
            and monotonic() < deadline
        ):
            sleep(0.2)

    def set_exposure_settings(self, auto_exposure: bool, exposure: int, gain: int) -> None:
        self._auto_gain.setInt(1 if auto_exposure else 0)
        self._exposure.setInt(exposure)
        self._gain.setInt(gain)
        with self._lock:
            self._apply_exposure_locked()
        self.retune_gain()

    def get_exposure_settings(self) -> dict:
        return {
            "auto_exposure": self._auto_gain.valueInt() != 0,
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
                captured = self._capture_time_ns_locked()
                if platform.system() == "Linux":
                    self._auto_gain_locked(frame)
                return frame, captured

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

if __name__ == "__main__":
    assert next_gain(50, 95, 100) == (50, 0)
    assert next_gain(50, 25, 100) == (62, 1)
    assert next_gain(50, 400, 100) == (40, -1)
    assert next_gain(GAIN_MAX, 10, 100) == (GAIN_MAX, 1)
    assert next_gain(0, 300, 100) == (0, -1)
    assert next_gain(0, 10, 100) == (1, 1)

    QUAD = np.float32([[20, 20], [39, 20], [39, 39], [20, 39]])
    still = np.full((60, 80), 40, np.uint8)
    still[20:40, 20:40] = 200
    assert abs(tag_white(still, [QUAD]) - 200) < 1 and tag_white(still, []) is None

    class _Setting:
        def __init__(self, value):
            self.value = value
        def valueInt(self):
            return self.value
        def setInt(self, value):
            self.value = value

    class _FakeCap:
        def __init__(self, accepts, limit):
            self.gain, self.accepts, self.limit = 50, accepts, limit
        def set(self, prop, value):
            if self.accepts:
                self.gain = min(int(value), self.limit)
        def get(self, prop):
            return self.gain

    def fake_camera(accepts=True, limit=GAIN_MAX):
        cam = CameraReader.__new__(CameraReader)
        cam.camera_id, cam.cap, cam._lock = "test", _FakeCap(accepts, limit), Lock()
        cam._auto_gain, cam._exposure, cam._gain = _Setting(1), _Setting(15), _Setting(50)
        cam._gain_target, cam._tag_white_target = _Setting(100), _Setting(190)
        cam._live_gain, cam._frames_since_gain, cam._gain_push, cam._gain_push_steps, cam._gain_push_start = 50, 0, 0, 0, 50
        cam._gain_stuck, cam._gain_locked, cam._gain_outside, cam._gain_save_on_lock = False, False, 0, True
        cam._tag_frame, cam._tag_corners, cam._frames_since_tags = None, None, AUTO_GAIN_EVERY + 1
        cam._gain_source, cam._exposure_status = "frame average", ""
        return cam

    def run(cam, background, white=None, seconds=10):
        for _ in range(60 * seconds):
            g = cam.cap.gain
            frame = np.full((60, 80), min(background * g + 3, 255), np.uint8)
            if white is not None:
                frame[20:40, 20:40] = min(white * g + 3, 255)
                cam.note_tags(frame, [QUAD])
            cam._auto_gain_locked(frame)
        return cam

    for background in (0.5, 1.0, 2.0, 8.0):
        cam = run(fake_camera(), background)
        assert cam._gain_locked and abs(background * cam.cap.gain + 3 - 100) <= 12, (background, cam.cap.gain)
        assert cam._exposure_status.endswith("locked on frame average"), cam._exposure_status

    cam = run(fake_camera(), 0.1, white=1.0)
    assert cam._gain_locked and 170 <= cam.cap.gain + 3 <= 210, cam.cap.gain
    assert cam._exposure_status.endswith("locked on tag white"), cam._exposure_status

    cam = run(fake_camera(), 0.3, white=1.0)
    locked_gain = cam.cap.gain
    assert cam._gain.value == locked_gain
    run(cam, 0.3, white=3.0, seconds=2)
    assert cam._gain_locked and cam.cap.gain == locked_gain, cam.cap.gain
    run(cam, 0.05)
    assert cam._gain_locked and cam.cap.gain == locked_gain, cam.cap.gain
    run(cam, 0.3, white=3.0)
    assert cam._gain_locked and cam.cap.gain < locked_gain / 2 and cam._gain.value == locked_gain, (locked_gain, cam.cap.gain)

    cam.retune_gain()
    run(cam, 0.3, white=1.5)
    assert cam._gain_locked and cam._gain.value == cam.cap.gain and abs(1.5 * cam.cap.gain + 3 - 190) <= 20, cam.cap.gain

    cam = run(fake_camera(accepts=False), 0.5)
    assert cam._gain_stuck and "may not accept" in cam._exposure_status, cam._exposure_status

    cam = run(fake_camera(limit=100), 0.5)
    assert cam._gain_stuck and "can't go further" in cam._exposure_status, cam._exposure_status
    run(cam, 2.0)
    assert not cam._gain_stuck and cam._exposure_status.endswith("locked on frame average"), cam._exposure_status
    print("auto gain ok")
