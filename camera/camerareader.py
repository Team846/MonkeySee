import cv2
from cv2.typing import MatLike
from util.logger import Logger
from time import time_ns, sleep
from typing import Tuple, Optional
import platform
import os

logger = Logger("Camera")


class CameraReader:
    MAX_READ_RETRIES = 200
    REOPEN_SLEEP_S = 0.5

    def __init__(self, camera_id: str, use_preprocessing: bool = False):
        self.camera_id = camera_id
        self.use_preprocessing = use_preprocessing

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

    def _open_camera(self):
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
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
            self.cap.set(cv2.CAP_PROP_FPS, 120)
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            self.cap.set(cv2.CAP_PROP_FPS, 60)

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

    def get_frame(self) -> Tuple[Optional[MatLike], int]:
        if not self._opened:
            self._open_camera()
            self._opened = True
        if not self.cap or not self.cap.isOpened():
            logger.Warn("Camera not open, reopening...")
            self._open_camera()
            return None, time_ns()

        ret, frame = self.cap.read()

        if ret and frame is not None:
            self.fail_count = 0
            return frame, time_ns()

        self.fail_count += 1

        if self.fail_count % 20 == 0:
            logger.Warn(f"Camera read failed ({self.fail_count}x)")

        if self.fail_count >= self.MAX_READ_RETRIES:
            logger.Warn("Camera stalled, reopening...")
            self._open_camera()

        return None, time_ns()

    def get_raw_frame(self) -> Optional[MatLike]:
        frame, _ = self.get_frame()
        return frame

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
