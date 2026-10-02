import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
from threading import Lock
from util.config import ConfigCategory
from util.logger import Logger
from localization.undistort import store_calibration
import os
import time

logger = Logger("CheckerboardCalibrator")

_cal_prefs = ConfigCategory("Calibration")

fix_principal_point = _cal_prefs.getIntConfig("fix_principal_point", 0)
zero_tangent_dist = _cal_prefs.getIntConfig("zero_tangent_dist", 0)
save_samples = _cal_prefs.getIntConfig("save_samples", 0)

SAMPLES_DIR = os.path.join("captures", "calibration")

class CheckerboardCalibrator:
    INNER_COLS = 10     
    INNER_ROWS = 7    
    SQUARE_SIZE = 0.0212  

    MIN_SAMPLES = 15
    TARGET_SAMPLES = 54

    def __init__(self, camera_id: int, resolution: Tuple[int, int] = (800, 600)):
        self.camera_id = camera_id
        self.resolution = resolution
        self.lock = Lock()

        self.pattern_size = (self.INNER_COLS, self.INNER_ROWS)
        self._object_points = self._make_object_points()

        self.all_corners: List[np.ndarray] = []
        self.sample_dir: Optional[str] = None
        self.image_size: Optional[Tuple[int, int]] = None
        self.is_calibrating = False
        self.calibration_result: Optional[Dict] = None
        self.last_frame_annotated: Optional[np.ndarray] = None
        self.status_message = "Ready to calibrate"
        self.progress = 0.0

        self.auto_capture_enabled = True
        self.last_capture_time = 0.0
        self.capture_cooldown = 0.3
        self.last_good_frame: Optional[np.ndarray] = None

        self.coverage_grid = np.zeros((3, 3), dtype=int)
        self.max_samples_per_cell = 6

        self.min_corners_live = 20
        self.min_corners_capture = 30

        self.blur_laplacian_var_min = 50.0
        self.min_board_bbox_area_frac = 0.03
        self.min_board_span_frac = 0.15
        self.min_quadrants_covered = 1

    def _make_object_points(self) -> np.ndarray:
        objp = np.zeros((self.INNER_COLS * self.INNER_ROWS, 3), dtype=np.float32)
        objp[:, :2] = np.mgrid[0 : self.INNER_COLS, 0 : self.INNER_ROWS].T.reshape(-1, 2)
        objp *= self.SQUARE_SIZE
        return objp

    def generate_board_image(self, width: int = 1000, height: int = 1000) -> np.ndarray:
        cols = self.INNER_COLS + 1
        rows = self.INNER_ROWS + 1
        square_px = min(width // cols, height // rows)
        w, h = cols * square_px, rows * square_px
        board = np.ones((h, w), dtype=np.uint8) * 255
        for i in range(rows):
            for j in range(cols):
                if (i + j) % 2 == 0:
                    x1, y1 = j * square_px, i * square_px
                    x2, y2 = (j + 1) * square_px, (i + 1) * square_px
                    board[y1:y2, x1:x2] = 0
        if len(board.shape) == 2:
            board = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
        return board

    def _get_board_position(self, corners: np.ndarray, frame_shape: Tuple[int, int]) -> Tuple[Optional[int], Optional[int]]:
        if corners is None or len(corners) == 0:
            return None, None
        pts = corners.reshape(-1, 2)
        center_x = np.mean(pts[:, 0])
        center_y = np.mean(pts[:, 1])
        h, w = frame_shape[:2]
        grid_col = max(0, min(2, int(center_x / w * 3)))
        grid_row = max(0, min(2, int(center_y / h * 3)))
        return grid_row, grid_col

    def _draw_coverage_grid(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        cell_h, cell_w = h // 3, w // 3
        overlay = frame.copy()
        for row in range(3):
            for col in range(3):
                count = self.coverage_grid[row, col]
                x1, y1 = col * cell_w, row * cell_h
                x2, y2 = (col + 1) * cell_w, (row + 1) * cell_h
                if count == 0:
                    color, alpha = (0, 0, 200), 0.2
                elif count < self.max_samples_per_cell:
                    color, alpha = (0, 200, 200), 0.15
                else:
                    color, alpha = (0, 200, 0), 0.1
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (200, 200, 200), 2)
                text = str(count)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                cx, cy = x1 + (cell_w - tw) // 2, y1 + (cell_h + th) // 2
                cv2.putText(overlay, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    def _laplacian_variance(self, gray: np.ndarray) -> float:
        try:
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception:
            return 0.0

    def _is_sample_acceptable(
        self,
        gray: np.ndarray,
        corners: np.ndarray,
        frame_shape: Tuple[int, int],
        required_min_points: int,
    ) -> Tuple[bool, str]:
        if corners is None or corners.size < required_min_points * 2:
            return False, f"Need {required_min_points}+ corners"
        lap_var = self._laplacian_variance(gray)
        if lap_var < self.blur_laplacian_var_min:
            return False, f"Too blurry ({lap_var:.0f})"
        h, w = frame_shape[:2]
        pts = corners.reshape(-1, 2)
        min_xy = np.min(pts, axis=0)
        max_xy = np.max(pts, axis=0)
        bw = float(max_xy[0] - min_xy[0])
        bh = float(max_xy[1] - min_xy[1])
        if bw <= 1.0 or bh <= 1.0:
            return False, "Bad geometry"
        area_frac = (bw * bh) / float(w * h)
        if area_frac < self.min_board_bbox_area_frac:
            return False, "Board too small"
        if max(bw / w, bh / h) < self.min_board_span_frac:
            return False, "Board too small"
        cx, cy = w * 0.5, h * 0.5
        q = set()
        for x, y in pts:
            q.add((0 if x < cx else 1, 0 if y < cy else 1))
        if len(q) < self.min_quadrants_covered:
            return False, "Move board around"
        return True, "OK"

    def find_corners(self, frame: np.ndarray, downsample: bool = True) -> Tuple[bool, Optional[np.ndarray], np.ndarray]:
        if downsample and frame.shape[0] > 480:
            scale = 480.0 / frame.shape[0]
            frame_detect = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            frame_detect = frame

        gray_detect = cv2.cvtColor(frame_detect, cv2.COLOR_BGR2GRAY)
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray_detect, self.pattern_size, None)
        if ret and corners is not None and len(corners) > 0:
            scale = frame.shape[1] / frame_detect.shape[1]
            corners = corners * scale
            corners = cv2.cornerSubPix(
                gray_full, corners, (5, 5), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
        return ret, corners, gray_full

    def process_frame(
        self, frame: np.ndarray, downsample: bool = True, auto_capture: bool = True
    ) -> Tuple[Optional[np.ndarray], bool, str]:
        if frame is None:
            logger.Warn("process_frame called with None frame")
            return None, False, "No frame"
        if self.is_calibrating:
            return frame, False, self.status_message

        ret, corners, gray_full = self.find_corners(frame, downsample)

        n_corners = len(corners) if corners is not None else 0
        frame_count = len(self.all_corners)
        if frame_count % 5 == 0 and n_corners > 0:
            logger.Log(f"Checkerboard corners: {n_corners}")

        annotated = frame.copy()
        annotated = self._draw_coverage_grid(annotated)
        cv2.putText(
            annotated, f"Frame: {frame.shape[1]}x{frame.shape[0]}",
            (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

        is_good_sample = False
        should_capture = False
        status = ""

        if ret and corners is not None and n_corners >= self.min_corners_live:
            cv2.drawChessboardCorners(annotated, self.pattern_size, corners, ret)
            cv2.putText(annotated, f"Corners: {n_corners}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            is_good_sample = True
            ok, reason = self._is_sample_acceptable(
                gray_full, corners, frame.shape, self.min_corners_live
            )
            if not ok:
                is_good_sample = False
                status = reason

            grid_row, grid_col = self._get_board_position(corners, frame.shape)
            if is_good_sample and grid_row is not None:
                samples_in_cell = self.coverage_grid[grid_row, grid_col]
                if samples_in_cell < self.max_samples_per_cell:
                    should_capture = True
                    status = f"✓ Good sample! ({n_corners} corners) - Position [{grid_row},{grid_col}]"
                    cv2.putText(annotated, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    should_capture = False
                    status = f"Move to different area (position [{grid_row},{grid_col}] full)"
                    cv2.putText(annotated, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            elif is_good_sample:
                status = "✓ Good sample!"
                cv2.putText(annotated, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                status = status or f"Need {self.min_corners_live}+ corners / sharper"
                cv2.putText(annotated, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            status = "No checkerboard detected"
            cv2.putText(annotated, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(annotated, f"Pattern: {self.INNER_COLS}x{self.INNER_ROWS} inner corners",
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.putText(annotated, f"Samples: {len(self.all_corners)}/{self.TARGET_SAMPLES}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        with self.lock:
            self.last_frame_annotated = annotated
            if self.is_calibrating or self.calibration_result is not None:
                return annotated, is_good_sample, status
            self.status_message = status
            self.progress = len(self.all_corners) / self.TARGET_SAMPLES
            if (
                auto_capture
                and self.auto_capture_enabled
                and should_capture
                and len(self.all_corners) < self.TARGET_SAMPLES
            ):
                current_time = time.time()
                if current_time - self.last_capture_time >= self.capture_cooldown:
                    if corners is not None and len(corners) >= self.min_corners_capture:
                        self.last_good_frame = (frame.copy(), corners.copy())
                        logger.Log("Queued frame for capture")
                else:
                    logger.Log("Waiting for cooldown")

        return annotated, is_good_sample, status

    def try_auto_capture(self) -> bool:
        with self.lock:
            if self.last_good_frame is None:
                return False
            frame, corners = self.last_good_frame
            self.last_good_frame = None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        good = corners is not None and len(corners) >= self.min_corners_capture
        if good:
            ok, _ = self._is_sample_acceptable(gray, corners, frame.shape, self.min_corners_capture)
            good = ok

        if good:
            if save_samples.valueInt():
                self._save_sample(gray, len(self.all_corners) + 1)
            with self.lock:
                self.all_corners.append(corners)
                if self.image_size is None:
                    self.image_size = gray.shape[::-1]
                grid_row, grid_col = self._get_board_position(corners, frame.shape)
                if grid_row is not None:
                    self.coverage_grid[grid_row, grid_col] += 1
                self.last_capture_time = time.time()
                logger.Log(f"Added sample {len(self.all_corners)}/{self.TARGET_SAMPLES}")
            return True
        return False

    def _save_sample(self, gray: np.ndarray, index: int) -> None:
        if self.sample_dir is None:
            h, w = gray.shape[:2]
            self.sample_dir = os.path.join(SAMPLES_DIR, f"cam{self.camera_id}_{w}x{h}_{time.strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(self.sample_dir, exist_ok=True)
        path = os.path.join(self.sample_dir, f"{index:03d}.png")
        if not cv2.imwrite(path, gray):
            logger.Warn(f"Failed to save calibration sample {path}")

    def calibrate(self) -> Optional[Dict]:
        with self.lock:
            if len(self.all_corners) < self.MIN_SAMPLES:
                logger.Warn(f"Not enough samples: {len(self.all_corners)}/{self.MIN_SAMPLES}")
                self.status_message = f"Need {self.MIN_SAMPLES - len(self.all_corners)} more samples"
                return None
            self.is_calibrating = True
            self.status_message = "Preparing data... (0%)"
            self.progress = 0.0
            samples = list(self.all_corners)
            logger.Log(f"Starting calibration with {len(samples)} samples...")

        try:
            start_time = time.time()
            width, height = self.image_size

            cal_flags = 0
            if hasattr(cv2, "CALIB_USE_INTRINSIC_GUESS"):
                cal_flags |= cv2.CALIB_USE_INTRINSIC_GUESS
            if fix_principal_point.valueInt() and hasattr(cv2, "CALIB_FIX_PRINCIPAL_POINT"):
                cal_flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
                logger.Warn("Principal point pinned to image centre (Calibration.fix_principal_point=1)")
            if zero_tangent_dist.valueInt() and hasattr(cv2, "CALIB_ZERO_TANGENT_DIST"):
                cal_flags |= cv2.CALIB_ZERO_TANGENT_DIST
                logger.Warn("Tangential distortion forced to zero (Calibration.zero_tangent_dist=1)")

            cx_init = width / 2.0
            cy_init = height / 2.0
            camera_matrix_init = np.array([
                [width, 0, cx_init],
                [0, width, cy_init],
                [0, 0, 1],
            ], dtype=np.float64)
            dist_coeffs_init = np.zeros(5, dtype=np.float64)

            with self.lock:
                self.progress = 0.10
                self.status_message = "Running calibration... (10%)"

            obj_points = [self._object_points] * len(samples)
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                obj_points,
                samples,
                (width, height),
                camera_matrix_init,
                dist_coeffs_init,
                flags=cal_flags,
            )
            mean_error = float(ret)
            if camera_matrix is None or np.any(np.isnan(camera_matrix)):
                raise ValueError("calibrateCamera returned invalid camera matrix")

            with self.lock:
                self.progress = 0.90
                self.status_message = "Finalizing... (90%)"

            elapsed = time.time() - start_time
            fx = camera_matrix[0, 0]
            fy = camera_matrix[1, 1]
            cx = camera_matrix[0, 2]
            cy = camera_matrix[1, 2]

            fov_x_deg = 2 * np.arctan(width / (2 * fx)) * 180 / np.pi
            fov_y_deg = 2 * np.arctan(height / (2 * fy)) * 180 / np.pi
            tan_half_x = width / (2 * fx)
            tan_half_y = height / (2 * fy)
            fov_diag = 2 * np.arctan(np.sqrt(tan_half_x**2 + tan_half_y**2)) * 180 / np.pi

            logger.Log(f"Calibration completed in {elapsed:.2f}s")
            logger.Log(f"Checkerboard {self.INNER_COLS}x{self.INNER_ROWS}, square {self.SQUARE_SIZE*1000:.1f}mm")
            logger.Log(f"Camera matrix: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
            logger.Log(f"FOV: H={fov_x_deg:.1f}°, V={fov_y_deg:.1f}°, D={fov_diag:.1f}°")
            logger.Log(f"Reprojection error: {mean_error:.4f} px")

            dcx = cx - width / 2.0
            dcy = cy - height / 2.0
            logger.Log(
                f"Principal point offset from centre: {dcx:+.1f}, {dcy:+.1f} px "
                f"({np.degrees(np.arctan(dcx / fx)):+.2f}deg, {np.degrees(np.arctan(dcy / fy)):+.2f}deg)"
            )
            if abs(dcx) > 0.10 * width or abs(dcy) > 0.10 * height:
                logger.Warn(
                    "Principal point is far from the image centre. Calibrate with the fixed model"
                )
            if fov_x_deg > 120 or fov_y_deg > 120:
                logger.Warn("FOV implausibly wide - check INNER_COLS/INNER_ROWS and SQUARE_SIZE")
            if fov_x_deg < 30 or fov_y_deg < 30:
                logger.Warn("FOV implausibly narrow - check INNER_COLS/INNER_ROWS and SQUARE_SIZE")

            calibration_dict = {
                "meta": {
                    "resolution": {"width": width, "height": height},
                    "camera_matrix": camera_matrix.tolist(),
                    "dist_coeffs": dist_coeffs.tolist(),
                    "reprojection_error": float(mean_error),
                    "num_samples": len(samples),
                    "calibration_time": elapsed,
                    "fov": {
                        "horizontal": float(fov_x_deg),
                        "vertical": float(fov_y_deg),
                        "diagonal": float(fov_diag),
                    },
                }
            }
            distortion_viz = self._generate_distortion_visualization(
                camera_matrix, dist_coeffs, self.image_size
            )
            calibration_dict["meta"]["distortion_viz_base64"] = distortion_viz

            with self.lock:
                self.calibration_result = calibration_dict
                self.is_calibrating = False
                self.progress = 1.0
                self.status_message = f"✓ Calibration complete! Error: {mean_error:.4f} px. Scroll down and press Save Calibration"
            return calibration_dict

        except Exception as e:
            logger.Error(f"Calibration error: {e}")
            with self.lock:
                self.is_calibrating = False
                self.progress = 0.0
                self.status_message = f"Error: {str(e)}"
            return None

    def save_calibration(self) -> bool:
        if self.calibration_result is None:
            logger.Error("No calibration result to save")
            return False
        try:
            store_calibration(self.camera_id, self.calibration_result)
            res = self.calibration_result["meta"]["resolution"]
            with self.lock:
                self.status_message = f"✓ Saved {res['width']}x{res['height']} calibration"
            logger.Log(f"Calibration saved for camera {self.camera_id}")
            return True
        except Exception as e:
            logger.Error(f"Failed to save calibration: {e}")
            return False

    def reset(self) -> None:
        with self.lock:
            num = len(self.all_corners)
            self.all_corners.clear()
            self.sample_dir = None
            self.image_size = None
            self.calibration_result = None
            self.status_message = "Ready to calibrate"
            self.progress = 0.0
            self.coverage_grid = np.zeros((3, 3), dtype=int)
            self.last_capture_time = 0.0
            self.last_good_frame = None
            logger.Log(f"Calibration reset for camera {self.camera_id} (was {num} samples)")

    def get_status(self) -> Dict:
        with self.lock:
            return {
                "num_samples": len(self.all_corners),
                "target_samples": self.TARGET_SAMPLES,
                "min_samples": self.MIN_SAMPLES,
                "progress": self.progress,
                "status_message": self.status_message,
                "is_calibrating": self.is_calibrating,
                "is_ready": len(self.all_corners) >= self.MIN_SAMPLES,
                "calibration_complete": self.calibration_result is not None,
                "sample_dir": self.sample_dir,
            }

    def get_last_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.last_frame_annotated

    def _generate_distortion_visualization(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        image_size: Tuple[int, int],
    ) -> str:
        import base64
        width, height = image_size
        grid_size = 20
        y_coords = np.linspace(0, height - 1, grid_size)
        x_coords = np.linspace(0, width - 1, grid_size)
        xx, yy = np.meshgrid(x_coords, y_coords)
        points = np.stack([xx.ravel(), yy.ravel()], axis=-1).reshape(-1, 1, 2).astype(np.float32)
        undistorted = cv2.undistortPoints(points, camera_matrix, dist_coeffs, P=camera_matrix)
        distortion = np.linalg.norm(undistorted - points, axis=2).reshape(grid_size, grid_size)
        fig_h, fig_w = 300, int(300 * width / height)
        viz = np.zeros((fig_h, fig_w, 3), dtype=np.uint8)
        max_d = np.max(distortion)
        distortion_norm = (distortion / max_d) if max_d > 0 else distortion
        cell_h, cell_w = fig_h // grid_size, fig_w // grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                d = distortion_norm[i, j]
                if d < 0.33:
                    r, g, b = 0, int(255 * d / 0.33), int(255 * (1 - d / 0.33))
                elif d < 0.66:
                    r, g, b = int(255 * (d - 0.33) / 0.33), 255, 0
                else:
                    r, g, b = 255, int(255 * (1 - (d - 0.66) / 0.34)), 0
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                cv2.rectangle(viz, (x1, y1), (x2, y2), (b, g, r), -1)
        for i in range(grid_size + 1):
            cv2.line(viz, (0, i * cell_h), (fig_w, i * cell_h), (80, 80, 80), 1)
        for j in range(grid_size + 1):
            cv2.line(viz, (j * cell_w, 0), (j * cell_w, fig_h), (80, 80, 80), 1)
        cv2.putText(viz, "Lens Distortion Heatmap", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(viz, f"Max distortion: {max_d:.2f} px", (10, fig_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        _, buffer = cv2.imencode(".png", viz)
        return base64.b64encode(buffer).decode("utf-8")
