import cv2
import numpy as np
import json
from typing import List, Tuple, Optional, Dict, Callable
from threading import Lock
from util.logger import Logger
import time

logger = Logger("CharucoCalibrator")


class CharucoCalibrator:
    SQUARES_X = 11
    SQUARES_Y = 8
    SQUARE_LENGTH = 0.0212
    MARKER_LENGTH = 0.0155466
    ARUCO_DICT = cv2.aruco.DICT_4X4_50
    
    MIN_SAMPLES = 20
    TARGET_SAMPLES = 54
    
    def __init__(self, camera_id: int, resolution: Tuple[int, int] = (800, 600)):
        self.camera_id = camera_id
        self.resolution = resolution
        self.lock = Lock()
        
        self.detector_params = cv2.aruco.DetectorParameters()
        if hasattr(self.detector_params, "minMarkerPerimeterRate"):
            self.detector_params.minMarkerPerimeterRate = 0.04 
        if hasattr(self.detector_params, "maxErroneousBitsInBorderRate"):
            self.detector_params.maxErroneousBitsInBorderRate = 0.35 

        if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
            self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            if hasattr(self.detector_params, "cornerRefinementWinSize"):
                self.detector_params.cornerRefinementWinSize = 5
            if hasattr(self.detector_params, "cornerRefinementMaxIterations"):
                self.detector_params.cornerRefinementMaxIterations = 30
            if hasattr(self.detector_params, "cornerRefinementMinAccuracy"):
                self.detector_params.cornerRefinementMinAccuracy = 0.2

        self._dict_candidates = [
            cv2.aruco.DICT_4X4_50,
            cv2.aruco.DICT_4X4_100,
            cv2.aruco.DICT_4X4_250,
            cv2.aruco.DICT_4X4_1000,
        ]
        self._dict_autodetected = False
        self._layout_candidates = [(11, 8), (8, 11)]
        self._layout_autodetected = False
        self._set_aruco_dictionary(self.ARUCO_DICT)

        self._aruco_detector = None
        self._rebuild_aruco_detector()

        self._has_charuco = hasattr(cv2.aruco, "interpolateCornersCharuco")
        
        self.all_corners: List[np.ndarray] = []
        self.all_ids: List[np.ndarray] = []
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
        

        self.min_charuco_corners_live = 6
        self.min_charuco_corners_capture = 10

        self.blur_laplacian_var_min = 50.0
        self.min_board_bbox_area_frac = 0.03
        self.min_board_span_frac = 0.15

        self.min_quadrants_covered = 1

    def _set_aruco_dictionary(self, dict_id: int) -> None:
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        self.board = cv2.aruco.CharucoBoard(
            (self.SQUARES_X, self.SQUARES_Y),
            self.SQUARE_LENGTH,
            self.MARKER_LENGTH,
            self.aruco_dict
        )
        self._rebuild_aruco_detector()

    def _rebuild_aruco_detector(self) -> None:
        self._aruco_detector = None
        if hasattr(cv2.aruco, "ArucoDetector"):
            self._aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)

    def _detect_markers(self, gray_image: np.ndarray):
        if self._aruco_detector is not None:
            corners, ids, rejected = self._aruco_detector.detectMarkers(gray_image)
        elif hasattr(cv2.aruco, "detectMarkers"):
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray_image, self.aruco_dict, parameters=self.detector_params
            )
        else:
            raise RuntimeError(
                "ArUco marker detection is unavailable in this OpenCV build. "
                "This build is missing both cv2.aruco.ArucoDetector and cv2.aruco.detectMarkers."
            )

        if (ids is None or len(ids) == 0) and not self._dict_autodetected:
            best = (0, None, None, None)  # (count, corners, ids, rejected)
            best_dict = None
            for cand in self._dict_candidates:
                if cand == self.ARUCO_DICT:
                    continue
                try:
                    tmp_dict = cv2.aruco.getPredefinedDictionary(cand)
                    if self._aruco_detector is not None and hasattr(cv2.aruco, "ArucoDetector"):
                        tmp_detector = cv2.aruco.ArucoDetector(tmp_dict, self.detector_params)
                        c2, i2, r2 = tmp_detector.detectMarkers(gray_image)
                    else:
                        c2, i2, r2 = cv2.aruco.detectMarkers(gray_image, tmp_dict, parameters=self.detector_params)
                    cnt = 0 if i2 is None else len(i2)
                    if cnt > best[0]:
                        best = (cnt, c2, i2, r2)
                        best_dict = cand
                except Exception:
                    continue

            if best_dict is not None and best[0] > 0:
                logger.Log(f"Auto-detected ArUco dictionary: {best_dict}")
                self.ARUCO_DICT = best_dict
                self._set_aruco_dictionary(best_dict)
                self._dict_autodetected = True
                return best[1], best[2], best[3]

        return corners, ids, rejected
        
    def generate_board_image(self, width: int = 1000, height: int = 1000) -> np.ndarray:
        board_image = self.board.generateImage((width, height), marginSize=20)
        return board_image
    
    def _get_board_position(self, corners: np.ndarray, frame_shape: Tuple[int, int]) -> Tuple[int, int]:
        if corners is None or len(corners) == 0:
            return None, None
        
        center_x = np.mean(corners[:, 0, 0])
        center_y = np.mean(corners[:, 0, 1])
        
        h, w = frame_shape[:2]
        
        grid_col = int(center_x / w * 3)
        grid_row = int(center_y / h * 3)
        
        grid_col = max(0, min(2, grid_col))
        grid_row = max(0, min(2, grid_row))
        
        return grid_row, grid_col
    
    def _draw_coverage_grid(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        cell_h = h // 3
        cell_w = w // 3
        
        overlay = frame.copy()
        
        for row in range(3):
            for col in range(3):
                count = self.coverage_grid[row, col]
                
                x1, y1 = col * cell_w, row * cell_h
                x2, y2 = (col + 1) * cell_w, (row + 1) * cell_h
                
                if count == 0:
                    color = (0, 0, 200)
                    alpha = 0.2
                elif count < self.max_samples_per_cell:
                    color = (0, 200, 200)
                    alpha = 0.15
                else:
                    color = (0, 200, 0)
                    alpha = 0.1
                
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (200, 200, 200), 2)
                
                text = str(count)
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                text_x = x1 + (cell_w - text_size[0]) // 2
                text_y = y1 + (cell_h + text_size[1]) // 2
                cv2.putText(overlay, text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        return result

    def _laplacian_variance(self, gray: np.ndarray) -> float:
        try:
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception:
            return 0.0

    def _points_xy(self, pts: np.ndarray) -> Optional[np.ndarray]:
        if pts is None:
            return None
        arr = np.array(pts, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[1] == 1 and arr.shape[2] == 2:
            return arr[:, 0, :]
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr
        return None

    def _sample_quadrants_covered(self, xy: np.ndarray, frame_shape: Tuple[int, int]) -> int:
        h, w = frame_shape[:2]
        cx, cy = w * 0.5, h * 0.5
        q = set()
        for x, y in xy:
            qx = 0 if x < cx else 1
            qy = 0 if y < cy else 1
            q.add((qx, qy))
        return len(q)

    def _is_sample_acceptable(
        self,
        gray: np.ndarray,
        sample_points: np.ndarray,
        frame_shape: Tuple[int, int],
        required_min_points: int,
    ) -> Tuple[bool, str]:
        xy = self._points_xy(sample_points)
        if xy is None or len(xy) < required_min_points:
            return False, f"Need {required_min_points}+ points"

        lap_var = self._laplacian_variance(gray)
        if lap_var < self.blur_laplacian_var_min:
            return False, f"Too blurry ({lap_var:.0f})"

        h, w = frame_shape[:2]
        min_xy = np.min(xy, axis=0)
        max_xy = np.max(xy, axis=0)
        bw = float(max_xy[0] - min_xy[0])
        bh = float(max_xy[1] - min_xy[1])
        if bw <= 1.0 or bh <= 1.0:
            return False, "Bad geometry"

        area_frac = (bw * bh) / float(w * h)
        span_x = bw / float(w)
        span_y = bh / float(h)
        if area_frac < self.min_board_bbox_area_frac:
            return False, "Board too small"
        if max(span_x, span_y) < self.min_board_span_frac:
            return False, "Board too small"

        q = self._sample_quadrants_covered(xy, frame_shape)
        if q < self.min_quadrants_covered:
            return False, "Move board around"

        return True, "OK"
    
    def process_frame(self, frame: np.ndarray, downsample: bool = True, auto_capture: bool = True) -> Tuple[np.ndarray, bool, str]:
        if frame is None:
            logger.Warn("process_frame called with None frame")
            return None, False, "No frame"
        
        if downsample and frame.shape[0] > 480:
            scale = 480.0 / frame.shape[0]
            frame_detect = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            frame_detect = frame
        
        gray = cv2.cvtColor(frame_detect, cv2.COLOR_BGR2GRAY)
        
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        marker_corners, marker_ids, rejected = self._detect_markers(gray_full)
        
        # if (marker_ids is not None and len(marker_ids) > 0 and rejected is not None and len(rejected) > 0 and
        #         hasattr(cv2.aruco, "refineDetectedMarkers")):
        #     try:
        #         cv2.aruco.refineDetectedMarkers(
        #             gray_full, self.board, marker_corners, marker_ids, rejected,
        #             minRepDistance=5.0, errorCorrectionRate=5.0,
        #             parameters=self.detector_params
        #         )
        #     except Exception as e:
        #         logger.Log(f"refineDetectedMarkers skipped: {e}")
        
        charuco_corners, charuco_ids = None, None
        if marker_ids is not None and len(marker_ids) > 0:
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners,
                marker_ids,
                gray_full,
                self.board,
                minMarkers=1
            )
            if ret == 0 and not self._layout_autodetected:
                for sx, sy in self._layout_candidates:
                    if (sx, sy) == (self.SQUARES_X, self.SQUARES_Y):
                        continue
                    try:
                        alt_board = cv2.aruco.CharucoBoard(
                            (sx, sy), self.SQUARE_LENGTH, self.MARKER_LENGTH, self.aruco_dict
                        )
                        ret2, cc2, ci2 = cv2.aruco.interpolateCornersCharuco(
                            marker_corners, marker_ids, gray_full, alt_board, minMarkers=1
                        )
                        if ret2 > 0 and cc2 is not None and len(cc2) > 0:
                            logger.Log(f"Auto-detected board layout: {sx}x{sy} (was {self.SQUARES_X}x{self.SQUARES_Y})")
                            self.SQUARES_X, self.SQUARES_Y = sx, sy
                            self._set_aruco_dictionary(self.ARUCO_DICT)
                            self._layout_autodetected = True
                            charuco_corners, charuco_ids = cc2, ci2
                            break
                    except Exception:
                        continue
            if ret == 0 and charuco_corners is None:
                charuco_corners, charuco_ids = None, None
        
        frame_count = len(self.all_corners)
        if marker_ids is not None and len(marker_ids) > 0:
            corner_count = len(charuco_corners) if charuco_corners is not None else 0
            if frame_count % 5 == 0:
                logger.Log(f"Detected {len(marker_ids)} ArUco markers, {corner_count} ChArUco corners")
            if corner_count == 0 and len(marker_ids) > 10:
                logger.Warn(f"Many markers ({len(marker_ids)}) but no corners - board may be blurry, warped, or at steep angle")
        elif frame_count % 30 == 0:
            logger.Log(f"No markers detected (frame shape: {gray_full.shape})")
        
        annotated = frame.copy()
        
        annotated = self._draw_coverage_grid(annotated)
        
        is_good_sample = False
        should_capture = False
        status = ""
        
        cv2.putText(annotated, f"Frame: {frame.shape[1]}x{frame.shape[0]}", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(annotated, marker_corners, marker_ids)
            
            corner_count = len(charuco_corners) if charuco_corners is not None else 0
            cv2.putText(annotated, f"Markers: {len(marker_ids)}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(annotated, f"Corners: {corner_count} (need {self.min_charuco_corners_live}+)", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            sample_points = None
            if self._has_charuco and charuco_corners is not None and len(charuco_corners) >= 3:
                if hasattr(cv2.aruco, "drawDetectedCornersCharuco"):
                    cv2.aruco.drawDetectedCornersCharuco(annotated, charuco_corners, charuco_ids)
                sample_points = charuco_corners
                is_good_sample = len(charuco_corners) >= self.min_charuco_corners_live
            else:
                is_good_sample = False

            if is_good_sample and sample_points is not None:
                req = self.min_charuco_corners_live
                ok, reason = self._is_sample_acceptable(gray_full, sample_points, frame.shape, req)
                if not ok:
                    is_good_sample = False
                    status = reason

            grid_row, grid_col = None, None
            if sample_points is not None:
                grid_row, grid_col = self._get_board_position(sample_points, frame.shape)

            if is_good_sample and grid_row is not None:
                samples_in_cell = self.coverage_grid[grid_row, grid_col]
                
                if samples_in_cell < self.max_samples_per_cell:
                    should_capture = True
                    status = f"✓ Good sample! ({len(charuco_corners)} corners) - Position [{grid_row},{grid_col}]"
                    cv2.putText(annotated, status, (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    logger.Log(f"Ready to capture at position [{grid_row},{grid_col}] - {samples_in_cell}/{self.max_samples_per_cell} samples in cell")
                else:
                    should_capture = False
                    status = f"Move to different area (position [{grid_row},{grid_col}] full)"
                    cv2.putText(annotated, status, (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    cv2.putText(annotated, "Try corners or edges", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    logger.Log(f"Grid cell [{grid_row},{grid_col}] full ({samples_in_cell}/{self.max_samples_per_cell}), need different position")
            elif is_good_sample:
                status = "✓ Good sample!"
                cv2.putText(annotated, status, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                if corner_count < self.min_charuco_corners_live and corner_count > 0:
                    status = status or f"Need {self.min_charuco_corners_live}+ corners (have {corner_count})"
                elif corner_count == 0 and len(marker_ids) > 5:
                    status = status or "No ChArUco corners - show full board, reduce blur/angle (or check board is 11x8)"
                else:
                    status = status or "Need more corners / better view"
                cv2.putText(annotated, "Show more of the board", (10, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                cv2.putText(annotated, status, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            status = "No ArUco markers detected"
            cv2.putText(annotated, status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(annotated, "Show ChArUco board to camera", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(annotated, f"Board: {self.SQUARES_X}x{self.SQUARES_Y}, Dict: 4x4", (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        progress_text = f"Samples: {len(self.all_corners)}/{self.TARGET_SAMPLES}"
        cv2.putText(annotated, progress_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        with self.lock:
            self.last_frame_annotated = annotated
            self.status_message = status
            self.progress = len(self.all_corners) / self.TARGET_SAMPLES
            
            if (auto_capture and self.auto_capture_enabled and should_capture and 
                len(self.all_corners) < self.TARGET_SAMPLES):
                current_time = time.time()
                time_since_last = current_time - self.last_capture_time
                if time_since_last >= self.capture_cooldown:
                    if self._has_charuco and charuco_corners is not None and charuco_ids is not None:
                        self.last_good_frame = (frame.copy(), charuco_corners, charuco_ids.copy())
                        logger.Log(f"Queued frame for capture (cooldown satisfied: {time_since_last:.1f}s)")
                else:
                    logger.Log(f"Waiting for cooldown: {time_since_last:.1f}s / {self.capture_cooldown}s")
        
        return annotated, is_good_sample, status

    def _reject_calibration_outliers(
        self,
        corners_list: List[np.ndarray],
        ids_list: List[np.ndarray],
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        rvecs: List[np.ndarray],
        tvecs: List[np.ndarray],
        max_per_view_error_px: float = 2.5,
        progress_callback: Optional[Callable[[float], None]] = None,
        per_view_errors_from_calibration: Optional[np.ndarray] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Drop views with per-view reprojection error above threshold. Uses OpenCV per-view errors
        when provided; does not compute from getChessboardCorners() (ordering can differ from
        interpolateCornersCharuco(), causing wrong errors and dropping all views)."""
        n_views = len(corners_list)
        if n_views == 0:
            return corners_list, ids_list

        if per_view_errors_from_calibration is not None and len(per_view_errors_from_calibration) == n_views:
            per_view_errors = np.asarray(per_view_errors_from_calibration, dtype=np.float64)
        else:
            per_view_errors = None

        if per_view_errors is None:
            return corners_list, ids_list

        valid = np.isfinite(per_view_errors)
        if not np.any(valid):
            return corners_list, ids_list
        median_err = np.median(per_view_errors[valid])
        threshold = min(max_per_view_error_px, max(1.0, median_err + 2 * np.std(per_view_errors[valid])))
        keep = [i for i in range(n_views) if per_view_errors[i] <= threshold]
        dropped = n_views - len(keep)
        if dropped > 0:
            logger.Log(f"Dropping {dropped} outlier view(s) (per-view error > {threshold:.2f} px)")
            return [corners_list[i] for i in keep], [ids_list[i] for i in keep]
        return corners_list, ids_list
    
    def try_auto_capture(self) -> bool:
        with self.lock:
            if self.last_good_frame is None:
                return False
            frame, points, ids = self.last_good_frame
            self.last_good_frame = None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        good = self._has_charuco and points is not None and len(points) >= self.min_charuco_corners_capture
        if good:
            ok, _ = self._is_sample_acceptable(
                gray, points, frame.shape, self.min_charuco_corners_capture
            )
            good = good and ok

        if good:
            with self.lock:
                self.all_corners.append(points)
                self.all_ids.append(ids)
                if self.image_size is None:
                    self.image_size = gray.shape[::-1]
                
                grid_row, grid_col = self._get_board_position(points, frame.shape)
                if grid_row is not None:
                    self.coverage_grid[grid_row, grid_col] += 1
                
                self.last_capture_time = time.time()
                
                logger.Log(
                    f"Added sample {len(self.all_corners)}/{self.TARGET_SAMPLES} "
                    f"with {len(points)} points at position [{grid_row},{grid_col}]"
                )
            return True
        return False
    
    def calibrate(self) -> Optional[Dict]:
        with self.lock:
            if len(self.all_corners) < self.MIN_SAMPLES:
                logger.Warn(f"Not enough samples for calibration: {len(self.all_corners)}/{self.MIN_SAMPLES}")
                self.status_message = f"Need {self.MIN_SAMPLES - len(self.all_corners)} more samples"
                return None
            
            self.is_calibrating = True
            self.status_message = "Preparing data... (0%)"
            self.progress = 0.0
            logger.Log(f"Starting calibration with {len(self.all_corners)} samples...")
        
        try:
            start_time = time.time()
            cal_flags = 0
            if hasattr(cv2, "CALIB_USE_LU"):
                cal_flags = cv2.CALIB_USE_LU
            if not self._has_charuco or not hasattr(cv2.aruco, "calibrateCameraCharuco"):
                raise ValueError("ChArUco calibration is not available in this OpenCV build")

            corners_list = list(self.all_corners)
            ids_list = list(self.all_ids)
            n_views = len(corners_list)

            with self.lock:
                self.progress = 0.05
                self.status_message = "Preparing data... (5%)"
            with self.lock:
                self.progress = 0.10
                self.status_message = "Running initial calibration... (10%)"

            def _run_calibration(corners, ids):
                try:
                    return cv2.aruco.calibrateCameraCharuco(
                        corners, ids, self.board, self.image_size, None, None, flags=cal_flags
                    )
                except TypeError:
                    return cv2.aruco.calibrateCameraCharuco(
                        corners, ids, self.board, self.image_size, None, None
                    )

            ret, camera_matrix, dist_coeffs, rvecs, tvecs = _run_calibration(corners_list, ids_list)
            mean_error = float(ret)
            per_view_errors_from_cv = None
            if camera_matrix is None or np.any(np.isnan(camera_matrix)):
                raise ValueError("calibrateCameraCharuco returned invalid camera matrix")

            with self.lock:
                self.progress = 0.40
                self.status_message = "Initial calibration done. Checking view quality... (40%)"

            def _outlier_progress(frac: float) -> None:
                with self.lock:
                    self.progress = 0.40 + frac * 0.05
                    self.status_message = f"Checking view quality... ({int(100 * self.progress)}%)"

            corners_list, ids_list = self._reject_calibration_outliers(
                corners_list, ids_list, camera_matrix, dist_coeffs, rvecs, tvecs,
                per_view_errors_from_calibration=per_view_errors_from_cv,
                progress_callback=_outlier_progress,
            )
            if len(corners_list) < self.MIN_SAMPLES:
                raise ValueError(
                    f"After removing outlier views, too few samples ({len(corners_list)} < {self.MIN_SAMPLES}). "
                    "Recapture with sharper, well-lit views of the board."
                )

            if len(corners_list) < n_views:
                with self.lock:
                    self.progress = 0.50
                    self.status_message = "Re-running calibration without outliers... (50%)"
                ret, camera_matrix, dist_coeffs, rvecs, tvecs = _run_calibration(corners_list, ids_list)
                mean_error = float(ret)
                logger.Log(f"Recalibrated with {len(corners_list)} views (dropped {n_views - len(corners_list)} outliers)")
                with self.lock:
                    self.progress = 0.80
                    self.status_message = "Re-calibration done. Building result... (80%)"
            else:
                with self.lock:
                    self.progress = 0.75
                    self.status_message = "Building result... (75%)"

            with self.lock:
                self.progress = 0.90
                self.status_message = "Finalizing... (90%)"
            
            elapsed = time.time() - start_time
            
            fx = camera_matrix[0, 0]
            fy = camera_matrix[1, 1]
            cx = camera_matrix[0, 2]
            cy = camera_matrix[1, 2]
            
            width, height = self.image_size
            
            fov_x_deg = 2 * np.arctan(width / (2 * fx)) * 180 / np.pi
            fov_y_deg = 2 * np.arctan(height / (2 * fy)) * 180 / np.pi
            tan_half_x = width / (2 * fx)
            tan_half_y = height / (2 * fy)
            fov_diag = 2 * np.arctan(np.sqrt(tan_half_x**2 + tan_half_y**2)) * 180 / np.pi
            
            logger.Log(f"Calibration completed in {elapsed:.2f}s")
            logger.Log(f"Board config: {self.SQUARES_X}x{self.SQUARES_Y} squares, {self.SQUARE_LENGTH*1000:.1f}mm square size")
            logger.Log(f"Camera matrix: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
            logger.Log(f"FOV: H={fov_x_deg:.1f}°, V={fov_y_deg:.1f}°, D={fov_diag:.1f}°")
            logger.Log(f"Reprojection error: {mean_error:.4f} px")
            
            if fov_x_deg > 120 or fov_y_deg > 120:
                logger.Warn(f"FOV seems unusually wide - check board dimensions (SQUARES_X/Y) and SQUARE_LENGTH")
            if fov_x_deg < 30 or fov_y_deg < 30:
                logger.Warn(f"FOV seems unusually narrow - check board dimensions (SQUARES_X/Y) and SQUARE_LENGTH")
            
            calibration_dict = {
                "meta": {
                    "resolution": {
                        "width": width,
                        "height": height
                    },
                    "camera_matrix": camera_matrix.tolist(),
                    "dist_coeffs": dist_coeffs.tolist(),
                    "reprojection_error": float(mean_error),
                    "num_samples": len(corners_list),
                    "calibration_time": elapsed,
                    "fov": {
                        'horizontal': float(fov_x_deg),
                        'vertical': float(fov_y_deg),
                        'diagonal': float(fov_diag)
                    }
                }
            }
            
            distortion_viz = self._generate_distortion_visualization(
                camera_matrix, dist_coeffs, self.image_size
            )
            calibration_dict['meta']['distortion_viz_base64'] = distortion_viz
            
            with self.lock:
                self.calibration_result = calibration_dict
                self.is_calibrating = False
                self.progress = 1.0
                self.status_message = f"✓ Calibration complete! Error: {mean_error:.4f}"
            
            return calibration_dict
            
        except Exception as e:
            logger.Error(f"Calibration error: {e}")
            with self.lock:
                self.is_calibrating = False
                self.progress = 0.0
                self.status_message = f"Error: {str(e)}"
            return None
    
    def save_calibration(self, output_path: str = "cal.json") -> bool:
        if self.calibration_result is None:
            logger.Error("No calibration result to save")
            return False
        
        try:
            try:
                with open(output_path, 'r') as f:
                    all_calibrations = json.load(f)
            except FileNotFoundError:
                all_calibrations = {}
            
            all_calibrations[str(self.camera_id)] = self.calibration_result
            
            with open(output_path, 'w') as f:
                json.dump(all_calibrations, f, indent=2)
            
            logger.Log(f"Calibration saved for camera {self.camera_id} to {output_path}")
            return True
            
        except Exception as e:
            logger.Error(f"Failed to save calibration: {e}")
            return False
    
    def reset(self):
        with self.lock:
            num_samples = len(self.all_corners)
            self.all_corners.clear()
            self.all_ids.clear()
            self.image_size = None
            self.calibration_result = None
            self.status_message = "Ready to calibrate"
            self.progress = 0.0
            self.coverage_grid = np.zeros((3, 3), dtype=int)
            self.last_capture_time = 0.0
            self.last_good_frame = None
            logger.Log(f"Calibration reset for camera {self.camera_id} (was at {num_samples} samples)")
    
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
                "calibration_complete": self.calibration_result is not None
            }
    
    def get_last_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.last_frame_annotated
    
    def _generate_distortion_visualization(self, camera_matrix: np.ndarray, 
                                          dist_coeffs: np.ndarray, 
                                          image_size: Tuple[int, int]) -> str:
        import base64
        
        width, height = image_size
        
        grid_size = 20
        y_coords = np.linspace(0, height - 1, grid_size)
        x_coords = np.linspace(0, width - 1, grid_size)
        
        xx, yy = np.meshgrid(x_coords, y_coords)
        points = np.stack([xx.ravel(), yy.ravel()], axis=-1).reshape(-1, 1, 2).astype(np.float32)
        
        undistorted = cv2.undistortPoints(points, camera_matrix, dist_coeffs, P=camera_matrix)
        
        distortion = np.linalg.norm(undistorted - points, axis=2).reshape(grid_size, grid_size)
        
        fig_height = 300
        fig_width = int(fig_height * width / height)
        
        viz = np.zeros((fig_height, fig_width, 3), dtype=np.uint8)
        
        max_distortion = np.max(distortion)
        if max_distortion > 0:
            distortion_norm = distortion / max_distortion
        else:
            distortion_norm = distortion
        
        cell_h = fig_height // grid_size
        cell_w = fig_width // grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                dist_val = distortion_norm[i, j]
                
                if dist_val < 0.33:
                    r = 0
                    g = int(255 * (dist_val / 0.33))
                    b = int(255 * (1 - dist_val / 0.33))
                elif dist_val < 0.66:
                    r = int(255 * ((dist_val - 0.33) / 0.33))
                    g = 255
                    b = 0
                else:
                    r = 255
                    g = int(255 * (1 - (dist_val - 0.66) / 0.34))
                    b = 0
                
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                cv2.rectangle(viz, (x1, y1), (x2, y2), (b, g, r), -1)
        
        for i in range(grid_size + 1):
            y = i * cell_h
            cv2.line(viz, (0, y), (fig_width, y), (80, 80, 80), 1)
        for j in range(grid_size + 1):
            x = j * cell_w
            cv2.line(viz, (x, 0), (x, fig_height), (80, 80, 80), 1)
        
        cv2.putText(viz, "Lens Distortion Heatmap", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(viz, f"Max distortion: {max_distortion:.2f} pixels", (10, fig_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        _, buffer = cv2.imencode('.png', viz)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return img_base64

