import cv2
import numpy as np
import json
from typing import List, Tuple, Optional, Dict
from threading import Lock
from util.logger import Logger
import time

logger = Logger("CharucoCalibrator")


class CharucoCalibrator:
    SQUARES_X = 11
    SQUARES_Y = 8
    SQUARE_LENGTH = 0.015
    MARKER_LENGTH = 0.011
    ARUCO_DICT = cv2.aruco.DICT_4X4_50
    
    MIN_SAMPLES = 20
    TARGET_SAMPLES = 50
    
    def __init__(self, camera_id: int, resolution: Tuple[int, int] = (800, 600)):
        self.camera_id = camera_id
        self.resolution = resolution
        self.lock = Lock()
        
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.ARUCO_DICT)
        self.board = cv2.aruco.CharucoBoard(
            (self.SQUARES_X, self.SQUARES_Y),
            self.SQUARE_LENGTH,
            self.MARKER_LENGTH,
            self.aruco_dict
        )
        
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector_params.adaptiveThreshWinSizeMin = 3
        self.detector_params.adaptiveThreshWinSizeMax = 25
        self.detector_params.adaptiveThreshWinSizeStep = 8
        self.detector_params.minMarkerPerimeterRate = 0.005
        self.detector_params.maxMarkerPerimeterRate = 4.5
        self.detector_params.polygonalApproxAccuracyRate = 0.15
        self.detector_params.minCornerDistanceRate = 0.005
        self.detector_params.minDistanceToBorder = 0
        self.detector_params.minOtsuStdDev = 1.5
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector_params.cornerRefinementWinSize = 4
        self.detector_params.cornerRefinementMaxIterations = 20
        
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
        
        charuco_params = cv2.aruco.CharucoParameters()
        charuco_params.cameraMatrix = None
        charuco_params.distCoeffs = None
        charuco_params.minMarkers = 2
        
        try:
            self.charuco_detector = cv2.aruco.CharucoDetector(
                self.board,
                charuco_params,
                self.detector_params
            )
        except:
            self.charuco_detector = cv2.aruco.CharucoDetector(self.board)
        
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
    
    def process_frame(self, frame: np.ndarray, downsample: bool = True, auto_capture: bool = True) -> Tuple[np.ndarray, bool, str]:
        if frame is None:
            return None, False, "No frame"
        
        if downsample and frame.shape[0] > 480:
            scale = 480.0 / frame.shape[0]
            frame_detect = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            frame_detect = frame
        
        gray = cv2.cvtColor(frame_detect, cv2.COLOR_BGR2GRAY)
        
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray_full)
        
        charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(gray_enhanced)
        
        if marker_ids is not None and len(self.all_corners) % 10 == 0:
            corner_count = len(charuco_corners) if charuco_corners is not None else 0
            logger.Log(f"Detected {len(marker_ids)} ArUco markers, {corner_count} ChArUco corners")
            if corner_count == 0 and len(marker_ids) > 10:
                logger.Warn(f"Many markers ({len(marker_ids)}) but no corners - board may be blurry, warped, or at steep angle")
        
        annotated = frame.copy()
        
        annotated = self._draw_coverage_grid(annotated)
        
        is_good_sample = False
        should_capture = False
        status = ""
        
        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(annotated, marker_corners, marker_ids)
            
            cv2.putText(annotated, f"Markers: {len(marker_ids)}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            if charuco_corners is not None and len(charuco_corners) > 0:
                cv2.putText(annotated, f"Corners: {len(charuco_corners)}", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            if charuco_corners is not None and len(charuco_corners) >= 3:
                cv2.aruco.drawDetectedCornersCharuco(annotated, charuco_corners, charuco_ids)
                
                is_good_sample = len(charuco_corners) >= 4
                
                grid_row, grid_col = self._get_board_position(charuco_corners, frame.shape)
                
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
                    status = f"✓ Good sample! ({len(charuco_corners)} corners)"
                    cv2.putText(annotated, status, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    status = f"Need more corners ({len(charuco_corners)}/12)"
                    cv2.putText(annotated, status, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            else:
                if charuco_corners is not None and len(charuco_corners) > 0:
                    status = f"Need more corners ({len(charuco_corners)}/6 minimum)"
                    cv2.putText(annotated, "Show more of the board", (10, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                else:
                    status = "Can't detect corners"
                    cv2.putText(annotated, "Adjust lighting or print larger board", (10, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(annotated, status, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            status = "No ArUco markers detected"
            cv2.putText(annotated, status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(annotated, "Show ChArUco board to camera", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        progress_text = f"Samples: {len(self.all_corners)}/{self.TARGET_SAMPLES}"
        cv2.putText(annotated, progress_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        with self.lock:
            self.last_frame_annotated = annotated
            self.status_message = status
            self.progress = len(self.all_corners) / self.TARGET_SAMPLES
            
            if (auto_capture and self.auto_capture_enabled and should_capture and 
                len(self.all_corners) < self.TARGET_SAMPLES and
                charuco_corners is not None and charuco_ids is not None):
                current_time = time.time()
                time_since_last = current_time - self.last_capture_time
                if time_since_last >= self.capture_cooldown:
                    self.last_good_frame = (frame.copy(), charuco_corners.copy(), charuco_ids.copy())
                    logger.Log(f"Queued frame for capture (cooldown satisfied: {time_since_last:.1f}s)")
                else:
                    logger.Log(f"Waiting for cooldown: {time_since_last:.1f}s / {self.capture_cooldown}s")
        
        return annotated, is_good_sample, status
    
    def try_auto_capture(self) -> bool:
        with self.lock:
            if self.last_good_frame is None:
                return False
            frame, charuco_corners, charuco_ids = self.last_good_frame
            self.last_good_frame = None
        
        if charuco_corners is not None and len(charuco_corners) >= 12:
            with self.lock:
                self.all_corners.append(charuco_corners)
                self.all_ids.append(charuco_ids)
                if self.image_size is None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    self.image_size = gray.shape[::-1]
                
                grid_row, grid_col = self._get_board_position(charuco_corners, frame.shape)
                if grid_row is not None:
                    self.coverage_grid[grid_row, grid_col] += 1
                
                self.last_capture_time = time.time()
                
                logger.Log(f"Added sample {len(self.all_corners)}/{self.TARGET_SAMPLES} "
                         f"with {len(charuco_corners)} corners at position [{grid_row},{grid_col}]")
            return True
        return False
    
    def add_sample(self, frame: np.ndarray, downsample: bool = False) -> bool:
        if frame is None:
            return False
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)
        
        charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(gray_enhanced)
        
        if charuco_corners is not None and len(charuco_corners) >= 6:  # Lowered threshold
                with self.lock:
                    self.all_corners.append(charuco_corners)
                    self.all_ids.append(charuco_ids)
                    if self.image_size is None:
                        self.image_size = gray.shape[::-1]
                    logger.Log(f"Added sample {len(self.all_corners)}/{self.TARGET_SAMPLES} "
                             f"with {len(charuco_corners)} corners")
                return True
        
        return False
    
    def calibrate(self) -> Optional[Dict]:
        with self.lock:
            if len(self.all_corners) < self.MIN_SAMPLES:
                logger.Warn(f"Not enough samples for calibration: {len(self.all_corners)}/{self.MIN_SAMPLES}")
                self.status_message = f"Need {self.MIN_SAMPLES - len(self.all_corners)} more samples"
                return None
            
            self.is_calibrating = True
            self.status_message = "Computing calibration..."
            logger.Log(f"Starting calibration with {len(self.all_corners)} samples...")
        
        try:
            start_time = time.time()
            logger.Log(f"Preparing calibration data from {len(self.all_corners)} samples...")
            
            all_obj_points = []
            all_img_points = []
            
            for corners, ids in zip(self.all_corners, self.all_ids):
                obj_points = self.board.getChessboardCorners()[ids.flatten()]
                all_obj_points.append(obj_points)
                all_img_points.append(corners)
            
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                all_obj_points,
                all_img_points,
                self.image_size,
                None,
                None
            )
            
            if not ret:
                logger.Error("Calibration failed")
                with self.lock:
                    self.is_calibrating = False
                    self.status_message = "Calibration failed"
                return None
            
            mean_error = 0
            for i in range(len(all_obj_points)):
                corners_reprojected, _ = cv2.projectPoints(
                    all_obj_points[i],
                    rvecs[i], tvecs[i], camera_matrix, dist_coeffs
                )
                error = cv2.norm(all_img_points[i], corners_reprojected, cv2.NORM_L2) / len(corners_reprojected)
                mean_error += error
            
            mean_error /= len(all_obj_points)
            
            elapsed = time.time() - start_time
            logger.Log(f"Calibration completed in {elapsed:.2f}s with mean error: {mean_error:.4f}")
            
            fx = camera_matrix[0, 0]
            fy = camera_matrix[1, 1]
            cx = camera_matrix[0, 2]
            cy = camera_matrix[1, 2]
            
            width, height = self.image_size
            
            fov_x = 2 * np.arctan(width / (2 * fx))
            fov_y = 2 * np.arctan(height / (2 * fy))
            fov_x_deg = fov_x * 180 / np.pi
            fov_y_deg = fov_y * 180 / np.pi
            fov_diag = 2 * np.arctan(np.sqrt(width**2 + height**2) / (2 * fx)) * 180 / np.pi
            
            calibration_dict = {
                "meta": {
                    "resolution": {
                        "width": width,
                        "height": height
                    },
                    "camera_matrix": camera_matrix.tolist(),
                    "dist_coeffs": dist_coeffs.tolist(),
                    "reprojection_error": float(mean_error),
                    "num_samples": len(self.all_corners),
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
                self.status_message = f"✓ Calibration complete! Error: {mean_error:.4f}"
            
            return calibration_dict
            
        except Exception as e:
            logger.Error(f"Calibration error: {e}")
            with self.lock:
                self.is_calibrating = False
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

