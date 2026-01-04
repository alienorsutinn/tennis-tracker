"""
Tennis Analysis Pipeline
========================
A complete video processing pipeline for tennis match analysis.
Based on CourtCheck (AggieSportsAnalytics), yastrebksv/TennisProject, and roboflow/sports.

Features:
- Court keypoint detection (17 keypoints)
- Ball tracking using TrackNet
- Homography transformation to 2D court view
- Player detection and tracking
- Bounce detection
- IN/OUT call automation

Usage:
    python tennis_pipeline.py --input video.mp4 --output output.mp4

Requirements:
    pip install opencv-python numpy torch torchvision ultralytics
"""

import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import argparse
from pathlib import Path
import json

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Pipeline configuration"""
    # Video settings
    input_resolution: Tuple[int, int] = (1280, 720)
    output_resolution: Tuple[int, int] = (1280, 720)
    fps: int = 30
    
    # Court keypoints (17 points as per CourtCheck)
    keypoint_names: List[str] = field(default_factory=lambda: [
        "BTL",   # Bottom Top Left (doubles corner)
        "BTLI",  # Bottom Top Left Inner (singles)
        "ITL",   # Inner Top Left (service box)
        "ITM",   # Inner Top Middle (T-junction)
        "ITR",   # Inner Top Right (service box)
        "BTRI",  # Bottom Top Right Inner (singles)
        "BTR",   # Bottom Top Right (doubles corner)
        "NL",    # Net Left
        "NM",    # Net Middle
        "NR",    # Net Right
        "IBL",   # Inner Bottom Left (service box)
        "IBM",   # Inner Bottom Middle (T-junction)
        "IBR",   # Inner Bottom Right (service box)
        "BBL",   # Bottom Bottom Left (doubles corner)
        "BBLI",  # Bottom Bottom Left Inner (singles)
        "BBRI",  # Bottom Bottom Right Inner (singles)
        "BBR",   # Bottom Bottom Right (doubles corner)
    ])
    
    # Court lines to draw (connections between keypoints)
    court_lines: List[Tuple[str, str]] = field(default_factory=lambda: [
        # Top baseline
        ("BTL", "BTLI"), ("BTLI", "BTRI"), ("BTRI", "BTR"),
        # Bottom baseline  
        ("BBL", "BBLI"), ("BBLI", "BBRI"), ("BBRI", "BBR"),
        # Left sidelines
        ("BTL", "NL"), ("NL", "BBL"),
        # Right sidelines
        ("BTR", "NR"), ("NR", "BBR"),
        # Singles lines
        ("BTLI", "ITL"), ("ITL", "IBL"), ("IBL", "BBLI"),
        ("BTRI", "ITR"), ("ITR", "IBR"), ("IBR", "BBRI"),
        # Service lines
        ("ITL", "ITM"), ("ITM", "ITR"),
        ("IBL", "IBM"), ("IBM", "IBR"),
        # Center service line
        ("ITM", "NM"), ("NM", "IBM"),
        # Net
        ("NL", "NM"), ("NM", "NR"),
    ])
    
    # 2D court dimensions (pixels)
    court_2d_width: int = 300
    court_2d_height: int = 600
    
    # Tracking settings
    ball_trail_length: int = 10
    keypoint_history_length: int = 10
    
    # Detection thresholds
    ball_confidence_threshold: float = 0.5
    keypoint_confidence_threshold: float = 0.3
    bounce_detection_threshold: float = 5.0


# =============================================================================
# COURT REFERENCE (Based on yastrebksv/TennisCourtDetector)
# =============================================================================

class CourtReference:
    """
    Reference tennis court with real-world dimensions.
    Used for homography transformation to 2D view.
    """
    
    # ITF standard court dimensions in meters
    COURT_LENGTH = 23.77
    COURT_WIDTH = 10.97  # Doubles
    SINGLES_WIDTH = 8.23
    SERVICE_LINE_DISTANCE = 6.40
    NET_HEIGHT = 0.914
    
    def __init__(self, width: int = 300, height: int = 600):
        self.width = width
        self.height = height
        self.margin = 20
        
        # Calculate scale
        self.scale_x = (width - 2 * self.margin) / self.COURT_WIDTH
        self.scale_y = (height - 2 * self.margin) / self.COURT_LENGTH
        
        # Generate reference keypoints
        self.keypoints = self._generate_keypoints()
        
    def _generate_keypoints(self) -> Dict[str, Tuple[int, int]]:
        """Generate 2D reference court keypoints"""
        m = self.margin
        w = self.width - 2 * m
        h = self.height - 2 * m
        
        # Calculate positions
        doubles_left = m
        doubles_right = m + w
        singles_left = m + (self.COURT_WIDTH - self.SINGLES_WIDTH) / 2 * self.scale_x
        singles_right = m + (self.COURT_WIDTH + self.SINGLES_WIDTH) / 2 * self.scale_x
        
        top_baseline = m
        bottom_baseline = m + h
        net_line = m + h / 2
        
        service_top = m + (self.COURT_LENGTH / 2 - self.SERVICE_LINE_DISTANCE) * self.scale_y
        service_bottom = m + (self.COURT_LENGTH / 2 + self.SERVICE_LINE_DISTANCE) * self.scale_y
        
        center_x = m + w / 2
        
        return {
            # Top side
            "BTL": (int(doubles_left), int(top_baseline)),
            "BTLI": (int(singles_left), int(top_baseline)),
            "BTRI": (int(singles_right), int(top_baseline)),
            "BTR": (int(doubles_right), int(top_baseline)),
            
            # Service boxes top
            "ITL": (int(singles_left), int(service_top)),
            "ITM": (int(center_x), int(service_top)),
            "ITR": (int(singles_right), int(service_top)),
            
            # Net
            "NL": (int(doubles_left), int(net_line)),
            "NM": (int(center_x), int(net_line)),
            "NR": (int(doubles_right), int(net_line)),
            
            # Service boxes bottom
            "IBL": (int(singles_left), int(service_bottom)),
            "IBM": (int(center_x), int(service_bottom)),
            "IBR": (int(singles_right), int(service_bottom)),
            
            # Bottom side
            "BBL": (int(doubles_left), int(bottom_baseline)),
            "BBLI": (int(singles_left), int(bottom_baseline)),
            "BBRI": (int(singles_right), int(bottom_baseline)),
            "BBR": (int(doubles_right), int(bottom_baseline)),
        }
    
    def get_corner_points(self) -> np.ndarray:
        """Get the 4 corner points for homography calculation"""
        return np.array([
            self.keypoints["BTL"],
            self.keypoints["BTR"],
            self.keypoints["BBL"],
            self.keypoints["BBR"],
        ], dtype=np.float32)
    
    def draw_court(self, img: np.ndarray = None, config: Config = None) -> np.ndarray:
        """Draw the reference court"""
        if img is None:
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            img[:] = (30, 30, 30)  # Dark background
        
        # Draw court surface
        court_pts = np.array([
            self.keypoints["BTL"],
            self.keypoints["BTR"],
            self.keypoints["BBR"],
            self.keypoints["BBL"],
        ], dtype=np.int32)
        cv2.fillPoly(img, [court_pts], (0, 100, 0))  # Green court
        
        # Draw lines
        if config:
            for start_name, end_name in config.court_lines:
                if start_name in self.keypoints and end_name in self.keypoints:
                    start = self.keypoints[start_name]
                    end = self.keypoints[end_name]
                    cv2.line(img, start, end, (255, 255, 255), 2)
        
        return img


# =============================================================================
# KEYPOINT STABILIZER (Based on CourtCheck)
# =============================================================================

class KeypointStabilizer:
    """
    Stabilizes keypoint detections using temporal averaging.
    Reduces jitter in court detection.
    """
    
    def __init__(self, keypoint_names: List[str], history_length: int = 10):
        self.keypoint_names = keypoint_names
        self.history = {name: deque(maxlen=history_length) for name in keypoint_names}
        
    def stabilize(self, keypoints: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[int, int]]:
        """Stabilize keypoints by averaging over history"""
        stabilized = {}
        
        for name, point in keypoints.items():
            if point is not None and not np.isnan(point).any():
                self.history[name].append(point)
                
            if len(self.history[name]) > 0:
                avg = np.mean(list(self.history[name]), axis=0)
                stabilized[name] = (int(avg[0]), int(avg[1]))
            else:
                stabilized[name] = None
                
        return stabilized
    
    def reset(self):
        """Reset all history"""
        for name in self.keypoint_names:
            self.history[name].clear()


# =============================================================================
# HOMOGRAPHY TRANSFORMER (Based on yastrebksv and CourtCheck)
# =============================================================================

class HomographyTransformer:
    """
    Computes and applies homography transformation between
    camera view and 2D court reference.
    """
    
    def __init__(self):
        self.matrix = None
        self.inverse_matrix = None
        
    def compute_homography(self, 
                          src_points: np.ndarray, 
                          dst_points: np.ndarray) -> bool:
        """
        Compute homography matrix from source to destination points.
        
        Args:
            src_points: 4 points from camera view [4, 2]
            dst_points: 4 points from 2D reference [4, 2]
            
        Returns:
            True if successful, False otherwise
        """
        if len(src_points) < 4 or len(dst_points) < 4:
            return False
            
        try:
            self.matrix = cv2.getPerspectiveTransform(
                src_points.astype(np.float32),
                dst_points.astype(np.float32)
            )
            self.inverse_matrix = cv2.getPerspectiveTransform(
                dst_points.astype(np.float32),
                src_points.astype(np.float32)
            )
            return True
        except Exception as e:
            print(f"Homography computation failed: {e}")
            return False
    
    def transform_point(self, point: Tuple[float, float]) -> Optional[Tuple[int, int]]:
        """Transform a single point from camera to 2D view"""
        if self.matrix is None:
            return None
            
        pt = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.matrix)
        return (int(transformed[0, 0, 0]), int(transformed[0, 0, 1]))
    
    def inverse_transform_point(self, point: Tuple[float, float]) -> Optional[Tuple[int, int]]:
        """Transform a single point from 2D view to camera"""
        if self.inverse_matrix is None:
            return None
            
        pt = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.inverse_matrix)
        return (int(transformed[0, 0, 0]), int(transformed[0, 0, 1]))
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform multiple points"""
        if self.matrix is None:
            return points
            
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(pts, self.matrix)
        return transformed.reshape(-1, 2)
    
    def warp_image(self, img: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
        """Warp entire image to 2D view"""
        if self.matrix is None:
            return np.zeros((*output_size[::-1], 3), dtype=np.uint8)
            
        return cv2.warpPerspective(img, self.matrix, output_size)


# =============================================================================
# BALL TRACKER (Based on TrackNet architecture)
# =============================================================================

class BallTracker:
    """
    Ball tracking using trajectory analysis.
    Simplified version of TrackNet for demonstration.
    Real implementation would use the neural network.
    """
    
    def __init__(self, trail_length: int = 10):
        self.trail_length = trail_length
        self.positions = deque(maxlen=trail_length * 3)
        self.velocities = deque(maxlen=trail_length)
        self.last_position = None
        self.confidence = 0.0
        
    def update(self, position: Optional[Tuple[int, int]], confidence: float = 1.0):
        """Update tracker with new ball position"""
        if position is not None:
            if self.last_position is not None:
                velocity = (
                    position[0] - self.last_position[0],
                    position[1] - self.last_position[1]
                )
                self.velocities.append(velocity)
            
            self.positions.append(position)
            self.last_position = position
            self.confidence = confidence
        else:
            # Predict position based on velocity
            if self.last_position and len(self.velocities) > 0:
                avg_vel = np.mean(list(self.velocities), axis=0)
                predicted = (
                    int(self.last_position[0] + avg_vel[0]),
                    int(self.last_position[1] + avg_vel[1])
                )
                self.positions.append(predicted)
                self.last_position = predicted
                self.confidence *= 0.9
    
    def get_trail(self) -> List[Tuple[int, int]]:
        """Get ball trail for visualization"""
        return list(self.positions)[-self.trail_length:]
    
    def get_position(self) -> Optional[Tuple[int, int]]:
        """Get current ball position"""
        return self.last_position
    
    def get_speed(self) -> float:
        """Calculate ball speed in pixels/frame"""
        if len(self.velocities) < 2:
            return 0.0
        recent_vel = list(self.velocities)[-5:]
        speeds = [np.linalg.norm(v) for v in recent_vel]
        return np.mean(speeds)
    
    def reset(self):
        """Reset tracker"""
        self.positions.clear()
        self.velocities.clear()
        self.last_position = None
        self.confidence = 0.0


# =============================================================================
# BOUNCE DETECTOR (Based on yastrebksv)
# =============================================================================

class BounceDetector:
    """
    Detects ball bounces based on trajectory analysis.
    Uses velocity sign change detection.
    """
    
    def __init__(self, threshold: float = 5.0, min_frames_between: int = 15):
        self.positions = deque(maxlen=20)
        self.threshold = threshold
        self.min_frames_between = min_frames_between
        self.last_bounce_frame = -100
        self.bounces = []
        
    def update(self, position: Optional[Tuple[int, int]], frame_num: int) -> Optional[Dict]:
        """
        Update detector and check for bounce.
        
        Returns bounce info if detected, None otherwise.
        """
        if position is None:
            return None
            
        self.positions.append({'pos': position, 'frame': frame_num})
        
        if len(self.positions) < 5:
            return None
            
        if frame_num - self.last_bounce_frame < self.min_frames_between:
            return None
        
        # Calculate vertical velocities
        recent = list(self.positions)[-5:]
        velocities = []
        for i in range(1, len(recent)):
            vy = recent[i]['pos'][1] - recent[i-1]['pos'][1]
            velocities.append(vy)
        
        # Detect sign change (downward to upward motion)
        for i in range(1, len(velocities)):
            if velocities[i-1] > self.threshold and velocities[i] < -self.threshold:
                bounce = {
                    'position': recent[i]['pos'],
                    'frame': frame_num,
                    'confidence': 0.85
                }
                self.bounces.append(bounce)
                self.last_bounce_frame = frame_num
                return bounce
        
        return None
    
    def check_in_bounds(self, position: Tuple[int, int], court_bounds: Dict) -> bool:
        """Check if bounce position is within court bounds"""
        x, y = position
        return (court_bounds['left'] < x < court_bounds['right'] and
                court_bounds['top'] < y < court_bounds['bottom'])
    
    def get_recent_bounces(self, count: int = 5) -> List[Dict]:
        """Get most recent bounces"""
        return self.bounces[-count:]
    
    def reset(self):
        """Reset detector"""
        self.positions.clear()
        self.bounces.clear()
        self.last_bounce_frame = -100


# =============================================================================
# PLAYER TRACKER
# =============================================================================

class PlayerTracker:
    """
    Simple player tracking using position history.
    Real implementation would use YOLO + ByteTrack.
    """
    
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.positions = deque(maxlen=100)
        self.current_position = None
        self.bbox = None
        self.total_distance = 0.0
        self.current_speed = 0.0
        
    def update(self, position: Tuple[int, int], bbox: Tuple[int, int, int, int] = None):
        """Update player position"""
        if self.current_position is not None:
            dist = np.linalg.norm(np.array(position) - np.array(self.current_position))
            self.total_distance += dist
            self.current_speed = dist
        
        self.positions.append(position)
        self.current_position = position
        self.bbox = bbox
    
    def get_trail(self, length: int = 20) -> List[Tuple[int, int]]:
        """Get position trail"""
        return list(self.positions)[-length:]


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class TennisAnalysisPipeline:
    """
    Complete tennis analysis pipeline.
    Processes video frames and generates annotated output.
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        # Initialize components
        self.court_reference = CourtReference(
            self.config.court_2d_width,
            self.config.court_2d_height
        )
        self.keypoint_stabilizer = KeypointStabilizer(
            self.config.keypoint_names,
            self.config.keypoint_history_length
        )
        self.homography = HomographyTransformer()
        self.ball_tracker = BallTracker(self.config.ball_trail_length)
        self.bounce_detector = BounceDetector(
            self.config.bounce_detection_threshold
        )
        self.players = {
            1: PlayerTracker(1),
            2: PlayerTracker(2)
        }
        
        # Detection models would be loaded here
        # self.court_model = load_court_detection_model()
        # self.ball_model = load_tracknet_model()
        # self.player_model = load_yolo_model()
        
        self.frame_count = 0
        
    def detect_court_keypoints(self, frame: np.ndarray) -> Dict[str, Tuple[int, int]]:
        """
        Detect court keypoints in frame.
        
        In production, this would use Detectron2 or similar model.
        Here we simulate detection for demonstration.
        """
        h, w = frame.shape[:2]
        
        # Simulated keypoint detection
        # Replace with actual model inference:
        # outputs = self.court_model(frame)
        # keypoints = outputs["instances"].pred_keypoints.cpu().numpy()[0]
        
        # Demo: estimate keypoints from frame geometry
        keypoints = {
            "BTL": (int(w * 0.1), int(h * 0.1)),
            "BTLI": (int(w * 0.2), int(h * 0.1)),
            "BTRI": (int(w * 0.8), int(h * 0.1)),
            "BTR": (int(w * 0.9), int(h * 0.1)),
            "ITL": (int(w * 0.2), int(h * 0.3)),
            "ITM": (int(w * 0.5), int(h * 0.3)),
            "ITR": (int(w * 0.8), int(h * 0.3)),
            "NL": (int(w * 0.1), int(h * 0.5)),
            "NM": (int(w * 0.5), int(h * 0.5)),
            "NR": (int(w * 0.9), int(h * 0.5)),
            "IBL": (int(w * 0.2), int(h * 0.7)),
            "IBM": (int(w * 0.5), int(h * 0.7)),
            "IBR": (int(w * 0.8), int(h * 0.7)),
            "BBL": (int(w * 0.1), int(h * 0.9)),
            "BBLI": (int(w * 0.2), int(h * 0.9)),
            "BBRI": (int(w * 0.8), int(h * 0.9)),
            "BBR": (int(w * 0.9), int(h * 0.9)),
        }
        
        return keypoints
    
    def detect_ball(self, frame: np.ndarray, prev_frames: List[np.ndarray] = None) -> Optional[Tuple[int, int]]:
        """
        Detect ball in frame using TrackNet-style approach.
        
        In production, this would use TrackNet model with 3 consecutive frames.
        """
        # Simulated ball detection
        # Replace with actual TrackNet inference:
        # x_pred, y_pred = detect_ball(self.ball_model, device, frame, prev_frame, prev_prev_frame)
        
        h, w = frame.shape[:2]
        
        # Demo: simple color-based ball detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Yellow/green tennis ball color range
        lower = np.array([25, 50, 50])
        upper = np.array([45, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the most circular contour
            best_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(best_contour) > 10:
                M = cv2.moments(best_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (cx, cy)
        
        return None
    
    def detect_players(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect players in frame.
        
        In production, this would use YOLO or Faster R-CNN.
        """
        # Simulated player detection
        # Replace with actual model:
        # results = self.player_model(frame)
        
        h, w = frame.shape[:2]
        
        # Demo: return placeholder player positions
        return [
            {'id': 1, 'position': (int(w * 0.5), int(h * 0.8)), 'bbox': (int(w * 0.45), int(h * 0.7), 50, 100)},
            {'id': 2, 'position': (int(w * 0.5), int(h * 0.2)), 'bbox': (int(w * 0.45), int(h * 0.1), 50, 100)},
        ]
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single frame.
        
        Returns:
            annotated_frame: Main frame with overlays
            court_2d: 2D court view
        """
        self.frame_count += 1
        
        # 1. Detect court keypoints
        raw_keypoints = self.detect_court_keypoints(frame)
        keypoints = self.keypoint_stabilizer.stabilize(raw_keypoints)
        
        # 2. Compute homography
        if all(keypoints.get(k) for k in ["BTL", "BTR", "BBL", "BBR"]):
            src_pts = np.array([
                keypoints["BTL"],
                keypoints["BTR"],
                keypoints["BBL"],
                keypoints["BBR"],
            ], dtype=np.float32)
            dst_pts = self.court_reference.get_corner_points()
            self.homography.compute_homography(src_pts, dst_pts)
        
        # 3. Detect ball
        ball_pos = self.detect_ball(frame)
        self.ball_tracker.update(ball_pos)
        
        # 4. Detect bounces
        bounce = self.bounce_detector.update(ball_pos, self.frame_count)
        
        # 5. Detect players
        players = self.detect_players(frame)
        for p in players:
            if p['id'] in self.players:
                self.players[p['id']].update(p['position'], p['bbox'])
        
        # 6. Draw annotations on main frame
        annotated = self.draw_main_frame(frame, keypoints, bounce)
        
        # 7. Generate 2D court view
        court_2d = self.draw_court_2d()
        
        return annotated, court_2d
    
    def draw_main_frame(self, frame: np.ndarray, 
                        keypoints: Dict, 
                        bounce: Optional[Dict] = None) -> np.ndarray:
        """Draw annotations on main frame"""
        annotated = frame.copy()
        
        # Draw court lines
        for start_name, end_name in self.config.court_lines:
            start = keypoints.get(start_name)
            end = keypoints.get(end_name)
            if start and end:
                cv2.line(annotated, start, end, (0, 255, 0), 2)
        
        # Draw keypoints
        for name, point in keypoints.items():
            if point:
                cv2.circle(annotated, point, 5, (0, 255, 255), -1)
        
        # Draw ball trail
        trail = self.ball_tracker.get_trail()
        for i, pos in enumerate(trail):
            alpha = (i + 1) / len(trail)
            radius = int(3 + alpha * 5)
            color = (0, int(255 * alpha), 255)
            cv2.circle(annotated, pos, radius, color, -1)
        
        # Draw current ball position
        ball_pos = self.ball_tracker.get_position()
        if ball_pos:
            cv2.circle(annotated, ball_pos, 10, (0, 255, 255), 2)
            
            # Draw speed
            speed = self.ball_tracker.get_speed()
            cv2.putText(annotated, f"{speed:.1f} px/f", 
                       (ball_pos[0] + 15, ball_pos[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw bounce markers
        for b in self.bounce_detector.get_recent_bounces():
            pos = b['position']
            cv2.circle(annotated, pos, 12, (0, 0, 255), 3)
            cv2.putText(annotated, "BOUNCE", (pos[0] - 30, pos[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw players
        for pid, tracker in self.players.items():
            if tracker.current_position:
                pos = tracker.current_position
                color = (0, 0, 255) if pid == 1 else (255, 0, 0)
                cv2.circle(annotated, pos, 15, color, -1)
                cv2.putText(annotated, f"P{pid}", (pos[0] - 10, pos[1] + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw info overlay
        cv2.putText(annotated, f"Frame: {self.frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, f"Bounces: {len(self.bounce_detector.bounces)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated
    
    def draw_court_2d(self) -> np.ndarray:
        """Draw 2D court view with transformed positions"""
        # Start with reference court
        court = self.court_reference.draw_court(config=self.config)
        
        # Transform and draw ball
        ball_pos = self.ball_tracker.get_position()
        if ball_pos:
            ball_2d = self.homography.transform_point(ball_pos)
            if ball_2d:
                cv2.circle(court, ball_2d, 6, (0, 255, 255), -1)
        
        # Transform and draw ball trail
        trail = self.ball_tracker.get_trail()
        for i, pos in enumerate(trail[:-1]):
            pos_2d = self.homography.transform_point(pos)
            if pos_2d:
                alpha = (i + 1) / len(trail)
                cv2.circle(court, pos_2d, int(2 + alpha * 3), (0, int(200 * alpha), 200), -1)
        
        # Transform and draw players
        for pid, tracker in self.players.items():
            if tracker.current_position:
                pos_2d = self.homography.transform_point(tracker.current_position)
                if pos_2d:
                    color = (0, 0, 255) if pid == 1 else (255, 0, 0)
                    cv2.circle(court, pos_2d, 8, color, -1)
        
        # Transform and draw bounces
        for b in self.bounce_detector.get_recent_bounces():
            pos_2d = self.homography.transform_point(b['position'])
            if pos_2d:
                cv2.circle(court, pos_2d, 8, (0, 0, 255), 2)
        
        return court
    
    def process_video(self, input_path: str, output_path: str):
        """
        Process entire video file.
        
        Args:
            input_path: Path to input MP4 file
            output_path: Path to output MP4 file
        """
        print(f"Processing: {input_path}")
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Calculate output size (main frame + 2D court side by side)
        court_height = height
        court_width = int(self.config.court_2d_width * (height / self.config.court_2d_height))
        output_width = width + court_width
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, height))
        
        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                annotated, court_2d = self.process_frame(frame)
                
                # Resize 2D court to match height
                court_2d_resized = cv2.resize(court_2d, (court_width, court_height))
                
                # Combine frames side by side
                combined = np.hstack([annotated, court_2d_resized])
                
                # Write frame
                out.write(combined)
                
                frame_idx += 1
                if frame_idx % 30 == 0:
                    print(f"Processed {frame_idx}/{total_frames} frames ({100*frame_idx/total_frames:.1f}%)")
                    
        finally:
            cap.release()
            out.release()
        
        print(f"Output saved to: {output_path}")
        print(f"Total bounces detected: {len(self.bounce_detector.bounces)}")
        
        # Save analysis data
        self.save_analysis(output_path.replace('.mp4', '_analysis.json'))
    
    def save_analysis(self, output_path: str):
        """Save analysis results to JSON"""
        data = {
            'total_frames': self.frame_count,
            'bounces': [
                {
                    'frame': b['frame'],
                    'position': list(b['position']),
                    'confidence': b['confidence']
                }
                for b in self.bounce_detector.bounces
            ],
            'players': {
                pid: {
                    'total_distance': tracker.total_distance,
                    'final_position': list(tracker.current_position) if tracker.current_position else None
                }
                for pid, tracker in self.players.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Analysis saved to: {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Tennis Video Analysis Pipeline')
    parser.add_argument('--input', '-i', required=True, help='Input video path (MP4)')
    parser.add_argument('--output', '-o', required=True, help='Output video path (MP4)')
    parser.add_argument('--config', '-c', help='Config JSON file (optional)')
    
    args = parser.parse_args()
    
    # Load config if provided
    config = Config()
    if args.config:
        with open(args.config) as f:
            config_data = json.load(f)
            # Update config with loaded values
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Create pipeline and process video
    pipeline = TennisAnalysisPipeline(config)
    pipeline.process_video(args.input, args.output)


if __name__ == '__main__':
    main()
