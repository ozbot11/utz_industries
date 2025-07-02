"""
Complete Human Overlay Tracking System
For 100% realistic AI character overlay on real person videos

Captures:
- Full body pose (17 keypoints)
- Precision hands (21 landmarks each)
- Face mesh (468 landmarks)
- 3D pose estimation
- Head orientation (pitch, yaw, roll)
- Eye gaze tracking
- Facial expressions
"""

import cv2
import numpy as np
import json
import time
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

@dataclass
class CompleteHumanFrame:
    """Complete human tracking data for realistic overlay"""
    timestamp: float
    frame_number: int
    
    # Body pose (17 COCO keypoints)
    body_keypoints_2d: Optional[List[float]] = None  # [x, y, confidence] * 17
    body_keypoints_3d: Optional[List[float]] = None
    
    # Hands (21 landmarks each)
    left_hand_2d: Optional[List[float]] = None   # [x, y, confidence] * 21
    right_hand_2d: Optional[List[float]] = None
    left_hand_3d: Optional[List[List[float]]] = None   # [[x, y, z]] * 21
    right_hand_3d: Optional[List[List[float]]] = None
    
    # Face (468 landmarks)
    face_landmarks_2d: Optional[List[float]] = None
    face_landmarks_3d: Optional[List[List[float]]] = None
    
    # 3D head orientation
    head_pose: Optional[Dict[str, float]] = None  # pitch, yaw, roll
    
    # Eye tracking
    left_eye_landmarks: Optional[List[float]] = None
    right_eye_landmarks: Optional[List[float]] = None
    gaze_direction: Optional[Dict[str, float]] = None
    
    # Facial expressions
    facial_expressions: Optional[Dict[str, float]] = None
    
    # Body 3D orientation
    body_orientation: Optional[Dict[str, float]] = None

class CompleteHumanTracker:
    """Complete tracking system for realistic human overlay"""
    
    def __init__(self):
        self.tracking_data = []
        self.frame_count = 0
        
        print("🎭 Complete Human Overlay Tracking System")
        print("=" * 55)
        
        # Initialize YOLOv8 for body tracking
        self.yolo_available = YOLO_AVAILABLE
        if self.yolo_available:
            print("🚀 Loading YOLOv8-Pose for body tracking...")
            try:
                self.yolo_model = YOLO('yolov8n-pose.pt')
                print("✅ YOLOv8-Pose loaded!")
            except Exception as e:
                print(f"❌ YOLOv8 failed: {e}")
                self.yolo_available = False
        
        # Initialize MediaPipe for face, hands, and 3D pose
        self.mediapipe_available = MEDIAPIPE_AVAILABLE
        if self.mediapipe_available:
            print("🖐️ Loading MediaPipe components...")
            
            # Face mesh for detailed facial tracking
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,  # High-quality landmarks
                min_detection_confidence=0.3,  # Lowered for better detection
                min_tracking_confidence=0.3    # Lowered for better detection
            )
            
            # Hands for precision finger tracking
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,  # Slightly lowered
                min_tracking_confidence=0.3,   # Lowered for better tracking
                model_complexity=1
            )
            
            # Pose for 3D body tracking (backup/enhancement)
            self.mp_pose = mp.solutions.pose
            self.pose_3d = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,  # Reduced for better performance
                enable_segmentation=False,
                min_detection_confidence=0.3,  # Lowered for better detection
                min_tracking_confidence=0.3    # Lowered for better detection
            )
            
            # Drawing utilities
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            print("✅ MediaPipe components loaded!")
            print("   • Face mesh (468 landmarks)")
            print("   • Precision hands (21 landmarks each)")
            print("   • 3D pose estimation")
            print("   • Head orientation tracking")
        
        # Camera matrix for 3D calculations (approximate)
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))
        
        print("\n🎯 Complete tracking capabilities:")
        print("   • Full body pose (17 keypoints)")
        print("   • Facial expressions (468 landmarks)")
        print("   • Precision hands (21 landmarks each)")
        print("   • 3D head orientation (pitch, yaw, roll)")
        print("   • Eye gaze tracking")
        print("   • Body 3D orientation")
        print("   • Enhanced leg detection")
        print("   • Ready for realistic character overlay!")
    
    def setup_camera_calibration(self, frame_width: int, frame_height: int):
        """Setup camera matrix for 3D calculations"""
        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)
        
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)
    
    def process_frame(self, frame) -> CompleteHumanFrame:
        """Process a frame and extract all human tracking data"""
        h, w = frame.shape[:2]
        
        if self.camera_matrix is None:
            self.setup_camera_calibration(w, h)
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Initialize frame data
        frame_data = CompleteHumanFrame(
            timestamp=time.time(),
            frame_number=self.frame_count
        )
        
        # 1. Body pose tracking with YOLOv8
        if self.yolo_available:
            frame_data = self._process_body_yolo(frame, frame_data)
        
        # 2. Face tracking with MediaPipe
        if self.mediapipe_available:
            frame_data = self._process_face(frame_rgb, frame_data, w, h)
            
            # 3. Hand tracking
            frame_data = self._process_hands(frame_rgb, frame_data, w, h)
            
            # 4. 3D pose enhancement
            frame_data = self._process_3d_pose(frame_rgb, frame_data, w, h)
        
        self.frame_count += 1
        return frame_data
    
    def _process_body_yolo(self, frame, frame_data: CompleteHumanFrame) -> CompleteHumanFrame:
        """Process body pose with YOLOv8"""
        results = self.yolo_model(frame, verbose=False)
        
        for result in results:
            if result.keypoints is not None:
                keypoints = result.keypoints.xy.cpu().numpy()
                confidences = result.keypoints.conf.cpu().numpy()
                
                if len(keypoints) > 0:
                    # Take first person detected
                    person_keypoints = keypoints[0]
                    person_confidences = confidences[0]
                    
                    # Convert to flat list [x, y, conf, x, y, conf, ...]
                    body_keypoints_2d = []
                    for i in range(17):
                        x, y = person_keypoints[i]
                        conf = person_confidences[i]
                        body_keypoints_2d.extend([float(x), float(y), float(conf)])
                    
                    frame_data.body_keypoints_2d = body_keypoints_2d
                    break
        
        return frame_data
    
    def _process_face(self, frame_rgb, frame_data: CompleteHumanFrame, w: int, h: int) -> CompleteHumanFrame:
        """Process facial landmarks and head pose"""
        try:
            face_results = self.face_mesh.process(frame_rgb)
            
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]
                
                # Extract 2D face landmarks
                face_2d = []
                face_3d_points = []
                face_3d_normalized = []
                
                for landmark in face_landmarks.landmark:
                    x = landmark.x * w
                    y = landmark.y * h
                    z = landmark.z
                    
                    face_2d.extend([x, y, 1.0])  # Add confidence
                    face_3d_points.append([x, y, z * w])  # Scale Z
                    face_3d_normalized.append([landmark.x, landmark.y, landmark.z])
                
                frame_data.face_landmarks_2d = face_2d
                frame_data.face_landmarks_3d = face_3d_normalized
                
                # Calculate head pose (pitch, yaw, roll)
                frame_data.head_pose = self._calculate_head_pose(face_3d_points, w, h)
                
                # Extract eye landmarks for gaze tracking
                frame_data = self._extract_eye_landmarks(face_landmarks, frame_data, w, h)
                
                # Calculate facial expressions
                frame_data.facial_expressions = self._calculate_facial_expressions(face_landmarks)
            
        except Exception as e:
            print(f"Face processing error: {e}")
        
        return frame_data
    
    def _process_hands(self, frame_rgb, frame_data: CompleteHumanFrame, w: int, h: int) -> CompleteHumanFrame:
        """Process hand landmarks"""
        hand_results = self.hands.process(frame_rgb)
        
        if hand_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                # Get hand classification
                hand_label = hand_results.multi_handedness[idx].classification[0].label
                hand_confidence = hand_results.multi_handedness[idx].classification[0].score
                
                # Extract landmarks
                landmarks_2d = []
                landmarks_3d = []
                
                for landmark in hand_landmarks.landmark:
                    # 2D coordinates
                    landmarks_2d.extend([
                        float(landmark.x * w),
                        float(landmark.y * h),
                        float(hand_confidence)
                    ])
                    
                    # 3D coordinates (normalized)
                    landmarks_3d.append([
                        float(landmark.x),
                        float(landmark.y),
                        float(landmark.z)
                    ])
                
                # Store based on hand type
                if hand_label == "Left":
                    frame_data.left_hand_2d = landmarks_2d
                    frame_data.left_hand_3d = landmarks_3d
                else:
                    frame_data.right_hand_2d = landmarks_2d
                    frame_data.right_hand_3d = landmarks_3d
        
        return frame_data
    
    def _process_3d_pose(self, frame_rgb, frame_data: CompleteHumanFrame, w: int, h: int) -> CompleteHumanFrame:
        """Process 3D pose for body orientation"""
        try:
            pose_results = self.pose_3d.process(frame_rgb)
            
            if pose_results.pose_landmarks:
                # Calculate body orientation from 3D landmarks
                landmarks_3d = []
                for landmark in pose_results.pose_landmarks.landmark:
                    landmarks_3d.append([landmark.x, landmark.y, landmark.z])
                
                frame_data.body_keypoints_3d = landmarks_3d
                frame_data.body_orientation = self._calculate_body_orientation(landmarks_3d)
        
        except Exception as e:
            print(f"3D pose processing error: {e}")
        
        return frame_data
    
    def _calculate_head_pose(self, face_3d_points: List[List[float]], w: int, h: int) -> Dict[str, float]:
        """Calculate head pose angles (pitch, yaw, roll)"""
        if len(face_3d_points) < 6:
            return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
        
        # Use specific facial landmarks for head pose
        nose_tip = np.array(face_3d_points[1], dtype=np.float32)
        chin = np.array(face_3d_points[175], dtype=np.float32) 
        left_eye = np.array(face_3d_points[33], dtype=np.float32)
        right_eye = np.array(face_3d_points[263], dtype=np.float32)
        left_mouth = np.array(face_3d_points[61], dtype=np.float32)
        right_mouth = np.array(face_3d_points[291], dtype=np.float32)
        
        # 3D model points
        model_points = np.array([
            [0.0, 0.0, 0.0],           # Nose tip
            [0.0, -330.0, -65.0],     # Chin
            [-225.0, 170.0, -135.0],  # Left eye corner
            [225.0, 170.0, -135.0],   # Right eye corner
            [-150.0, -150.0, -125.0], # Left mouth corner
            [150.0, -150.0, -125.0]   # Right mouth corner
        ], dtype=np.float32)
        
        # 2D image points
        image_points = np.array([
            nose_tip[:2],
            chin[:2], 
            left_eye[:2],
            right_eye[:2],
            left_mouth[:2],
            right_mouth[:2]
        ], dtype=np.float32)
        
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, self.camera_matrix, self.dist_coeffs
        )
        
        if success:
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            # Extract angles
            sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
            singular = sy < 1e-6
            
            if not singular:
                pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                yaw = math.atan2(-rotation_matrix[2, 0], sy)
                roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            else:
                pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                yaw = math.atan2(-rotation_matrix[2, 0], sy)
                roll = 0
            
            return {
                "pitch": float(math.degrees(pitch)),
                "yaw": float(math.degrees(yaw)),
                "roll": float(math.degrees(roll))
            }
        
        return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    
    def _extract_eye_landmarks(self, face_landmarks, frame_data: CompleteHumanFrame, w: int, h: int) -> CompleteHumanFrame:
        """Extract eye landmarks for gaze tracking"""
        # Left eye landmarks (6 points around eye)
        left_eye_indices = [33, 7, 163, 144, 145, 153]
        right_eye_indices = [362, 382, 381, 380, 374, 373]
        
        left_eye = []
        right_eye = []
        
        for idx in left_eye_indices:
            landmark = face_landmarks.landmark[idx]
            left_eye.extend([landmark.x * w, landmark.y * h, 1.0])
        
        for idx in right_eye_indices:
            landmark = face_landmarks.landmark[idx]
            right_eye.extend([landmark.x * w, landmark.y * h, 1.0])
        
        frame_data.left_eye_landmarks = left_eye
        frame_data.right_eye_landmarks = right_eye
        
        # Calculate gaze direction (simplified)
        if len(left_eye) >= 6 and len(right_eye) >= 6:
            left_center = np.mean([[left_eye[i], left_eye[i+1]] for i in range(0, len(left_eye), 3)], axis=0)
            right_center = np.mean([[right_eye[i], right_eye[i+1]] for i in range(0, len(right_eye), 3)], axis=0)
            
            # Simple gaze direction calculation
            eye_center = (left_center + right_center) / 2
            frame_center = np.array([w/2, h/2])
            gaze_vector = eye_center - frame_center
            
            frame_data.gaze_direction = {
                "x": float(gaze_vector[0] / w),
                "y": float(gaze_vector[1] / h)
            }
        
        return frame_data
    
    def _calculate_facial_expressions(self, face_landmarks) -> Dict[str, float]:
        """Calculate basic facial expressions"""
        landmarks = face_landmarks.landmark
        
        # Mouth landmarks for smile detection
        mouth_left = landmarks[61]
        mouth_right = landmarks[291]
        mouth_top = landmarks[13]
        mouth_bottom = landmarks[14]
        
        # Calculate mouth width vs height ratio for smile
        mouth_width = abs(mouth_right.x - mouth_left.x)
        mouth_height = abs(mouth_top.y - mouth_bottom.y)
        smile_ratio = mouth_width / mouth_height if mouth_height > 0 else 1.0
        
        # Eyebrow landmarks for expressions
        left_eyebrow = landmarks[70]
        right_eyebrow = landmarks[296]
        
        return {
            "smile": float(min(smile_ratio / 3.0, 1.0)),  # Normalize
            "eyebrow_raise": float(abs(left_eyebrow.y - right_eyebrow.y)),
            "mouth_open": float(mouth_height * 10),  # Scale up
        }
    
    def _calculate_body_orientation(self, landmarks_3d: List[List[float]]) -> Dict[str, float]:
        """Calculate body orientation from 3D pose landmarks"""
        if len(landmarks_3d) < 33:
            return {"body_yaw": 0.0, "body_pitch": 0.0, "body_roll": 0.0}
        
        # Use shoulder and hip landmarks to calculate body orientation
        left_shoulder = np.array(landmarks_3d[11])
        right_shoulder = np.array(landmarks_3d[12])
        left_hip = np.array(landmarks_3d[23])
        right_hip = np.array(landmarks_3d[24])
        
        # Calculate shoulder vector
        shoulder_vector = right_shoulder - left_shoulder
        hip_vector = right_hip - left_hip
        
        # Calculate angles
        body_yaw = float(math.degrees(math.atan2(shoulder_vector[0], shoulder_vector[2])))
        body_pitch = float(math.degrees(math.atan2(shoulder_vector[1], shoulder_vector[2])))
        
        # Calculate roll from shoulder-hip alignment
        torso_vector = (left_shoulder + right_shoulder) / 2 - (left_hip + right_hip) / 2
        body_roll = float(math.degrees(math.atan2(torso_vector[0], torso_vector[1])))
        
        return {
            "body_yaw": body_yaw,
            "body_pitch": body_pitch, 
            "body_roll": body_roll
        }
    
    def draw_complete_tracking(self, frame, frame_data: CompleteHumanFrame):
        """Draw all tracking data on frame with clear visual indicators"""
        # Draw body pose with enhanced leg visualization
        if frame_data.body_keypoints_2d and len(frame_data.body_keypoints_2d) > 0:
            self._draw_body_pose(frame, frame_data.body_keypoints_2d)
        
        # Draw hands with clear indicators
        if frame_data.left_hand_2d and len(frame_data.left_hand_2d) > 0:
            self._draw_precision_hand(frame, frame_data.left_hand_2d, (255, 100, 100), "LEFT")
            # Add clear indicator
            cv2.putText(frame, "LEFT HAND TRACKED", (300, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
            
        if frame_data.right_hand_2d and len(frame_data.right_hand_2d) > 0:
            self._draw_precision_hand(frame, frame_data.right_hand_2d, (100, 100, 255), "RIGHT")
            # Add clear indicator
            cv2.putText(frame, "RIGHT HAND TRACKED", (300, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)
        
        # Draw face landmarks (with clear indicators)
        if frame_data.face_landmarks_2d and len(frame_data.face_landmarks_2d) > 0:
            self._draw_face_outline(frame, frame_data.face_landmarks_2d)
        
        # Draw head pose info (with visual indicators)
        if frame_data.head_pose:
            self._draw_head_pose_info(frame, frame_data.head_pose)
        
        # Draw leg tracking status
        if frame_data.body_keypoints_2d and len(frame_data.body_keypoints_2d) > 0:
            # Check leg detection specifically
            leg_indices = [11, 12, 13, 14, 15, 16]  # Hip, knee, ankle for both legs
            leg_points_detected = 0
            
            for i in leg_indices:
                idx = i * 3  # Each keypoint has x, y, confidence
                if idx + 2 < len(frame_data.body_keypoints_2d):
                    conf = frame_data.body_keypoints_2d[idx + 2]
                    if conf > 0.3:
                        leg_points_detected += 1
            
            if leg_points_detected > 0:
                cv2.putText(frame, "LEG TRACKING ACTIVE", (300, 230), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Leg Points: {leg_points_detected}/6", (300, 260), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw 3D pose indicator
        if frame_data.body_keypoints_3d and len(frame_data.body_keypoints_3d) > 0:
            cv2.putText(frame, "3D POSE ACTIVE", (300, 290), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"3D Points: {len(frame_data.body_keypoints_3d)}", (300, 320), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw tracking info (ignore the broken status for now)
        self._draw_tracking_info(frame, frame_data)
        
        return frame
    
    def _draw_body_pose(self, frame, body_keypoints):
        """Draw body pose skeleton with enhanced leg visualization"""
        if len(body_keypoints) < 51:  # 17 * 3
            return
        
        # COCO pose connections
        arm_connections = [(5, 7), (7, 9), (6, 8), (8, 10)]  # Arms
        torso_connections = [(5, 6), (5, 11), (6, 12), (11, 12)]  # Torso
        leg_connections = [(11, 13), (13, 15), (12, 14), (14, 16)]  # Legs
        head_connections = [(0, 1), (0, 2), (1, 3), (2, 4)]  # Head
        
        # Extract points
        points = []
        for i in range(0, len(body_keypoints), 3):
            x, y, conf = body_keypoints[i:i+3]
            if conf > 0.3:
                points.append((int(x), int(y)))
            else:
                points.append(None)
        
        # Draw connections with different colors for different body parts
        # Draw legs in bright green to make them obvious
        for start_idx, end_idx in leg_connections:
            if start_idx < len(points) and end_idx < len(points):
                start_point = points[start_idx]
                end_point = points[end_idx]
                if start_point and end_point:
                    cv2.line(frame, start_point, end_point, (0, 255, 0), 4)  # Thick green for legs
        
        # Draw torso in blue
        for start_idx, end_idx in torso_connections:
            if start_idx < len(points) and end_idx < len(points):
                start_point = points[start_idx]
                end_point = points[end_idx]
                if start_point and end_point:
                    cv2.line(frame, start_point, end_point, (255, 0, 0), 3)  # Blue for torso
        
        # Draw arms in yellow
        for start_idx, end_idx in arm_connections:
            if start_idx < len(points) and end_idx < len(points):
                start_point = points[start_idx]
                end_point = points[end_idx]
                if start_point and end_point:
                    cv2.line(frame, start_point, end_point, (0, 255, 255), 3)  # Yellow for arms
        
        # Draw head in white
        for start_idx, end_idx in head_connections:
            if start_idx < len(points) and end_idx < len(points):
                start_point = points[start_idx]
                end_point = points[end_idx]
                if start_point and end_point:
                    cv2.line(frame, start_point, end_point, (255, 255, 255), 3)  # White for head
        
        # Draw keypoints with special emphasis on leg points
        coco_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        for i, point in enumerate(points):
            if point:
                # Make leg keypoints larger and more colorful
                if i in [11, 12, 13, 14, 15, 16]:  # Hip, knee, ankle indices
                    cv2.circle(frame, point, 8, (0, 255, 0), -1)  # Large green circles for legs
                    cv2.circle(frame, point, 10, (255, 255, 255), 2)  # White outline
                    
                    # Add labels for leg points
                    if i < len(coco_names):
                        label = coco_names[i].replace('_', ' ').title()
                        cv2.putText(frame, label, (point[0] + 10, point[1]), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                else:
                    cv2.circle(frame, point, 5, (0, 0, 255), -1)  # Regular red circles for other points
        
        # Count and display leg detection status
        leg_points = [points[i] for i in [11, 12, 13, 14, 15, 16] if i < len(points)]
        detected_legs = sum(1 for p in leg_points if p is not None)
        
        if detected_legs > 0:
            cv2.putText(frame, f"LEGS DETECTED: {detected_legs}/6 points", (300, 170), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show which specific leg parts are detected
            leg_parts = ["L.Hip", "R.Hip", "L.Knee", "R.Knee", "L.Ankle", "R.Ankle"]
            detected_parts = []
            for i, part in enumerate(leg_parts):
                if (11 + i) < len(points) and points[11 + i] is not None:
                    detected_parts.append(part)
            
            if detected_parts:
                parts_text = ", ".join(detected_parts)
                cv2.putText(frame, f"Parts: {parts_text}", (300, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def _draw_precision_hand(self, frame, hand_keypoints, color, hand_label):
        """Draw precision hand with connections"""
        if len(hand_keypoints) < 63:  # 21 * 3
            return
        
        # Extract points
        points = []
        for i in range(0, len(hand_keypoints), 3):
            x, y, conf = hand_keypoints[i:i+3]
            if conf > 0.5:
                points.append((int(x), int(y)))
            else:
                points.append(None)
        
        # Hand connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
        ]
        
        # Draw connections
        for start_idx, end_idx in connections:
            if start_idx < len(points) and end_idx < len(points):
                start_point = points[start_idx]
                end_point = points[end_idx]
                if start_point and end_point:
                    cv2.line(frame, start_point, end_point, color, 2)
        
        # Draw keypoints
        for i, point in enumerate(points):
            if point:
                if i in [4, 8, 12, 16, 20]:  # Fingertips
                    cv2.circle(frame, point, 6, (0, 255, 0), -1)
                else:
                    cv2.circle(frame, point, 4, color, -1)
        
        # Add label
        if points[0]:
            cv2.putText(frame, f"{hand_label} HAND", 
                       (points[0][0] - 50, points[0][1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def _draw_face_outline(self, frame, face_landmarks):
        """Draw simplified face outline with clear visual indicators"""
        if len(face_landmarks) < 100:  # At least some face landmarks
            return
        
        # Draw key facial feature points to make it obvious face is detected
        # Draw every 15th landmark as larger circles to show face detection clearly
        for i in range(0, min(len(face_landmarks), 1400), 45):  # Every 15th landmark * 3
            if i + 1 < len(face_landmarks):
                x, y = int(face_landmarks[i]), int(face_landmarks[i+1])
                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)  # Bright cyan circles
        
        # Add a clear "FACE DETECTED" indicator
        if len(face_landmarks) > 300:
            cv2.putText(frame, "FACE DETECTED", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Landmarks: {len(face_landmarks)//3}", (50, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def _draw_head_pose_info(self, frame, head_pose):
        """Draw head pose information with clear visual indicator"""
        if not head_pose:
            return
            
        y_start = 160
        
        # Draw a clear "HEAD TRACKING" indicator
        cv2.putText(frame, "HEAD TRACKING ACTIVE", (50, y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        cv2.putText(frame, f"Pitch: {head_pose['pitch']:.1f}°", 
                   (50, y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Yaw: {head_pose['yaw']:.1f}°", 
                   (50, y_start + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Roll: {head_pose['roll']:.1f}°", 
                   (50, y_start + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw a visual indicator showing head orientation
        center_x, center_y = 200, y_start + 40
        
        # Draw head orientation arrow based on yaw
        yaw_rad = np.radians(head_pose['yaw'])
        arrow_length = 30
        end_x = int(center_x + arrow_length * np.sin(yaw_rad))
        end_y = int(center_y - arrow_length * np.cos(yaw_rad))
        
        cv2.circle(frame, (center_x, center_y), 25, (255, 0, 255), 2)
        cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), (255, 0, 255), 3)
    
    def _draw_tracking_info(self, frame, frame_data: CompleteHumanFrame):
        """Draw comprehensive tracking information"""
        # Check actual detection status properly
        body_detected = frame_data.body_keypoints_2d is not None and len(frame_data.body_keypoints_2d) > 0
        face_detected = frame_data.face_landmarks_2d is not None and len(frame_data.face_landmarks_2d) > 0
        left_hand_detected = frame_data.left_hand_2d is not None and len(frame_data.left_hand_2d) > 0
        right_hand_detected = frame_data.right_hand_2d is not None and len(frame_data.right_hand_2d) > 0
        head_pose_detected = frame_data.head_pose is not None
        pose_3d_detected = frame_data.body_keypoints_3d is not None and len(frame_data.body_keypoints_3d) > 0
        
        info_lines = [
            "Complete Human Overlay Tracker",
            f"Frame: {frame_data.frame_number}",
            f"Body: {'✓' if body_detected else '✗'}",
            f"Face: {'✓' if face_detected else '✗'}",
            f"L.Hand: {'✓' if left_hand_detected else '✗'}",
            f"R.Hand: {'✓' if right_hand_detected else '✗'}",
            f"Head Pose: {'✓' if head_pose_detected else '✗'}",
            f"3D Data: {'✓' if pose_3d_detected else '✗'}"
        ]
        
        # Add detailed detection counts
        detail_lines = [
            "",
            "DETECTION DETAILS:",
        ]
        
        if body_detected:
            detail_lines.append(f"Body keypoints: {len(frame_data.body_keypoints_2d)//3}")
        if face_detected:
            detail_lines.append(f"Face landmarks: {len(frame_data.face_landmarks_2d)//3}")
        if left_hand_detected:
            detail_lines.append(f"Left hand: {len(frame_data.left_hand_2d)//3} landmarks")
        if right_hand_detected:
            detail_lines.append(f"Right hand: {len(frame_data.right_hand_2d)//3} landmarks")
        if head_pose_detected:
            detail_lines.append(f"Head angles: P{frame_data.head_pose['pitch']:.1f}° Y{frame_data.head_pose['yaw']:.1f}° R{frame_data.head_pose['roll']:.1f}°")
        if pose_3d_detected:
            detail_lines.append(f"3D landmarks: {len(frame_data.body_keypoints_3d)}")
        
        all_lines = info_lines + detail_lines
        
        for i, line in enumerate(all_lines):
            if line == "":
                continue
            color = (0, 255, 0) if '✓' in line else (255, 255, 255)
            if "DETECTION DETAILS" in line:
                color = (255, 255, 0)
            elif any(x in line for x in ["keypoints:", "landmarks:", "angles:", "3D landmarks:"]):
                color = (0, 255, 255)  # Cyan for detail info
            
            cv2.putText(frame, line, (10, 30 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def live_tracking(self, camera_index=0):
        """Live complete human tracking"""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_index}")
            return
        
        print(f"🎥 Starting complete human tracking")
        print("Press 'q' to quit, 's' to save data")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            frame_data = self.process_frame(frame)
            
            # Draw all tracking
            frame = self.draw_complete_tracking(frame, frame_data)
            
            # Save data
            self.tracking_data.append(frame_data)
            
            cv2.imshow('Complete Human Overlay Tracker', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_complete_data()
        
        cap.release()
        cv2.destroyAllWindows()
    
    def save_complete_data(self):
        """Save complete tracking data"""
        if not self.tracking_data:
            print("❌ No data to save!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"complete_human_tracking_{timestamp}.json"
        
        try:
            # Convert to dict for JSON serialization
            data_dict = [asdict(frame) for frame in self.tracking_data]
            
            with open(filename, 'w') as f:
                json.dump(data_dict, f, indent=2)
            
            print(f"✅ Complete tracking data saved: {filename}")
            print(f"📊 {len(self.tracking_data)} frames with full human tracking")
            
        except Exception as e:
            print(f"❌ Error saving: {e}")

def main():
    """Main function"""
    print("Complete Human Overlay Tracking System")
    print("=" * 50)
    
    if not YOLO_AVAILABLE or not MEDIAPIPE_AVAILABLE:
        print("❌ Missing required libraries!")
        print("Install with:")
        print("  pip install ultralytics mediapipe opencv-python")
        return
    
    tracker = CompleteHumanTracker()
    
    print("\nChoose an option:")
    print("1. Live complete tracking")
    print("2. Exit")
    
    choice = input("\nEnter choice (1-2): ").strip()
    
    if choice == "1":
        camera_index = input("Enter camera index (0 for default): ").strip()
        try:
            camera_index = int(camera_index) if camera_index else 0
        except:
            camera_index = 0
        tracker.live_tracking(camera_index)
    
    elif choice == "2":
        print("Goodbye!")
        return
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()