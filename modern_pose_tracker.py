"""
YOLOv8-Pose Body Tracker
Modern, easy-to-install alternative to OpenPose with same accuracy
Perfect for skeleton overlay systems!
"""

import cv2
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
import sys

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

class ModernPoseTracker:
    def __init__(self, use_yolo=True, enable_precision_hands=True):
        """Initialize pose tracker with best available backend + precision hand tracking"""
        self.use_yolo = use_yolo and YOLO_AVAILABLE
        self.enable_precision_hands = enable_precision_hands and MEDIAPIPE_AVAILABLE
        self.tracking_data = []
        
        if self.use_yolo:
            print("🚀 Initializing YOLOv8-Pose (OpenPose-level accuracy)")
            try:
                self.model = YOLO('yolov8n-pose.pt')  # Will auto-download
                self.backend = "YOLOv8-Pose"
                print("✅ YOLOv8-Pose loaded successfully!")
            except Exception as e:
                print(f"❌ YOLOv8 failed: {e}")
                self.use_yolo = False
        
        # Initialize precision hand tracking (even if using YOLO for body)
        if self.enable_precision_hands:
            print("🖐️ Initializing MediaPipe Precision Hand Tracking")
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
                model_complexity=1  # Full precision like your original
            )
            print("✅ Precision hand tracking enabled!")
            print("   • 21 landmarks per hand")
            print("   • 3D coordinates (X, Y, Z)")
            print("   • Left/right classification")
        
        if not self.use_yolo and MEDIAPIPE_AVAILABLE:
            print("🔄 Using MediaPipe for full body pose")
            self.mp_pose = mp.solutions.pose
            self.mp_face = mp.solutions.face_mesh
            
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.backend = "MediaPipe"
            print("✅ MediaPipe full body loaded successfully!")
        
        elif not self.use_yolo:
            print("❌ No pose detection libraries available!")
            print("Install with: pip install ultralytics")
            sys.exit(1)
        
        # Set final backend description
        if self.use_yolo and self.enable_precision_hands:
            self.backend = "YOLOv8-Pose + MediaPipe Hands"
        elif self.use_yolo:
            self.backend = "YOLOv8-Pose"
        else:
            self.backend = "MediaPipe"
    
    def process_frame(self, frame):
        """Process a single frame and return keypoints with precision hands"""
        if self.use_yolo:
            return self._process_with_yolo_plus_hands(frame)
        else:
            return self._process_with_mediapipe(frame)
    
    def _process_with_yolo_plus_hands(self, frame):
        """Process frame with YOLOv8-Pose for body + MediaPipe for precision hands"""
        # First get body pose from YOLO
        results = self.model(frame, verbose=False)
        
        keypoints_data = {
            'timestamp': datetime.now().isoformat(),
            'backend': 'YOLOv8-Pose + MediaPipe Hands',
            'people': []
        }
        
        # Process YOLO body detection
        for result in results:
            if result.keypoints is not None:
                keypoints = result.keypoints.xy.cpu().numpy()  # Shape: [N, 17, 2]
                confidences = result.keypoints.conf.cpu().numpy()  # Shape: [N, 17]
                
                for person_idx in range(len(keypoints)):
                    person_keypoints = keypoints[person_idx]
                    person_confidences = confidences[person_idx]
                    
                    # Convert to OpenPose format for compatibility
                    pose_keypoints_2d = []
                    for i in range(17):  # COCO format has 17 keypoints
                        x, y = person_keypoints[i]
                        conf = person_confidences[i]
                        pose_keypoints_2d.extend([float(x), float(y), float(conf)])
                    
                    person_data = {
                        'pose_keypoints_2d': pose_keypoints_2d,
                        'pose_keypoints_count': 17,
                        'body_detection_method': 'YOLOv8-Pose'
                    }
                    
                    keypoints_data['people'].append(person_data)
        
        # Now add precision hand tracking with MediaPipe
        if self.enable_precision_hands:
            hand_data = self._get_precision_hands(frame)
            
            # If we have people detected, add hands to the first person
            # (In practice, you might want more sophisticated hand-to-person matching)
            if keypoints_data['people'] and hand_data:
                person = keypoints_data['people'][0]  # Add to first detected person
                
                if hand_data.get('left_hand'):
                    person['hand_left_keypoints_2d'] = hand_data['left_hand']
                    person['hand_left_confidence'] = hand_data.get('left_confidence', 1.0)
                
                if hand_data.get('right_hand'):
                    person['hand_right_keypoints_2d'] = hand_data['right_hand']
                    person['hand_right_confidence'] = hand_data.get('right_confidence', 1.0)
                
                person['hand_detection_method'] = 'MediaPipe Precision'
                person['hand_landmarks_count'] = '21 per hand'
            
            # If no people detected by YOLO but hands detected, create a person entry
            elif hand_data and (hand_data.get('left_hand') or hand_data.get('right_hand')):
                person_data = {
                    'pose_keypoints_2d': [],
                    'pose_keypoints_count': 0,
                    'body_detection_method': 'None',
                    'hand_detection_method': 'MediaPipe Precision',
                    'hand_landmarks_count': '21 per hand'
                }
                
                if hand_data.get('left_hand'):
                    person_data['hand_left_keypoints_2d'] = hand_data['left_hand']
                    person_data['hand_left_confidence'] = hand_data.get('left_confidence', 1.0)
                
                if hand_data.get('right_hand'):
                    person_data['hand_right_keypoints_2d'] = hand_data['right_hand']
                    person_data['hand_right_confidence'] = hand_data.get('right_confidence', 1.0)
                
                keypoints_data['people'].append(person_data)
        
        return keypoints_data
    
    def _get_precision_hands(self, frame):
        """Get precision hand data using MediaPipe (from your original code)"""
        if not self.enable_precision_hands:
            return {}
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hands with full MediaPipe precision
        results = self.hands.process(frame_rgb)
        
        hand_data = {}
        
        if results.multi_hand_landmarks:
            h, w = frame.shape[:2]
            
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get hand classification with confidence
                hand_info = results.multi_handedness[idx].classification[0]
                hand_label = hand_info.label
                hand_confidence = hand_info.score
                
                # Extract all 21 landmarks with full precision (like your original)
                landmarks_2d = []
                landmarks_3d = []
                for landmark in hand_landmarks.landmark:
                    # 2D coordinates for compatibility
                    landmarks_2d.extend([
                        float(landmark.x * w),  # Pixel X coordinate
                        float(landmark.y * h),  # Pixel Y coordinate
                        float(hand_confidence)  # Use hand confidence as landmark confidence
                    ])
                    
                    # Store 3D data separately for precision analysis
                    landmarks_3d.append([
                        float(landmark.x),  # Normalized X coordinate
                        float(landmark.y),  # Normalized Y coordinate  
                        float(landmark.z)   # Relative Z depth
                    ])
                
                # Store based on hand type (matching your original format)
                if hand_label == "Left":
                    hand_data['left_hand'] = landmarks_2d
                    hand_data['left_hand_3d'] = landmarks_3d
                    hand_data['left_confidence'] = hand_confidence
                else:
                    hand_data['right_hand'] = landmarks_2d
                    hand_data['right_hand_3d'] = landmarks_3d
                    hand_data['right_confidence'] = hand_confidence
        
        return hand_data
    
    def _process_with_mediapipe(self, frame):
        """Process frame with MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process pose
        pose_results = self.pose.process(rgb_frame)
        hands_results = self.hands.process(rgb_frame)
        
        keypoints_data = {
            'timestamp': datetime.now().isoformat(),
            'backend': 'MediaPipe',
            'people': []
        }
        
        if pose_results.pose_landmarks:
            h, w = frame.shape[:2]
            
            # Convert pose landmarks
            pose_keypoints_2d = []
            for landmark in pose_results.pose_landmarks.landmark:
                x = landmark.x * w
                y = landmark.y * h
                conf = landmark.visibility
                pose_keypoints_2d.extend([x, y, conf])
            
            person_data = {
                'pose_keypoints_2d': pose_keypoints_2d,
                'pose_keypoints_count': 33  # MediaPipe has 33 pose landmarks
            }
            
            # Add hand keypoints if detected
            if hands_results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                    hand_keypoints = []
                    for landmark in hand_landmarks.landmark:
                        x = landmark.x * w
                        y = landmark.y * h
                        conf = 1.0  # MediaPipe doesn't provide confidence for hands
                        hand_keypoints.extend([x, y, conf])
                    
                    # Determine if left or right hand
                    hand_label = hands_results.multi_handedness[hand_idx].classification[0].label
                    if hand_label == 'Left':
                        person_data['hand_left_keypoints_2d'] = hand_keypoints
                    else:
                        person_data['hand_right_keypoints_2d'] = hand_keypoints
            
            keypoints_data['people'].append(person_data)
        
        return keypoints_data
    
    def draw_skeleton(self, frame, keypoints_data):
        """Draw skeleton overlay on frame"""
        if not keypoints_data or 'people' not in keypoints_data:
            return frame
        
        for person in keypoints_data['people']:
            if 'pose_keypoints_2d' in person:
                pose_keypoints = person['pose_keypoints_2d']
                
                if self.use_yolo:
                    frame = self._draw_yolo_skeleton(frame, pose_keypoints)
                else:
                    frame = self._draw_mediapipe_skeleton(frame, pose_keypoints)
                
                # Draw hands if available (with precision visualization)
                for hand_key in ['hand_left_keypoints_2d', 'hand_right_keypoints_2d']:
                    if hand_key in person:
                        hand_keypoints = person[hand_key]
                        color = (255, 100, 100) if 'left' in hand_key else (100, 100, 255)
                        frame = self._draw_precision_hand_keypoints(frame, hand_keypoints, color, hand_key)
        
        return frame
    
    def _draw_yolo_skeleton(self, frame, pose_keypoints):
        """Draw YOLO/COCO skeleton (17 keypoints)"""
        # COCO pose connections
        connections = [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
            (0, 1), (0, 2), (1, 3), (2, 4)  # Head
        ]
        
        # Extract points
        points = []
        for i in range(0, len(pose_keypoints), 3):
            x, y, conf = pose_keypoints[i:i+3]
            if conf > 0.3:
                points.append((int(x), int(y)))
            else:
                points.append(None)
        
        # Draw connections
        for start_idx, end_idx in connections:
            if start_idx < len(points) and end_idx < len(points):
                start_point = points[start_idx]
                end_point = points[end_idx]
                if start_point and end_point:
                    cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        # Draw keypoints
        for point in points:
            if point:
                cv2.circle(frame, point, 4, (0, 0, 255), -1)
        
        return frame
    
    def _draw_mediapipe_skeleton(self, frame, pose_keypoints):
        """Draw MediaPipe skeleton (33 keypoints)"""
        # Draw all keypoints
        for i in range(0, len(pose_keypoints), 3):
            x, y, conf = pose_keypoints[i:i+3]
            if conf > 0.5:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
        
        return frame
    
    def _draw_precision_hand_keypoints(self, frame, hand_keypoints, color, hand_key):
        """Draw precision hand keypoints with connections (like your original)"""
        if len(hand_keypoints) < 63:  # 21 landmarks * 3 values = 63
            return frame
        
        # Extract points
        points = []
        for i in range(0, len(hand_keypoints), 3):
            x, y, conf = hand_keypoints[i:i+3]
            if conf > 0.5:
                points.append((int(x), int(y)))
            else:
                points.append(None)
        
        # MediaPipe hand connections (21 landmarks)
        hand_connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger  
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]
        
        # Draw connections
        for start_idx, end_idx in hand_connections:
            if start_idx < len(points) and end_idx < len(points):
                start_point = points[start_idx]
                end_point = points[end_idx]
                if start_point and end_point:
                    cv2.line(frame, start_point, end_point, color, 2)
        
        # Draw keypoints with different sizes for different finger parts
        for i, point in enumerate(points):
            if point:
                # Make fingertips bigger
                if i in [4, 8, 12, 16, 20]:  # Fingertips
                    cv2.circle(frame, point, 6, (0, 255, 0), -1)
                    cv2.circle(frame, point, 8, color, 2)
                else:
                    cv2.circle(frame, point, 4, color, -1)
        
        # Add hand label and confidence if available
        if points[0]:  # Wrist position
            hand_label = "LEFT" if 'left' in hand_key else "RIGHT"
            cv2.putText(frame, f"{hand_label} PRECISION", 
                       (points[0][0] - 50, points[0][1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def live_tracking(self, camera_index=0, save_data=True):
        """Live webcam tracking"""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_index}")
            return
        
        print(f"🎥 Starting live tracking with {self.backend}")
        print("Press 'q' to quit, 's' to save data, 'r' to reset data")
        
        frame_count = 0
        fps_counter = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                keypoints = self.process_frame(frame)
                
                # Draw skeleton
                frame = self.draw_skeleton(frame, keypoints)
                
                # Save data if enabled
                if save_data and keypoints and keypoints['people']:
                    self.tracking_data.append(keypoints)
                
                # Calculate FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - fps_counter)
                    fps_counter = time.time()
                    print(f"FPS: {fps:.1f} | People detected: {len(keypoints.get('people', []))}")
                
                # Add info overlay
                info_lines = [
                    f"{self.backend} - Press 'q' to quit",
                    f"People: {len(keypoints.get('people', []))}",
                ]
                
                # Add hand detection info
                if keypoints and keypoints.get('people'):
                    person = keypoints['people'][0]
                    hand_info = []
                    if 'hand_left_keypoints_2d' in person:
                        hand_info.append("Left Hand: ✓")
                    if 'hand_right_keypoints_2d' in person:
                        hand_info.append("Right Hand: ✓")
                    if hand_info:
                        info_lines.append(" | ".join(hand_info))
                    else:
                        info_lines.append("Hands: None detected")
                
                for i, line in enumerate(info_lines):
                    cv2.putText(frame, line, (10, 30 + i * 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Modern Pose Tracker + Precision Hands', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_tracking_data()
                elif key == ord('r'):
                    self.tracking_data = []
                    print("📊 Tracking data reset")
                
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            if save_data and self.tracking_data:
                self.save_tracking_data()
    
    def save_tracking_data(self, filename=None):
        """Save tracking data to JSON file"""
        if not self.tracking_data:
            print("📊 No tracking data to save")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pose_tracking_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.tracking_data, f, indent=2)
            
            print(f"💾 Saved {len(self.tracking_data)} frames to {filename}")
            
        except Exception as e:
            print(f"❌ Error saving data: {e}")
    
    def process_video(self, video_path, output_path=None):
        """Process a video file"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"🎬 Processing video with {self.backend}")
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            keypoints = self.process_frame(frame)
            frame = self.draw_skeleton(frame, keypoints)
            
            # Save tracking data
            if keypoints and keypoints['people']:
                self.tracking_data.append(keypoints)
            
            if output_path:
                out.write(frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        cap.release()
        if output_path:
            out.release()
            print(f"💾 Saved processed video to {output_path}")
        
        # Save tracking data
        self.save_tracking_data()

def main():
    """Main function"""
    print("Modern Pose Tracker")
    print("=" * 50)
    
    # Check available backends
    if not YOLO_AVAILABLE and not MEDIAPIPE_AVAILABLE:
        print("❌ No pose detection libraries found!")
        print("Install with:")
        print("  pip install ultralytics  # For YOLOv8-Pose")
        print("  pip install mediapipe    # For MediaPipe")
        return
    
    print("Available backends:")
    if YOLO_AVAILABLE:
        print("✅ YOLOv8-Pose (Recommended - OpenPose accuracy)")
    if MEDIAPIPE_AVAILABLE:
        print("✅ MediaPipe (Google's solution)")
    
    # Initialize tracker with precision hands enabled
    tracker = ModernPoseTracker(use_yolo=True, enable_precision_hands=True)
    
    print("\nChoose an option:")
    print("1. Live webcam tracking")
    print("2. Process video file")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        camera_index = input("Enter camera index (0 for default): ").strip()
        try:
            camera_index = int(camera_index) if camera_index else 0
        except:
            camera_index = 0
        tracker.live_tracking(camera_index)
    
    elif choice == "2":
        video_path = input("Enter path to video file: ").strip()
        output_path = input("Enter output path (or press Enter to skip): ").strip()
        output_path = output_path if output_path else None
        tracker.process_video(video_path, output_path)
    
    elif choice == "3":
        print("Goodbye!")
        return
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()