"""
Realistic Human Tracking Overlay
Creates a realistic human character that follows your exact movements
"""

import cv2
import numpy as np
import json
import time
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

# Import our existing systems
from live_skeleton_integration import LiveSkeletonSystem
from realistic_character_creator import CharacterProfile, FacialFeatures, BodyFeatures, HairFeatures, ClothingFeatures

class RealisticHumanRenderer:
    """Creates realistic human overlays that track your movements"""
    
    def __init__(self):
        print("👤 Initializing Realistic Human Renderer...")
        self.character_cache = {}
        print("✅ Realistic human renderer ready!")
    
    def create_realistic_human_parts(self, character: CharacterProfile, frame_shape: Tuple[int, int]) -> Dict:
        """Create realistic human body parts"""
        h, w = frame_shape[:2]
        
        # Create character parts with proper proportions
        parts = {
            'head': self._create_realistic_head(character, w, h),
            'hair': self._create_realistic_hair(character, w, h),
            'eyes': self._create_realistic_eyes(character, w, h),
            'torso': self._create_realistic_torso(character, w, h),
            'arms': self._create_realistic_arms(character, w, h),
            'legs': self._create_realistic_legs(character, w, h),
            'hands': self._create_realistic_hands(character, w, h)
        }
        
        return parts
    
    def _create_realistic_head(self, char: CharacterProfile, w: int, h: int) -> Dict:
        """Create realistic head shape"""
        base_size = min(w, h) // 8  # Proportional to frame size
        head_width = int(base_size * (0.8 + char.facial.face_width * 0.4))
        head_height = int(base_size * (0.9 + char.facial.face_length * 0.3))
        
        # Skin color
        skin = char.facial.skin_tone
        skin_bgr = (int(skin[2] * 255), int(skin[1] * 255), int(skin[0] * 255))
        
        return {
            'width': head_width,
            'height': head_height,
            'color': skin_bgr,
            'jawline': char.facial.jawline_definition
        }
    
    def _create_realistic_hair(self, char: CharacterProfile, w: int, h: int) -> Dict:
        """Create realistic hair"""
        hair_color = char.hair.color
        hair_bgr = (int(hair_color[2] * 255), int(hair_color[1] * 255), int(hair_color[0] * 255))
        
        base_size = min(w, h) // 8
        hair_size = int(base_size * (1.1 + char.hair.length * 0.5))
        
        return {
            'size': hair_size,
            'color': hair_bgr,
            'style': char.hair.style,
            'length': char.hair.length,
            'volume': char.hair.volume
        }
    
    def _create_realistic_eyes(self, char: CharacterProfile, w: int, h: int) -> Dict:
        """Create realistic eyes"""
        eye_color = char.facial.eye_color
        eye_bgr = (int(eye_color[2] * 255), int(eye_color[1] * 255), int(eye_color[0] * 255))
        
        base_size = min(w, h) // 40
        eye_size = int(base_size * (0.8 + char.facial.eye_size * 0.4))
        eye_spacing = int(base_size * (2 + char.facial.eye_spacing * 2))
        
        return {
            'size': eye_size,
            'spacing': eye_spacing,
            'color': eye_bgr,
            'shape': char.facial.eye_shape
        }
    
    def _create_realistic_torso(self, char: CharacterProfile, w: int, h: int) -> Dict:
        """Create realistic torso"""
        shirt_color = char.clothing.shirt_color
        shirt_bgr = (int(shirt_color[2] * 255), int(shirt_color[1] * 255), int(shirt_color[0] * 255))
        
        base_width = min(w, h) // 6
        torso_width = int(base_width * (0.8 + char.body.shoulder_width * 0.4))
        
        return {
            'width': torso_width,
            'color': shirt_bgr,
            'type': char.clothing.shirt_type,
            'muscle_definition': char.body.muscle_definition
        }
    
    def _create_realistic_arms(self, char: CharacterProfile, w: int, h: int) -> Dict:
        """Create realistic arms"""
        skin = char.facial.skin_tone
        skin_bgr = (int(skin[2] * 255), int(skin[1] * 255), int(skin[0] * 255))
        
        base_size = min(w, h) // 20
        arm_thickness = int(base_size * (0.8 + char.body.muscle_definition * 0.4))
        
        return {
            'thickness': arm_thickness,
            'color': skin_bgr,
            'muscle_definition': char.body.muscle_definition
        }
    
    def _create_realistic_legs(self, char: CharacterProfile, w: int, h: int) -> Dict:
        """Create realistic legs"""
        pants_color = char.clothing.pants_color
        pants_bgr = (int(pants_color[2] * 255), int(pants_color[1] * 255), int(pants_color[0] * 255))
        
        base_size = min(w, h) // 15
        leg_thickness = int(base_size * (0.8 + char.body.muscle_definition * 0.3))
        
        return {
            'thickness': leg_thickness,
            'color': pants_bgr,
            'type': char.clothing.pants_type
        }
    
    def _create_realistic_hands(self, char: CharacterProfile, w: int, h: int) -> Dict:
        """Create realistic hands"""
        skin = char.facial.skin_tone
        skin_bgr = (int(skin[2] * 255), int(skin[1] * 255), int(skin[0] * 255))
        
        base_size = min(w, h) // 30
        hand_size = int(base_size * (0.8 + char.body.height / 2.0))
        
        return {
            'size': hand_size,
            'color': skin_bgr
        }

class HumanTrackingOverlay:
    """Creates realistic human overlay that tracks your movements"""
    
    def __init__(self):
        print("👤 Human Tracking Overlay System")
        print("=" * 40)
        
        # Initialize systems
        self.skeleton_system = LiveSkeletonSystem()
        self.renderer = RealisticHumanRenderer()
        
        # Character data
        self.loaded_character = None
        self.character_parts = None
        self.animation_data = []
        
        print("✅ Human tracking overlay ready!")
        print("   • Real-time skeleton tracking: ON")
        print("   • Realistic human rendering: ON")
        print("   • Movement following: ON")
    
    def load_character(self, character_file: str) -> bool:
        """Load character for realistic overlay"""
        try:
            print(f"📁 Loading character: {character_file}")
            
            with open(character_file, 'r') as f:
                character_data = json.load(f)
            
            # Handle different file formats
            if 'character_profile' in character_data:
                profile_data = character_data['character_profile']
            else:
                profile_data = character_data
            
            # Reconstruct character
            self.loaded_character = CharacterProfile(
                name=profile_data['name'],
                facial=FacialFeatures(**profile_data['facial']),
                body=BodyFeatures(**profile_data['body']),
                hair=HairFeatures(**profile_data['hair']),
                clothing=ClothingFeatures(**profile_data['clothing'])
            )
            
            print(f"✅ Character loaded: {self.loaded_character.name}")
            self._print_character_info()
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading character: {e}")
            return False
    
    def _print_character_info(self):
        """Print character information"""
        char = self.loaded_character
        print(f"👤 Character: {char.name}")
        print(f"   • Height: {char.body.height:.2f}m")
        print(f"   • Hair: {char.hair.style.replace('_', ' ').title()} ({char.hair.color})")
        print(f"   • Eyes: {char.facial.eye_color}")
        print(f"   • Skin: {char.facial.skin_tone}")
        print(f"   • Outfit: {char.clothing.shirt_type} + {char.clothing.pants_type}")
    
    def start_live_tracking(self, camera_index: int = 0):
        """Start live human tracking overlay"""
        if not self.loaded_character:
            print("❌ No character loaded!")
            return
        
        print(f"🎬 Starting realistic human tracking")
        print(f"   Character: {self.loaded_character.name}")
        print("   Controls:")
        print("     'q' - Quit")
        print("     's' - Save tracking data")
        print("     'h' - Toggle human overlay")
        print("     'o' - Toggle original tracking")
        
        self._run_tracking_loop(camera_index)
    
    def _run_tracking_loop(self, camera_index: int):
        """Main tracking loop with realistic human overlay"""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_index}")
            return
        
        show_human = True
        show_original = True
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip for mirror effect
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                
                # Create character parts if not cached
                if self.character_parts is None:
                    self.character_parts = self.renderer.create_realistic_human_parts(
                        self.loaded_character, (h, w)
                    )
                
                # Get tracking data
                tracking_data = self.skeleton_system.tracker.process_frame(frame)
                
                # Draw original tracking overlay
                if show_original:
                    frame = self.skeleton_system.tracker.draw_complete_tracking(frame, tracking_data)
                
                # Draw realistic human overlay
                if show_human and tracking_data and tracking_data.body_keypoints_2d:
                    frame = self._draw_realistic_human_overlay(frame, tracking_data)
                
                # Draw info
                self._draw_overlay_info(frame, show_human, show_original, frame_count)
                
                cv2.imshow('Realistic Human Tracking', frame)
                
                # Handle controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('h'):
                    show_human = not show_human
                    print(f"Human overlay: {'ON' if show_human else 'OFF'}")
                elif key == ord('o'):
                    show_original = not show_original
                    print(f"Original tracking: {'ON' if show_original else 'OFF'}")
                elif key == ord('s'):
                    self._save_tracking_data(frame_count)
                
                frame_count += 1
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def _draw_realistic_human_overlay(self, frame: np.ndarray, tracking_data) -> np.ndarray:
        """Draw realistic human overlay that follows your movements"""
        if not tracking_data.body_keypoints_2d or len(tracking_data.body_keypoints_2d) < 51:
            return frame
        
        keypoints = tracking_data.body_keypoints_2d
        
        # Extract keypoint positions with confidence check
        def get_keypoint(index):
            if index * 3 + 2 < len(keypoints):
                x = keypoints[index * 3]
                y = keypoints[index * 3 + 1]
                conf = keypoints[index * 3 + 2]
                if conf > 0.3:
                    return (int(x), int(y))
            return None
        
        # COCO keypoint indices
        keypoint_positions = {
            'nose': get_keypoint(0),
            'left_eye': get_keypoint(1),
            'right_eye': get_keypoint(2),
            'left_shoulder': get_keypoint(5),
            'right_shoulder': get_keypoint(6),
            'left_elbow': get_keypoint(7),
            'right_elbow': get_keypoint(8),
            'left_wrist': get_keypoint(9),
            'right_wrist': get_keypoint(10),
            'left_hip': get_keypoint(11),
            'right_hip': get_keypoint(12),
            'left_knee': get_keypoint(13),
            'right_knee': get_keypoint(14),
            'left_ankle': get_keypoint(15),
            'right_ankle': get_keypoint(16)
        }
        
        # Draw realistic human parts
        frame = self._draw_human_head(frame, keypoint_positions)
        frame = self._draw_human_torso(frame, keypoint_positions)
        frame = self._draw_human_arms(frame, keypoint_positions)
        frame = self._draw_human_legs(frame, keypoint_positions)
        frame = self._draw_human_hands(frame, keypoint_positions, tracking_data)
        
        return frame
    
    def _draw_human_head(self, frame: np.ndarray, keypoints: Dict) -> np.ndarray:
        """Draw realistic human head that follows your head movement"""
        if not keypoints['nose']:
            return frame
        
        head_parts = self.character_parts['head']
        hair_parts = self.character_parts['hair']
        eye_parts = self.character_parts['eyes']
        
        nose_x, nose_y = keypoints['nose']
        
        # Calculate head center and size
        head_width = head_parts['width']
        head_height = head_parts['height']
        
        # Draw hair first (behind head)
        hair_radius = hair_parts['size']
        cv2.circle(frame, (nose_x, nose_y - 15), hair_radius, hair_parts['color'], -1)
        
        # Draw head shape
        cv2.ellipse(frame, (nose_x, nose_y), (head_width//2, head_height//2), 0, 0, 360, head_parts['color'], -1)
        cv2.ellipse(frame, (nose_x, nose_y), (head_width//2, head_height//2), 0, 0, 360, (255, 255, 255), 2)
        
        # Draw eyes
        eye_spacing = eye_parts['spacing']
        eye_size = eye_parts['size']
        eye_y = nose_y - head_height//4
        
        # Eye whites
        cv2.circle(frame, (nose_x - eye_spacing, eye_y), eye_size, (255, 255, 255), -1)
        cv2.circle(frame, (nose_x + eye_spacing, eye_y), eye_size, (255, 255, 255), -1)
        
        # Eye color (iris)
        iris_size = int(eye_size * 0.7)
        cv2.circle(frame, (nose_x - eye_spacing, eye_y), iris_size, eye_parts['color'], -1)
        cv2.circle(frame, (nose_x + eye_spacing, eye_y), iris_size, eye_parts['color'], -1)
        
        # Pupils
        pupil_size = int(eye_size * 0.3)
        cv2.circle(frame, (nose_x - eye_spacing, eye_y), pupil_size, (0, 0, 0), -1)
        cv2.circle(frame, (nose_x + eye_spacing, eye_y), pupil_size, (0, 0, 0), -1)
        
        # Eye outlines
        cv2.circle(frame, (nose_x - eye_spacing, eye_y), eye_size, (0, 0, 0), 1)
        cv2.circle(frame, (nose_x + eye_spacing, eye_y), eye_size, (0, 0, 0), 1)
        
        # Add character name
        cv2.putText(frame, self.loaded_character.name, (nose_x - 60, nose_y - hair_radius - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame
    
    def _draw_human_torso(self, frame: np.ndarray, keypoints: Dict) -> np.ndarray:
        """Draw realistic torso that follows your body movement"""
        if not (keypoints['left_shoulder'] and keypoints['right_shoulder'] and 
                keypoints['left_hip'] and keypoints['right_hip']):
            return frame
        
        torso_parts = self.character_parts['torso']
        
        # Calculate torso bounds
        ls_x, ls_y = keypoints['left_shoulder']
        rs_x, rs_y = keypoints['right_shoulder']
        lh_x, lh_y = keypoints['left_hip']
        rh_x, rh_y = keypoints['right_hip']
        
        # Torso center and dimensions
        shoulder_center_x = (ls_x + rs_x) // 2
        shoulder_center_y = (ls_y + rs_y) // 2
        hip_center_x = (lh_x + rh_x) // 2
        hip_center_y = (lh_y + rh_y) // 2
        
        torso_width = torso_parts['width']
        
        # Draw torso as rounded rectangle
        top_left = (shoulder_center_x - torso_width//2, shoulder_center_y - 20)
        bottom_right = (hip_center_x + torso_width//2, hip_center_y + 20)
        
        # Torso fill
        cv2.rectangle(frame, top_left, bottom_right, torso_parts['color'], -1)
        
        # Add muscle definition if high
        if torso_parts['muscle_definition'] > 0.6:
            # Add subtle muscle lines
            mid_x = (top_left[0] + bottom_right[0]) // 2
            mid_y = (top_left[1] + bottom_right[1]) // 2
            cv2.line(frame, (mid_x, top_left[1]), (mid_x, bottom_right[1]), (200, 200, 200), 1)
        
        # Torso outline
        cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), 2)
        
        # Add clothing label
        label_y = shoulder_center_y + 15
        cv2.putText(frame, f"{torso_parts['type'].replace('_', ' ').title()}", 
                   (shoulder_center_x - 50, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def _draw_human_arms(self, frame: np.ndarray, keypoints: Dict) -> np.ndarray:
        """Draw realistic arms that follow your arm movements"""
        arm_parts = self.character_parts['arms']
        thickness = arm_parts['thickness']
        color = arm_parts['color']
        
        # Left arm
        if keypoints['left_shoulder'] and keypoints['left_elbow']:
            cv2.line(frame, keypoints['left_shoulder'], keypoints['left_elbow'], color, thickness)
        if keypoints['left_elbow'] and keypoints['left_wrist']:
            cv2.line(frame, keypoints['left_elbow'], keypoints['left_wrist'], color, thickness)
        
        # Right arm
        if keypoints['right_shoulder'] and keypoints['right_elbow']:
            cv2.line(frame, keypoints['right_shoulder'], keypoints['right_elbow'], color, thickness)
        if keypoints['right_elbow'] and keypoints['right_wrist']:
            cv2.line(frame, keypoints['right_elbow'], keypoints['right_wrist'], color, thickness)
        
        # Draw joints
        for joint in ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow']:
            if keypoints[joint]:
                cv2.circle(frame, keypoints[joint], thickness//2, color, -1)
                cv2.circle(frame, keypoints[joint], thickness//2 + 1, (255, 255, 255), 1)
        
        return frame
    
    def _draw_human_legs(self, frame: np.ndarray, keypoints: Dict) -> np.ndarray:
        """Draw realistic legs that follow your leg movements"""
        leg_parts = self.character_parts['legs']
        thickness = leg_parts['thickness']
        color = leg_parts['color']
        
        # Left leg
        if keypoints['left_hip'] and keypoints['left_knee']:
            cv2.line(frame, keypoints['left_hip'], keypoints['left_knee'], color, thickness)
        if keypoints['left_knee'] and keypoints['left_ankle']:
            cv2.line(frame, keypoints['left_knee'], keypoints['left_ankle'], color, thickness)
        
        # Right leg
        if keypoints['right_hip'] and keypoints['right_knee']:
            cv2.line(frame, keypoints['right_hip'], keypoints['right_knee'], color, thickness)
        if keypoints['right_knee'] and keypoints['right_ankle']:
            cv2.line(frame, keypoints['right_knee'], keypoints['right_ankle'], color, thickness)
        
        # Draw joints
        for joint in ['left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']:
            if keypoints[joint]:
                cv2.circle(frame, keypoints[joint], thickness//2, color, -1)
                cv2.circle(frame, keypoints[joint], thickness//2 + 1, (255, 255, 255), 1)
        
        return frame
    
    def _draw_human_hands(self, frame: np.ndarray, keypoints: Dict, tracking_data) -> np.ndarray:
        """Draw realistic hands that follow your hand movements"""
        hand_parts = self.character_parts['hands']
        hand_size = hand_parts['size']
        color = hand_parts['color']
        
        # Draw hands at wrist positions
        if keypoints['left_wrist']:
            cv2.circle(frame, keypoints['left_wrist'], hand_size, color, -1)
            cv2.circle(frame, keypoints['left_wrist'], hand_size + 1, (255, 255, 255), 1)
        
        if keypoints['right_wrist']:
            cv2.circle(frame, keypoints['right_wrist'], hand_size, color, -1)
            cv2.circle(frame, keypoints['right_wrist'], hand_size + 1, (255, 255, 255), 1)
        
        # If we have detailed hand tracking, draw fingers
        if hasattr(tracking_data, 'left_hand_2d') and tracking_data.left_hand_2d:
            self._draw_detailed_hand(frame, tracking_data.left_hand_2d, color, "LEFT")
        
        if hasattr(tracking_data, 'right_hand_2d') and tracking_data.right_hand_2d:
            self._draw_detailed_hand(frame, tracking_data.right_hand_2d, color, "RIGHT")
        
        return frame
    
    def _draw_detailed_hand(self, frame: np.ndarray, hand_keypoints: List[float], color: Tuple[int, int, int], hand_label: str):
        """Draw detailed hand with fingers"""
        if len(hand_keypoints) < 63:
            return
        
        # Extract hand points
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
        
        # Draw finger lines
        for start_idx, end_idx in connections:
            if start_idx < len(points) and end_idx < len(points):
                start_point = points[start_idx]
                end_point = points[end_idx]
                if start_point and end_point:
                    cv2.line(frame, start_point, end_point, color, 3)
        
        # Draw finger joints
        for i, point in enumerate(points):
            if point:
                if i in [4, 8, 12, 16, 20]:  # Fingertips
                    cv2.circle(frame, point, 6, (255, 100, 100), -1)
                else:
                    cv2.circle(frame, point, 4, color, -1)
    
    def _draw_overlay_info(self, frame: np.ndarray, show_human: bool, show_original: bool, frame_count: int):
        """Draw overlay information"""
        info_lines = [
            f"REALISTIC HUMAN: {self.loaded_character.name if self.loaded_character else 'None'}",
            f"Human Overlay: {'ON' if show_human else 'OFF'}",
            f"Original Tracking: {'ON' if show_original else 'OFF'}",
            f"Frame: {frame_count}"
        ]
        
        for i, line in enumerate(info_lines):
            color = (0, 255, 255) if "REALISTIC" in line else (255, 255, 255)
            if "ON" in line and "ON" in line.split(": ")[1]:
                color = (0, 255, 0)
            elif "OFF" in line:
                color = (0, 0, 255)
            
            cv2.putText(frame, line, (10, 25 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def _save_tracking_data(self, frame_count: int):
        """Save tracking data"""
        timestamp = int(time.time())
        filename = f"realistic_human_tracking_{self.loaded_character.name}_{timestamp}.json"
        
        data = {
            'character': asdict(self.loaded_character),
            'frames_tracked': frame_count,
            'timestamp': timestamp,
            'tracking_type': 'realistic_human_overlay'
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Tracking data saved: {filename}")

def main():
    """Main function"""
    print("Realistic Human Tracking Overlay")
    print("=" * 40)
    
    overlay = HumanTrackingOverlay()
    
    print("\nOptions:")
    print("1. Load character and start realistic tracking")
    print("2. Exit")
    
    choice = input("\nEnter choice (1-2): ").strip()
    
    if choice == "1":
        character_file = input("Enter character JSON filename: ").strip()
        if overlay.load_character(character_file):
            camera_index = input("Camera index (0 for default): ").strip()
            try:
                camera_index = int(camera_index) if camera_index else 0
            except:
                camera_index = 0
            overlay.start_live_tracking(camera_index)
    
    elif choice == "2":
        print("Goodbye!")

if __name__ == "__main__":
    main()