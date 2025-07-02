"""
Skeleton Mapping System for Realistic Character Overlay
Converts tracking data to 3D character animation coordinates
"""

import numpy as np
import json
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from scipy.spatial.transform import Rotation
import cv2
import time

@dataclass
class BoneTransform:
    """Represents a bone transformation in 3D space"""
    position: Tuple[float, float, float]  # X, Y, Z world coordinates
    rotation: Tuple[float, float, float, float]  # Quaternion (w, x, y, z)
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)

@dataclass
class CharacterSkeleton:
    """Complete character skeleton with all bone transforms"""
    timestamp: float
    frame_number: int
    
    # Main body bones
    root: BoneTransform
    spine: BoneTransform
    chest: BoneTransform
    neck: BoneTransform
    head: BoneTransform
    
    # Arms
    left_shoulder: BoneTransform
    left_elbow: BoneTransform
    left_wrist: BoneTransform
    right_shoulder: BoneTransform
    right_elbow: BoneTransform
    right_wrist: BoneTransform
    
    # Legs
    left_hip: BoneTransform
    left_knee: BoneTransform
    left_ankle: BoneTransform
    right_hip: BoneTransform
    right_knee: BoneTransform
    right_ankle: BoneTransform
    
    # Hands (optional - detailed finger bones)
    left_hand_bones: Optional[Dict[str, BoneTransform]] = None
    right_hand_bones: Optional[Dict[str, BoneTransform]] = None
    
    # Facial bones (optional - for facial animation)
    facial_bones: Optional[Dict[str, BoneTransform]] = None

class SkeletonMapper:
    """Maps tracking data to 3D character skeleton"""
    
    def __init__(self):
        print("🦴 Initializing Skeleton Mapping System")
        print("=" * 50)
        
        # COCO keypoint indices (from your tracking data)
        self.coco_keypoints = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
        }
        
        # Character proportions (adjustable for different character types)
        self.character_proportions = {
            'height': 1.8,  # meters
            'arm_length_ratio': 0.4,  # relative to height
            'leg_length_ratio': 0.5,
            'shoulder_width_ratio': 0.25,
            'hip_width_ratio': 0.2
        }
        
        # Smoothing parameters
        self.smoothing_factor = 0.7  # For temporal smoothing
        self.previous_skeleton = None
        
        # Coordinate system conversion
        self.world_scale = 100.0  # Convert to centimeters
        self.reference_height_pixels = 480  # Reference for scale calculation
        
        print("✅ Skeleton mapping initialized")
        print(f"   • Character height: {self.character_proportions['height']}m")
        print(f"   • Smoothing enabled: {self.smoothing_factor}")
        print(f"   • World scale: {self.world_scale}cm per unit")
    
    def map_tracking_to_skeleton(self, tracking_data: dict) -> CharacterSkeleton:
        """Convert tracking data to 3D character skeleton"""
        
        if not tracking_data.get('body_keypoints_2d'):
            return None
        
        # Extract keypoints
        keypoints_2d = tracking_data['body_keypoints_2d']
        keypoints_3d = tracking_data.get('body_keypoints_3d', None)
        
        # Convert to 3D world coordinates
        world_positions = self._convert_to_world_coordinates(keypoints_2d, keypoints_3d)
        
        # Calculate bone transforms
        skeleton = self._calculate_bone_transforms(world_positions, tracking_data)
        
        # Add hand and face data if available
        skeleton = self._add_hand_transforms(skeleton, tracking_data)
        skeleton = self._add_facial_transforms(skeleton, tracking_data)
        
        # Apply smoothing
        # if self.previous_skeleton:
        #     skeleton = self._apply_temporal_smoothing(skeleton, self.previous_skeleton)
        
        self.previous_skeleton = skeleton
        return skeleton
    
    def _convert_to_world_coordinates(self, keypoints_2d: List[float], keypoints_3d: Optional[List[List[float]]]) -> Dict[str, Tuple[float, float, float]]:
        """Convert 2D/3D keypoints to world coordinates"""
        world_positions = {}
        
        # Calculate scale factor based on person height in frame
        scale_factor = self._calculate_scale_factor(keypoints_2d)
        
        for name, idx in self.coco_keypoints.items():
            # Extract 2D coordinates
            x_2d = keypoints_2d[idx * 3]
            y_2d = keypoints_2d[idx * 3 + 1]
            confidence = keypoints_2d[idx * 3 + 2]
            
            if confidence > 0.3:  # Only use confident detections
                # Convert to world coordinates
                # X: left-right (screen X)
                # Y: up-down (inverted screen Y) 
                # Z: forward-back (estimated depth)
                
                world_x = (x_2d - 320) * scale_factor  # Center and scale
                world_y = (240 - y_2d) * scale_factor  # Invert Y and scale
                
                # Estimate Z depth
                if keypoints_3d and idx < len(keypoints_3d):
                    # Use MediaPipe 3D data if available
                    world_z = keypoints_3d[idx][2] * scale_factor * 50  # Scale Z
                else:
                    # Estimate depth based on body part
                    world_z = self._estimate_depth(name, world_x, world_y)
                
                world_positions[name] = (world_x, world_y, world_z)
        
        return world_positions
    
    def _calculate_scale_factor(self, keypoints_2d: List[float]) -> float:
        """Calculate scale factor based on detected person size"""
        # Use shoulder-to-hip distance to estimate person scale
        left_shoulder_idx = self.coco_keypoints['left_shoulder'] * 3
        right_shoulder_idx = self.coco_keypoints['right_shoulder'] * 3
        left_hip_idx = self.coco_keypoints['left_hip'] * 3
        right_hip_idx = self.coco_keypoints['right_hip'] * 3
        
        # Calculate torso height in pixels
        shoulder_y = (keypoints_2d[left_shoulder_idx + 1] + keypoints_2d[right_shoulder_idx + 1]) / 2
        hip_y = (keypoints_2d[left_hip_idx + 1] + keypoints_2d[right_hip_idx + 1]) / 2
        torso_height_pixels = abs(shoulder_y - hip_y)
        
        if torso_height_pixels > 0:
            # Estimate full height (torso is ~40% of full height)
            estimated_height_pixels = torso_height_pixels / 0.4
            # Scale to world coordinates
            return (self.character_proportions['height'] * self.world_scale) / estimated_height_pixels
        
        return 1.0  # Default scale
    
    def _estimate_depth(self, joint_name: str, world_x: float, world_y: float) -> float:
        """Estimate Z depth for joints without 3D data"""
        # Basic depth estimation based on anatomy
        depth_offsets = {
            'nose': 10.0,
            'left_ear': 8.0, 'right_ear': 8.0,
            'left_shoulder': 5.0, 'right_shoulder': 5.0,
            'left_elbow': 15.0, 'right_elbow': 15.0,
            'left_wrist': 20.0, 'right_wrist': 20.0,
            'left_hip': 0.0, 'right_hip': 0.0,  # Reference depth
            'left_knee': 5.0, 'right_knee': 5.0,
            'left_ankle': 3.0, 'right_ankle': 3.0
        }
        
        return depth_offsets.get(joint_name, 0.0)
    
    def _calculate_bone_transforms(self, positions: Dict[str, Tuple[float, float, float]], tracking_data: dict) -> CharacterSkeleton:
        """Calculate bone transforms from joint positions"""
        
        # Root position (center of hips)
        if 'left_hip' in positions and 'right_hip' in positions:
            left_hip = np.array(positions['left_hip'])
            right_hip = np.array(positions['right_hip'])
            root_pos = ((left_hip + right_hip) / 2).tolist()
        else:
            root_pos = [0, 0, 0]
        
        # Calculate bone rotations using direction vectors
        bone_rotations = self._calculate_bone_rotations(positions)
        
        # Create skeleton
        skeleton = CharacterSkeleton(
            timestamp=tracking_data.get('timestamp', 0),
            frame_number=tracking_data.get('frame_number', 0),
            
            # Main body
            root=BoneTransform(tuple(root_pos), (1, 0, 0, 0)),
            spine=self._create_bone_transform('spine', positions, bone_rotations),
            chest=self._create_bone_transform('chest', positions, bone_rotations),
            neck=self._create_bone_transform('neck', positions, bone_rotations),
            head=self._create_bone_transform('head', positions, bone_rotations),
            
            # Arms
            left_shoulder=self._create_bone_transform('left_shoulder', positions, bone_rotations),
            left_elbow=self._create_bone_transform('left_elbow', positions, bone_rotations),
            left_wrist=self._create_bone_transform('left_wrist', positions, bone_rotations),
            right_shoulder=self._create_bone_transform('right_shoulder', positions, bone_rotations),
            right_elbow=self._create_bone_transform('right_elbow', positions, bone_rotations),
            right_wrist=self._create_bone_transform('right_wrist', positions, bone_rotations),
            
            # Legs
            left_hip=self._create_bone_transform('left_hip', positions, bone_rotations),
            left_knee=self._create_bone_transform('left_knee', positions, bone_rotations),
            left_ankle=self._create_bone_transform('left_ankle', positions, bone_rotations),
            right_hip=self._create_bone_transform('right_hip', positions, bone_rotations),
            right_knee=self._create_bone_transform('right_knee', positions, bone_rotations),
            right_ankle=self._create_bone_transform('right_ankle', positions, bone_rotations)
        )
        
        return skeleton
    
    def _calculate_bone_rotations(self, positions: Dict[str, Tuple[float, float, float]]) -> Dict[str, Tuple[float, float, float, float]]:
        """Calculate bone rotations from joint positions"""
        rotations = {}
        
        # Calculate direction vectors for major bones
        bone_vectors = {
            'left_upper_arm': self._get_direction_vector(positions, 'left_shoulder', 'left_elbow'),
            'left_forearm': self._get_direction_vector(positions, 'left_elbow', 'left_wrist'),
            'right_upper_arm': self._get_direction_vector(positions, 'right_shoulder', 'right_elbow'),
            'right_forearm': self._get_direction_vector(positions, 'right_elbow', 'right_wrist'),
            'left_thigh': self._get_direction_vector(positions, 'left_hip', 'left_knee'),
            'left_shin': self._get_direction_vector(positions, 'left_knee', 'left_ankle'),
            'right_thigh': self._get_direction_vector(positions, 'right_hip', 'right_knee'),
            'right_shin': self._get_direction_vector(positions, 'right_knee', 'right_ankle'),
            'spine': self._get_direction_vector(positions, 'left_hip', 'left_shoulder', 'right_hip', 'right_shoulder')
        }
        
        # Convert direction vectors to quaternions
        for bone_name, direction in bone_vectors.items():
            if direction is not None:
                rotations[bone_name] = self._direction_to_quaternion(direction)
            else:
                rotations[bone_name] = (1, 0, 0, 0)  # Identity quaternion
        
        return rotations
    
    def _get_direction_vector(self, positions: Dict, start_joint: str, end_joint: str, 
                            start_joint2: str = None, end_joint2: str = None) -> Optional[np.ndarray]:
        """Calculate direction vector between joints"""
        if start_joint2 and end_joint2:
            # Average of two vectors (for spine calculation)
            if all(joint in positions for joint in [start_joint, end_joint, start_joint2, end_joint2]):
                vec1 = np.array(positions[end_joint]) - np.array(positions[start_joint])
                vec2 = np.array(positions[end_joint2]) - np.array(positions[start_joint2])
                direction = (vec1 + vec2) / 2
            else:
                return None
        else:
            # Single vector
            if start_joint in positions and end_joint in positions:
                direction = np.array(positions[end_joint]) - np.array(positions[start_joint])
            else:
                return None
        
        # Normalize
        length = np.linalg.norm(direction)
        if length > 0:
            return direction / length
        return None
    
    def _direction_to_quaternion(self, direction: np.ndarray) -> Tuple[float, float, float, float]:
        """Convert direction vector to quaternion rotation"""
        # Default bone direction is along Y-axis (0, 1, 0)
        default_direction = np.array([0, 1, 0])
        
        # Calculate rotation axis and angle
        axis = np.cross(default_direction, direction)
        axis_length = np.linalg.norm(axis)
        
        if axis_length < 1e-6:  # Vectors are parallel
            if np.dot(default_direction, direction) > 0:
                return (1, 0, 0, 0)  # No rotation needed
            else:
                return (0, 0, 0, 1)  # 180-degree rotation
        
        axis = axis / axis_length
        angle = math.acos(np.clip(np.dot(default_direction, direction), -1, 1))
        
        # Convert to quaternion
        w = math.cos(angle / 2)
        x = axis[0] * math.sin(angle / 2)
        y = axis[1] * math.sin(angle / 2)
        z = axis[2] * math.sin(angle / 2)
        
        return (w, x, y, z)
    
    def _create_bone_transform(self, bone_name: str, positions: Dict, rotations: Dict) -> BoneTransform:
        """Create bone transform from position and rotation data"""
        # Map bone names to joint positions
        bone_to_joint = {
            'spine': 'left_hip',  # Approximate
            'chest': 'left_shoulder',  # Approximate  
            'neck': 'nose',  # Approximate
            'head': 'nose',
            'left_shoulder': 'left_shoulder',
            'left_elbow': 'left_elbow', 
            'left_wrist': 'left_wrist',
            'right_shoulder': 'right_shoulder',
            'right_elbow': 'right_elbow',
            'right_wrist': 'right_wrist',
            'left_hip': 'left_hip',
            'left_knee': 'left_knee',
            'left_ankle': 'left_ankle',
            'right_hip': 'right_hip', 
            'right_knee': 'right_knee',
            'right_ankle': 'right_ankle'
        }
        
        # Get position
        joint_name = bone_to_joint.get(bone_name, bone_name)
        if joint_name in positions:
            position = positions[joint_name]
        else:
            position = (0, 0, 0)
        
        # Get rotation
        rotation_key = self._get_rotation_key(bone_name)
        if rotation_key in rotations:
            rotation = rotations[rotation_key]
        else:
            rotation = (1, 0, 0, 0)  # Identity
        
        return BoneTransform(position, rotation)
    
    def _get_rotation_key(self, bone_name: str) -> str:
        """Map bone names to rotation keys"""
        rotation_mapping = {
            'left_shoulder': 'left_upper_arm',
            'left_elbow': 'left_forearm', 
            'right_shoulder': 'right_upper_arm',
            'right_elbow': 'right_forearm',
            'left_hip': 'left_thigh',
            'left_knee': 'left_shin',
            'right_hip': 'right_thigh',
            'right_knee': 'right_shin',
            'spine': 'spine',
            'chest': 'spine',
            'neck': 'spine'
        }
        return rotation_mapping.get(bone_name, bone_name)
    
    def _add_hand_transforms(self, skeleton: CharacterSkeleton, tracking_data: dict) -> CharacterSkeleton:
        """Add detailed hand bone transforms"""
        # Left hand
        if tracking_data.get('left_hand_2d'):
            skeleton.left_hand_bones = self._calculate_hand_bones(
                tracking_data['left_hand_2d'], 
                tracking_data.get('left_hand_3d'),
                'left'
            )
        
        # Right hand  
        if tracking_data.get('right_hand_2d'):
            skeleton.right_hand_bones = self._calculate_hand_bones(
                tracking_data['right_hand_2d'],
                tracking_data.get('right_hand_3d'), 
                'right'
            )
        
        return skeleton
    
    def _calculate_hand_bones(self, hand_2d: List[float], hand_3d: Optional[List[List[float]]], hand_side: str) -> Dict[str, BoneTransform]:
        """Calculate bone transforms for finger joints"""
        # MediaPipe hand landmark names
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        hand_bones = {}
        
        # Calculate scale factor for hand
        wrist_pos = np.array([hand_2d[0], hand_2d[1], 0])
        middle_tip_pos = np.array([hand_2d[12*3], hand_2d[12*3+1], 0])
        hand_scale = np.linalg.norm(middle_tip_pos - wrist_pos) * 0.01  # Convert to world scale
        
        # Process each finger
        for finger_idx, finger_name in enumerate(finger_names):
            finger_bones = self._calculate_finger_bones(hand_2d, hand_3d, finger_idx, hand_scale)
            hand_bones.update(finger_bones)
        
        return hand_bones
    
    def _calculate_finger_bones(self, hand_2d: List[float], hand_3d: Optional[List[List[float]]], 
                               finger_idx: int, scale: float) -> Dict[str, BoneTransform]:
        """Calculate bone transforms for a single finger"""
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        finger_name = finger_names[finger_idx]
        
        # MediaPipe landmark indices for each finger
        finger_landmarks = {
            0: [1, 2, 3, 4],    # thumb
            1: [5, 6, 7, 8],    # index
            2: [9, 10, 11, 12], # middle
            3: [13, 14, 15, 16],# ring
            4: [17, 18, 19, 20] # pinky
        }
        
        landmarks = finger_landmarks[finger_idx]
        finger_bones = {}
        
        # Calculate bone transforms for each segment
        bone_names = ['metacarpal', 'proximal', 'intermediate', 'distal']
        
        for i in range(len(landmarks) - 1):
            start_idx = landmarks[i] * 3
            end_idx = landmarks[i + 1] * 3
            
            # Position
            start_pos = np.array([hand_2d[start_idx], hand_2d[start_idx + 1], 0]) * scale
            end_pos = np.array([hand_2d[end_idx], hand_2d[end_idx + 1], 0]) * scale
            
            # Add 3D depth if available
            if hand_3d and len(hand_3d) > landmarks[i]:
                start_pos[2] = hand_3d[landmarks[i]][2] * scale * 10
                end_pos[2] = hand_3d[landmarks[i + 1]][2] * scale * 10
            
            # Direction and rotation
            direction = end_pos - start_pos
            rotation = self._direction_to_quaternion(direction / np.linalg.norm(direction)) if np.linalg.norm(direction) > 0 else (1, 0, 0, 0)
            
            bone_name = f"{finger_name}_{bone_names[i]}"
            finger_bones[bone_name] = BoneTransform(tuple(start_pos), rotation)
        
        return finger_bones
    
    def _add_facial_transforms(self, skeleton: CharacterSkeleton, tracking_data: dict) -> CharacterSkeleton:
        """Add facial bone transforms for expression"""
        if tracking_data.get('head_pose'):
            head_pose = tracking_data['head_pose']
            
            # Convert head pose to facial bones
            facial_bones = {
                'jaw': self._create_facial_bone('jaw', head_pose, tracking_data.get('facial_expressions')),
                'left_eyebrow': self._create_facial_bone('left_eyebrow', head_pose, tracking_data.get('facial_expressions')),
                'right_eyebrow': self._create_facial_bone('right_eyebrow', head_pose, tracking_data.get('facial_expressions')),
                'left_eye': self._create_facial_bone('left_eye', head_pose, tracking_data.get('facial_expressions')),
                'right_eye': self._create_facial_bone('right_eye', head_pose, tracking_data.get('facial_expressions'))
            }
            
            skeleton.facial_bones = facial_bones
        
        return skeleton
    
    def _create_facial_bone(self, bone_name: str, head_pose: Dict, expressions: Optional[Dict]) -> BoneTransform:
        """Create facial bone transform from head pose and expressions"""
        # Base position (relative to head)
        base_positions = {
            'jaw': (0, -10, 0),
            'left_eyebrow': (-5, 5, 2),
            'right_eyebrow': (5, 5, 2),
            'left_eye': (-3, 2, 3),
            'right_eye': (3, 2, 3)
        }
        
        position = base_positions.get(bone_name, (0, 0, 0))
        
        # Apply expression offsets
        if expressions:
            if bone_name == 'jaw' and 'mouth_open' in expressions:
                position = (position[0], position[1] - expressions['mouth_open'] * 5, position[2])
            elif 'eyebrow' in bone_name and 'eyebrow_raise' in expressions:
                position = (position[0], position[1] + expressions['eyebrow_raise'] * 3, position[2])
        
        # Rotation based on head pose
        pitch = math.radians(head_pose.get('pitch', 0))
        yaw = math.radians(head_pose.get('yaw', 0))
        roll = math.radians(head_pose.get('roll', 0))
        
        # Convert Euler angles to quaternion
        r = Rotation.from_euler('xyz', [pitch, yaw, roll])
        quat = r.as_quat()  # Returns [x, y, z, w]
        rotation = (quat[3], quat[0], quat[1], quat[2])  # Convert to (w, x, y, z)
        
        return BoneTransform(position, rotation)
    
    def _apply_temporal_smoothing(self, current: CharacterSkeleton, previous: CharacterSkeleton) -> CharacterSkeleton:
        """Apply temporal smoothing to reduce jitter"""
        # Smooth all bone transforms
        smoothed_skeleton = current  # Start with current
        
        # Apply smoothing to each bone
        bones_to_smooth = [
            'root', 'spine', 'chest', 'neck', 'head',
            'left_shoulder', 'left_elbow', 'left_wrist',
            'right_shoulder', 'right_elbow', 'right_wrist',
            'left_hip', 'left_knee', 'left_ankle',
            'right_hip', 'right_knee', 'right_ankle'
        ]
        
        for bone_name in bones_to_smooth:
            current_bone = getattr(current, bone_name)
            previous_bone = getattr(previous, bone_name)
            
            # Smooth position
            smooth_pos = self._smooth_vector(current_bone.position, previous_bone.position)
            
            # Smooth rotation (quaternion SLERP)
            smooth_rot = self._smooth_quaternion(current_bone.rotation, previous_bone.rotation)
            
            # Update bone
            smoothed_bone = BoneTransform(smooth_pos, smooth_rot, current_bone.scale)
            setattr(smoothed_skeleton, bone_name, smoothed_bone)
        
        return smoothed_skeleton
    
    def _smooth_vector(self, current: Tuple[float, float, float], previous: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Apply smoothing to a 3D vector"""
        factor = self.smoothing_factor
        return (
            current[0] * factor + previous[0] * (1 - factor),
            current[1] * factor + previous[1] * (1 - factor),
            current[2] * factor + previous[2] * (1 - factor)
        )
    
    def _smooth_quaternion(self, current: Tuple[float, float, float, float], previous: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """Apply SLERP smoothing to quaternions"""
        # Simple linear interpolation instead of SLERP for now
        factor = self.smoothing_factor
        
        # Ensure quaternions are in same hemisphere (shortest path)
        curr = np.array(current)
        prev = np.array(previous)
        
        # Check if we need to flip one quaternion for shortest path
        if np.dot(curr, prev) < 0:
            curr = -curr
        
        # Linear interpolation
        smoothed = curr * factor + prev * (1 - factor)
        
        # Normalize the result
        smoothed = smoothed / np.linalg.norm(smoothed)
        
        return tuple(smoothed)
    
    def export_skeleton_animation(self, skeleton_sequence: List[CharacterSkeleton], format: str = 'json') -> str:
        """Export skeleton sequence as animation data"""
        if format == 'json':
            return self._export_json(skeleton_sequence)
        elif format == 'bvh':
            return self._export_bvh(skeleton_sequence)
        elif format == 'fbx':
            return self._export_fbx(skeleton_sequence)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_json(self, skeleton_sequence: List[CharacterSkeleton]) -> str:
        """Export as JSON for easy parsing"""
        animation_data = {
            'version': '1.0',
            'character_type': 'humanoid',
            'frame_rate': 30,
            'total_frames': len(skeleton_sequence),
            'skeleton_data': [asdict(skeleton) for skeleton in skeleton_sequence]
        }
        
        filename = f"character_animation_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(animation_data, f, indent=2)
        
        return filename
    
    def _export_bvh(self, skeleton_sequence: List[CharacterSkeleton]) -> str:
        """Export as BVH for Blender/Maya compatibility"""
        # BVH header
        bvh_content = "HIERARCHY\n"
        bvh_content += "ROOT Root\n{\n"
        bvh_content += "  OFFSET 0.0 0.0 0.0\n"
        bvh_content += "  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n"
        
        # Add joint hierarchy
        bvh_content += self._generate_bvh_hierarchy()
        
        bvh_content += "}\n"
        bvh_content += f"MOTION\nFrames: {len(skeleton_sequence)}\n"
        bvh_content += "Frame Time: 0.033333\n"  # 30 FPS
        
        # Add frame data
        for skeleton in skeleton_sequence:
            bvh_content += self._skeleton_to_bvh_frame(skeleton) + "\n"
        
        filename = f"character_animation_{int(time.time())}.bvh"
        with open(filename, 'w') as f:
            f.write(bvh_content)
        
        return filename
    
    def _generate_bvh_hierarchy(self) -> str:
        """Generate BVH joint hierarchy"""
        # This would be a complex function to generate proper BVH hierarchy
        # For now, return a simplified version
        return """  JOINT Spine
{
  OFFSET 0.0 10.0 0.0
  CHANNELS 3 Zrotation Xrotation Yrotation
  JOINT Chest
  {
    OFFSET 0.0 15.0 0.0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT LeftShoulder
    {
      OFFSET -15.0 5.0 0.0
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT LeftElbow
      {
        OFFSET -25.0 0.0 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT LeftWrist
        {
          OFFSET -20.0 0.0 0.0
          CHANNELS 3 Zrotation Xrotation Yrotation
          End Site
          {
            OFFSET -5.0 0.0 0.0
          }
        }
      }
    }
  }
}"""

# Test the mapping system
def test_skeleton_mapping():
    """Test the skeleton mapping with sample data"""
    print("🧪 Testing Skeleton Mapping System")
    
    mapper = SkeletonMapper()
    
    # Sample tracking data (simplified)
    sample_data = {
        'timestamp': 1234567890.0,
        'frame_number': 1,
        'body_keypoints_2d': [
            # Sample COCO keypoints [x, y, confidence] * 17
            320, 100, 0.9,  # nose
            310, 95, 0.8,   # left_eye
            330, 95, 0.8,   # right_eye
            300, 100, 0.7,  # left_ear
            340, 100, 0.7,  # right_ear
            280, 150, 0.9,  # left_shoulder
            360, 150, 0.9,  # right_shoulder
            250, 200, 0.8,  # left_elbow
            390, 200, 0.8,  # right_elbow
            220, 250, 0.7,  # left_wrist
            420, 250, 0.7,  # right_wrist
            290, 250, 0.9,  # left_hip
            350, 250, 0.9,  # right_hip
            285, 350, 0.8,  # left_knee
            355, 350, 0.8,  # right_knee
            280, 450, 0.7,  # left_ankle
            360, 450, 0.7   # right_ankle
        ],
        'head_pose': {'pitch': 10.0, 'yaw': -5.0, 'roll': 2.0}
    }
    
    # Test mapping
    skeleton = mapper.map_tracking_to_skeleton(sample_data)
    
    if skeleton:
        print("✅ Skeleton mapping successful!")
        print(f"   Root position: {skeleton.root.position}")
        print(f"   Head rotation: {skeleton.head.rotation}")
        print(f"   Left shoulder: {skeleton.left_shoulder.position}")
        
        # Export test
        filename = mapper.export_skeleton_animation([skeleton], 'json')
        print(f"   Exported to: {filename}")
    else:
        print("❌ Skeleton mapping failed!")

if __name__ == "__main__":
    test_skeleton_mapping()