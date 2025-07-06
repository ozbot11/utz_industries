"""
OpenPose-Based Batch Processor
Uses dedicated OpenPose model for proper skeleton detection
"""

import cv2
import numpy as np
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import asdict
import platform

# Prevent system sleep during processing
try:
    if platform.system() == "Windows":
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_DISPLAY_REQUIRED = 0x00000002
        
        def prevent_sleep():
            ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
            )
            print("🔒 Sleep prevention activated")
        
        def allow_sleep():
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            print("🔓 Sleep prevention deactivated")
    else:
        def prevent_sleep():
            print("🔒 Please keep computer awake")
        def allow_sleep():
            pass
except ImportError:
    def prevent_sleep():
        print("🔒 Please keep computer awake")
    def allow_sleep():
        pass

# AI Generation imports
try:
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    from diffusers import UniPCMultistepScheduler
    from PIL import Image
    import torch
    AI_AVAILABLE = True
except ImportError as e:
    AI_AVAILABLE = False
    print(f"⚠️ AI libraries missing: {e}")

# OpenPose imports
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

from realistic_character_creator import CharacterProfile, FacialFeatures, BodyFeatures, HairFeatures, ClothingFeatures

class OpenPoseSkeleton:
    """Dedicated OpenPose skeleton generator"""
    
    def __init__(self):
        print("🦴 Initializing OpenPose Skeleton Generator...")
        
        if MEDIAPIPE_AVAILABLE:
            # Use MediaPipe Pose for better skeleton generation
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=True,  # Better for single images
                model_complexity=2,      # Highest accuracy
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            print("✅ MediaPipe OpenPose ready!")
        else:
            print("❌ MediaPipe not available!")
            
    def generate_openpose_skeleton(self, image_path: str) -> Optional[np.ndarray]:
        """Generate proper OpenPose skeleton from image"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Could not load: {image_path}")
                return None
                
            print(f"🦴 Generating skeleton for: {Path(image_path).name}")
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            # Process with MediaPipe
            results = self.pose.process(image_rgb)
            
            if not results.pose_landmarks:
                print(f"   No pose detected")
                return None
            
            # Create OpenPose skeleton
            skeleton = self._create_openpose_skeleton(results.pose_landmarks, h, w)
            
            if skeleton is not None:
                print(f"✅ Generated OpenPose skeleton")
                return skeleton
            else:
                print(f"❌ Failed to create skeleton")
                return None
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def _create_openpose_skeleton(self, pose_landmarks, height: int, width: int) -> np.ndarray:
        """Create refined OpenPose skeleton with anatomical corrections"""
        # Create blank canvas
        skeleton = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Extract and validate landmarks
        landmarks = []
        for i, landmark in enumerate(pose_landmarks.landmark):
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            visibility = landmark.visibility
            landmarks.append((x, y, visibility))
        
        # Create refined skeleton with anatomical corrections
        refined_points = self._refine_skeleton_anatomy(landmarks, width, height)
        
        if len(refined_points) < 8:  # Need minimum points for valid skeleton
            print(f"   Insufficient refined points: {len(refined_points)}")
            return None
        
        # Draw skeleton with improved connections
        skeleton = self._draw_refined_skeleton(skeleton, refined_points)
        
        return skeleton
    
    def _refine_skeleton_anatomy(self, landmarks: List, width: int, height: int) -> Dict:
        """Refine skeleton to fix anatomical proportions"""
        refined_points = {}
        
        # Key MediaPipe landmark indices
        NOSE = 0
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_ELBOW = 13
        RIGHT_ELBOW = 14
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_KNEE = 25
        RIGHT_KNEE = 26
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28
        
        # Extract core points with validation
        def get_point(idx, min_visibility=0.5):
            if idx < len(landmarks) and landmarks[idx][2] > min_visibility:
                x, y, vis = landmarks[idx]
                # Clamp to image bounds
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                return (x, y)
            return None
        
        # Get core structural points
        nose = get_point(NOSE, 0.3)  # Lower threshold for head
        left_shoulder = get_point(LEFT_SHOULDER)
        right_shoulder = get_point(RIGHT_SHOULDER)
        left_elbow = get_point(LEFT_ELBOW)
        right_elbow = get_point(RIGHT_ELBOW)
        left_wrist = get_point(LEFT_WRIST)
        right_wrist = get_point(RIGHT_WRIST)
        left_hip = get_point(LEFT_HIP)
        right_hip = get_point(RIGHT_HIP)
        left_knee = get_point(LEFT_KNEE)
        right_knee = get_point(RIGHT_KNEE)
        left_ankle = get_point(LEFT_ANKLE)
        right_ankle = get_point(RIGHT_ANKLE)
        
        # Calculate derived points with anatomical constraints
        
        # 1. Neck (between shoulders, above both)
        neck = None
        if left_shoulder and right_shoulder:
            neck_x = (left_shoulder[0] + right_shoulder[0]) // 2
            neck_y = min(left_shoulder[1], right_shoulder[1]) - 10  # Slightly above shoulders
            neck = (neck_x, max(neck_y, 0))
        
        # 2. Mid-hip (between hips)
        mid_hip = None
        if left_hip and right_hip:
            mid_hip_x = (left_hip[0] + right_hip[0]) // 2
            mid_hip_y = (left_hip[1] + right_hip[1]) // 2
            mid_hip = (mid_hip_x, mid_hip_y)
        
        # 3. Anatomical arm corrections
        if left_shoulder and left_elbow and left_wrist:
            # Fix left arm proportions
            left_elbow, left_wrist = self._fix_arm_proportions(
                left_shoulder, left_elbow, left_wrist, "left"
            )
        
        if right_shoulder and right_elbow and right_wrist:
            # Fix right arm proportions
            right_elbow, right_wrist = self._fix_arm_proportions(
                right_shoulder, right_elbow, right_wrist, "right"
            )
        
        # 4. Leg proportion fixes
        if left_hip and left_knee and left_ankle:
            left_knee, left_ankle = self._fix_leg_proportions(
                left_hip, left_knee, left_ankle
            )
        
        if right_hip and right_knee and right_ankle:
            right_knee, right_ankle = self._fix_leg_proportions(
                right_hip, right_knee, right_ankle
            )
        
        # Map to OpenPose indices with refined points
        openpose_mapping = {
            0: nose,           # Nose
            1: neck,           # Neck
            2: right_shoulder, # Right shoulder
            3: right_elbow,    # Right elbow
            4: right_wrist,    # Right wrist
            5: left_shoulder,  # Left shoulder
            6: left_elbow,     # Left elbow
            7: left_wrist,     # Left wrist
            8: mid_hip,        # Mid hip
            9: right_hip,      # Right hip
            10: right_knee,    # Right knee
            11: right_ankle,   # Right ankle
            12: left_hip,      # Left hip
            13: left_knee,     # Left knee
            14: left_ankle,    # Left ankle
        }
        
        # Add only valid points
        for idx, point in openpose_mapping.items():
            if point is not None:
                refined_points[idx] = point
        
        print(f"   Refined to {len(refined_points)} anatomically correct points")
        return refined_points
    
    def _fix_arm_proportions(self, shoulder: Tuple, elbow: Tuple, wrist: Tuple, side: str) -> Tuple:
        """Fix arm proportions to look more natural"""
        # Calculate arm vectors
        upper_arm = np.array(elbow) - np.array(shoulder)
        forearm = np.array(wrist) - np.array(elbow)
        
        # Ideal proportions: upper arm ≈ forearm length
        upper_arm_length = np.linalg.norm(upper_arm)
        forearm_length = np.linalg.norm(forearm)
        
        if upper_arm_length > 0 and forearm_length > 0:
            # Normalize forearm to be similar length to upper arm
            ideal_forearm_length = upper_arm_length * 0.9  # Slightly shorter
            
            if forearm_length > 0:
                forearm_unit = forearm / forearm_length
                corrected_forearm = forearm_unit * ideal_forearm_length
                corrected_wrist = tuple((np.array(elbow) + corrected_forearm).astype(int))
            else:
                corrected_wrist = wrist
            
            # Ensure elbow is reasonable distance from shoulder
            if upper_arm_length > 150:  # Too far
                elbow_unit = upper_arm / upper_arm_length
                corrected_upper_arm = elbow_unit * 120  # Reasonable upper arm length
                corrected_elbow = tuple((np.array(shoulder) + corrected_upper_arm).astype(int))
            else:
                corrected_elbow = elbow
            
            return corrected_elbow, corrected_wrist
        
        return elbow, wrist
    
    def _fix_leg_proportions(self, hip: Tuple, knee: Tuple, ankle: Tuple) -> Tuple:
        """Fix leg proportions to look more natural"""
        # Calculate leg vectors
        thigh = np.array(knee) - np.array(hip)
        shin = np.array(ankle) - np.array(knee)
        
        # Ideal proportions: thigh ≈ shin length
        thigh_length = np.linalg.norm(thigh)
        shin_length = np.linalg.norm(shin)
        
        if thigh_length > 0 and shin_length > 0:
            # Normalize shin to be similar length to thigh
            ideal_shin_length = thigh_length * 0.95
            
            if shin_length > 0:
                shin_unit = shin / shin_length
                corrected_shin = shin_unit * ideal_shin_length
                corrected_ankle = tuple((np.array(knee) + corrected_shin).astype(int))
            else:
                corrected_ankle = ankle
            
            return knee, corrected_ankle
        
        return knee, ankle
    
    def _draw_refined_skeleton(self, skeleton: np.ndarray, refined_points: Dict) -> np.ndarray:
        """Draw refined skeleton with better proportions"""
        
        # Essential OpenPose connections (anatomically correct)
        connections = [
            # Head and neck
            (0, 1),   # Nose to neck
            
            # Arms (with proper proportions)
            (1, 2), (2, 3), (3, 4),  # Right arm: neck->shoulder->elbow->wrist
            (1, 5), (5, 6), (6, 7),  # Left arm: neck->shoulder->elbow->wrist
            
            # Torso structure
            (1, 8),  # Neck to mid-hip (main spine)
            (2, 9), (5, 12),  # Shoulders to respective hips
            
            # Legs (with proper proportions)
            (8, 9), (9, 10), (10, 11),  # Right leg: mid-hip->hip->knee->ankle
            (8, 12), (12, 13), (13, 14), # Left leg: mid-hip->hip->knee->ankle
        ]
        
        # Draw connections with varying thickness for importance
        for start_idx, end_idx in connections:
            if start_idx in refined_points and end_idx in refined_points:
                start_point = refined_points[start_idx]
                end_point = refined_points[end_idx]
                
                # Main structural lines thicker
                if (start_idx, end_idx) in [(0, 1), (1, 8)]:  # Head-neck-spine
                    thickness = 6
                elif start_idx == 1 or end_idx == 1:  # Connections to neck
                    thickness = 5
                else:
                    thickness = 4
                
                cv2.line(skeleton, start_point, end_point, (255, 255, 255), thickness)
        
        # Draw keypoints with importance-based sizes
        for idx, point in refined_points.items():
            if idx == 0:  # Head/nose
                cv2.circle(skeleton, point, 10, (255, 255, 255), -1)
            elif idx in [1, 8]:  # Neck and mid-hip (structural)
                cv2.circle(skeleton, point, 8, (255, 255, 255), -1)
            elif idx in [2, 5, 9, 12]:  # Shoulders and hips
                cv2.circle(skeleton, point, 7, (255, 255, 255), -1)
            else:  # Other joints
                cv2.circle(skeleton, point, 6, (255, 255, 255), -1)
        
        return skeleton

class OpenPoseBatchProcessor:
    """Batch processor using OpenPose skeletons"""
    
    def __init__(self):
        print("🚀 OpenPose Batch Processor")
        print("=" * 40)
        
        if not AI_AVAILABLE:
            print("❌ AI libraries not available!")
            return
        
        # Initialize components
        self.skeleton_generator = OpenPoseSkeleton()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load AI models
        self._load_ai_models()
        
        # Stats
        self.total_processed = 0
        self.successful_generations = 0
        self.failed_generations = 0
        
        print("✅ OpenPose batch processor ready!")
    
    def _load_ai_models(self):
        """Load AI models for generation"""
        try:
            print("📥 Loading AI models...")
            
            # Load ControlNet
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-openpose",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # Load pipeline
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
            self.pipe = self.pipe.to(self.device)
            
            # Optimizations
            if self.device == "cuda":
                try:
                    self.pipe.enable_model_cpu_offload()
                except Exception:
                    pass
            
            print(f"✅ AI models loaded on {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.pipe = None
    
    def _load_character(self, character_file: str) -> Optional[CharacterProfile]:
        """Load character profile"""
        try:
            with open(character_file, 'r') as f:
                character_data = json.load(f)
            
            if 'character_profile' in character_data:
                profile_data = character_data['character_profile']
            else:
                profile_data = character_data
            
            character = CharacterProfile(
                name=profile_data['name'],
                facial=FacialFeatures(**profile_data['facial']),
                body=BodyFeatures(**profile_data['body']),
                hair=HairFeatures(**profile_data['hair']),
                clothing=ClothingFeatures(**profile_data['clothing'])
            )
            
            print(f"✅ Character loaded: {character.name}")
            return character
            
        except Exception as e:
            print(f"❌ Error loading character: {e}")
            return None
    
    def _generate_optimized_prompt(self, character: CharacterProfile) -> str:
        """Generate optimized prompt for photorealistic results"""
        prompt_parts = []
        
        # More specific realism keywords
        prompt_parts.append("photorealistic portrait")
        prompt_parts.append("beautiful woman")
        prompt_parts.append("front view")  # Force front-facing
        
        # Character features
        age_factor = character.facial.age_factor
        if age_factor < 0.4:
            prompt_parts.append("young woman")
        
        # Hair (more specific)
        hair_color = character.hair.color
        if hair_color[0] > 0.8 and hair_color[1] > 0.7:
            prompt_parts.append("long blonde wavy hair")
        elif hair_color[0] < 0.3:
            prompt_parts.append("black hair")
        else:
            prompt_parts.append("brown hair")
        
        # Eyes (more vivid)
        eye_color = character.facial.eye_color
        if eye_color[2] > 0.7:
            prompt_parts.append("bright blue eyes")
        elif eye_color[1] > 0.6:
            prompt_parts.append("green eyes")
        else:
            prompt_parts.append("brown eyes")
        
        # Clothing
        clothing_styles = {
            "athletic_top": "blue sports bra",
            "dress_shirt": "white shirt",
            "casual_tshirt": "casual top",
            "sweater": "sweater"
        }
        clothing = clothing_styles.get(character.clothing.shirt_type, "blue athletic top")
        prompt_parts.append(clothing)
        
        # Enhanced realism terms
        prompt_parts.extend([
            "realistic skin texture", "professional photography",
            "natural lighting", "highly detailed", "8k quality"
        ])
        
        prompt = ", ".join(prompt_parts)
        print(f"📝 Enhanced prompt ({len(prompt.split())} words): {prompt}")
        return prompt
    
    def _generate_negative_prompt(self) -> str:
        """Generate enhanced negative prompt for realism"""
        return ("cartoon, anime, 3d render, cgi, digital art, "
                "artificial, plastic, doll, toy, game character, "
                "multiple people, extra person, floating limbs, "
                "malformed hands, blurry, low quality, deformed, "
                "back view, rear view, turned away")
    
    def _generate_human(self, skeleton: np.ndarray, character: CharacterProfile, image_name: str) -> Optional[np.ndarray]:
        """Generate AI human from OpenPose skeleton"""
        if not self.pipe:
            return None
        
        try:
            # Convert skeleton to PIL
            skeleton_pil = Image.fromarray(skeleton)
            
            # Generate prompts
            prompt = self._generate_optimized_prompt(character)
            negative_prompt = self._generate_negative_prompt()
            
            print(f"🎨 Generating human for: {image_name}")
            
            start_time = time.time()
            
            # Generate with enhanced settings for realism
            with torch.autocast(self.device):
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=skeleton_pil,
                    num_inference_steps=25,  # Higher for better quality
                    guidance_scale=8.0,      # Higher for better prompt following
                    controlnet_conditioning_scale=1.1,  # Stronger pose control
                    generator=torch.Generator(device=self.device).manual_seed(42),
                    width=512,
                    height=512,
                )
            
            generation_time = time.time() - start_time
            generated_image = np.array(result.images[0])
            
            print(f"✅ Generated in {generation_time:.1f}s")
            return generated_image
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return None
    
    def process_folder(self, input_folder: str, output_folder: str, character_file: str):
        """Process all images in folder"""
        prevent_sleep()
        
        try:
            # Setup
            input_path = Path(input_folder)
            output_path = Path(output_folder)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Load character
            character = self._load_character(character_file)
            if not character:
                return
            
            # Find images
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(input_path.glob(f"*{ext}"))
                image_files.extend(input_path.glob(f"*{ext.upper()}"))
            
            if not image_files:
                print(f"❌ No images found in {input_folder}")
                return
            
            print(f"\n🚀 OPENPOSE BATCH PROCESSING")
            print("=" * 45)
            print(f"📁 Input: {input_folder}")
            print(f"📁 Output: {output_folder}")
            print(f"👤 Character: {character.name}")
            print(f"📸 Images: {len(image_files)}")
            
            # Process each image
            start_time = time.time()
            
            for i, image_file in enumerate(image_files, 1):
                print(f"\n🔄 Processing {i}/{len(image_files)}: {image_file.name}")
                print("-" * 40)
                
                try:
                    # Generate OpenPose skeleton
                    skeleton = self.skeleton_generator.generate_openpose_skeleton(str(image_file))
                    
                    if skeleton is not None:
                        # Generate AI human
                        ai_result = self._generate_human(skeleton, character, image_file.name)
                        
                        if ai_result is not None:
                            # Save results
                            output_file = output_path / f"openpose_human_{image_file.stem}.png"
                            cv2.imwrite(str(output_file), cv2.cvtColor(ai_result, cv2.COLOR_RGB2BGR))
                            
                            # Save skeleton
                            skeleton_file = output_path / f"openpose_skeleton_{image_file.stem}.png"
                            cv2.imwrite(str(skeleton_file), skeleton)
                            
                            self.successful_generations += 1
                            print(f"✅ Generated: {output_file.name}")
                        else:
                            self.failed_generations += 1
                    else:
                        self.failed_generations += 1
                        print(f"❌ No skeleton generated")
                    
                    self.total_processed += 1
                    
                    # Progress
                    elapsed = time.time() - start_time
                    avg_time = elapsed / i
                    remaining = len(image_files) - i
                    eta_seconds = remaining * avg_time
                    
                    print(f"📊 Progress: {i}/{len(image_files)} ({i/len(image_files)*100:.1f}%)")
                    if remaining > 0:
                        print(f"⏰ ETA: {eta_seconds/60:.1f} minutes")
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                    self.failed_generations += 1
                    self.total_processed += 1
            
            # Final report
            total_time = time.time() - start_time
            success_rate = (self.successful_generations / self.total_processed * 100) if self.total_processed > 0 else 0
            
            print(f"\n🎉 PROCESSING COMPLETE!")
            print("=" * 30)
            print(f"⏰ Total time: {total_time/60:.1f} minutes")
            print(f"✅ Successful: {self.successful_generations}")
            print(f"❌ Failed: {self.failed_generations}")
            print(f"📊 Success rate: {success_rate:.1f}%")
            
        finally:
            allow_sleep()

def main():
    """Main function"""
    print("OpenPose-Based AI Human Generator")
    print("=" * 40)
    
    if not AI_AVAILABLE:
        print("❌ AI libraries required!")
        return
    
    if not MEDIAPIPE_AVAILABLE:
        print("❌ MediaPipe required! Install with: pip install mediapipe")
        return
    
    processor = OpenPoseBatchProcessor()
    
    if not processor.pipe:
        print("❌ AI models failed to load")
        return
    
    # Get inputs
    input_folder = input("📁 Input folder with photos: ").strip()
    if not Path(input_folder).exists():
        print("❌ Folder not found!")
        return
    
    output_folder = input("📁 Output folder: ").strip() or "openpose_humans"
    character_file = input("👤 Character JSON file: ").strip()
    
    if not Path(character_file).exists():
        print("❌ Character file not found!")
        return
    
    # Process
    processor.process_folder(input_folder, output_folder, character_file)

if __name__ == "__main__":
    main()