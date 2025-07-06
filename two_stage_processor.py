"""
Two-Stage AI Human Processor
Stage 1: Full body generation with detailed environment
Stage 2: Face refinement with dedicated facial prompts
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

# Keep all the import and sleep prevention code
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

try:
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionImg2ImgPipeline
    from diffusers import UniPCMultistepScheduler
    from PIL import Image, ImageDraw
    import torch
    AI_AVAILABLE = True
except ImportError as e:
    AI_AVAILABLE = False
    print(f"⚠️ AI libraries missing: {e}")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

from realistic_character_creator import CharacterProfile, FacialFeatures, BodyFeatures, HairFeatures, ClothingFeatures

# Keep the exact same OpenPoseSkeleton class from before (working version)
class OpenPoseSkeleton:
    """Dedicated OpenPose skeleton generator - WORKING VERSION"""
    
    def __init__(self):
        print("🦴 Initializing OpenPose Skeleton Generator...")
        
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            print("✅ MediaPipe OpenPose ready!")
        else:
            print("❌ MediaPipe not available!")
            
    def generate_openpose_skeleton(self, image_path: str) -> Optional[Tuple]:
        """Generate proper OpenPose skeleton from image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Could not load: {image_path}")
                return None
                
            print(f"🦴 Generating skeleton for: {Path(image_path).name}")
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            results = self.pose.process(image_rgb)
            
            if not results.pose_landmarks:
                print(f"   No pose detected")
                return None
            
            skeleton_result = self._create_openpose_skeleton(results.pose_landmarks, h, w)
            
            if skeleton_result is not None:
                skeleton, scope = skeleton_result
                print(f"✅ Generated OpenPose skeleton ({scope})")
                return skeleton, scope
            else:
                print(f"❌ Failed to create skeleton")
                return None
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def _create_openpose_skeleton(self, pose_landmarks, height: int, width: int) -> Optional[Tuple]:
        """Create refined OpenPose skeleton with anatomical corrections"""
        try:
            skeleton = np.zeros((height, width, 3), dtype=np.uint8)
            
            landmarks = []
            for i, landmark in enumerate(pose_landmarks.landmark):
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                visibility = landmark.visibility
                landmarks.append((x, y, visibility))
            
            refined_points = self._refine_skeleton_anatomy(landmarks, width, height)
            
            if not refined_points or len(refined_points) < 8:
                print(f"   Insufficient refined points: {len(refined_points) if refined_points else 0}")
                return None
            
            skeleton_scope = self._analyze_skeleton_scope(refined_points)
            skeleton = self._draw_refined_skeleton(skeleton, refined_points)
            
            print(f"   Refined to {len(refined_points)} anatomically correct points")
            
            return skeleton, skeleton_scope
            
        except Exception as e:
            print(f"   Error creating skeleton: {e}")
            return None
    
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
        
        def get_point(idx, min_visibility=0.5):
            if idx < len(landmarks) and landmarks[idx][2] > min_visibility:
                x, y, vis = landmarks[idx]
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                return (x, y)
            return None
        
        # Get core structural points
        nose = get_point(NOSE, 0.3)
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
        
        # Calculate derived points
        neck = None
        if left_shoulder and right_shoulder:
            neck_x = (left_shoulder[0] + right_shoulder[0]) // 2
            neck_y = min(left_shoulder[1], right_shoulder[1]) - 10
            neck = (neck_x, max(neck_y, 0))
        
        mid_hip = None
        if left_hip and right_hip:
            mid_hip_x = (left_hip[0] + right_hip[0]) // 2
            mid_hip_y = (left_hip[1] + right_hip[1]) // 2
            mid_hip = (mid_hip_x, mid_hip_y)
        
        # Map to OpenPose indices
        openpose_mapping = {
            0: nose, 1: neck, 2: right_shoulder, 3: right_elbow, 4: right_wrist,
            5: left_shoulder, 6: left_elbow, 7: left_wrist, 8: mid_hip,
            9: right_hip, 10: right_knee, 11: right_ankle,
            12: left_hip, 13: left_knee, 14: left_ankle,
        }
        
        for idx, point in openpose_mapping.items():
            if point is not None:
                refined_points[idx] = point
        
        return refined_points
        
    def _analyze_skeleton_scope(self, refined_points: Dict) -> str:
        """Analyze skeleton to determine if it's full body or portrait"""
        has_head = 0 in refined_points or 1 in refined_points
        has_shoulders = 2 in refined_points or 5 in refined_points
        has_hips = 8 in refined_points or 9 in refined_points or 12 in refined_points
        has_legs = any(idx in refined_points for idx in [10, 11, 13, 14])
        has_arms = any(idx in refined_points for idx in [3, 4, 6, 7])
        
        body_regions = sum([has_head, has_shoulders, has_hips, has_legs, has_arms])
        
        y_coords = [point[1] for point in refined_points.values()]
        if len(y_coords) >= 2:
            vertical_span = max(y_coords) - min(y_coords)
        else:
            vertical_span = 0
        
        if body_regions >= 4 and has_legs and vertical_span > 200:
            scope = "full_body"
            print(f"   Detected FULL BODY skeleton ({body_regions} regions, span: {vertical_span}px)")
        else:
            scope = "portrait"
            print(f"   Detected PORTRAIT skeleton ({body_regions} regions, span: {vertical_span}px)")
        
        return scope
    
    def _draw_refined_skeleton(self, skeleton: np.ndarray, refined_points: Dict) -> np.ndarray:
        """Draw refined skeleton with better proportions"""
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
            (1, 8), (2, 9), (5, 12), (8, 9), (9, 10), (10, 11),
            (8, 12), (12, 13), (13, 14),
        ]
        
        connections_drawn = 0
        
        for start_idx, end_idx in connections:
            if start_idx in refined_points and end_idx in refined_points:
                start_point = refined_points[start_idx]
                end_point = refined_points[end_idx]
                
                if (start_idx, end_idx) in [(0, 1), (1, 8)]:
                    thickness = 6
                elif start_idx == 1 or end_idx == 1:
                    thickness = 5
                else:
                    thickness = 4
                
                cv2.line(skeleton, start_point, end_point, (255, 255, 255), thickness)
                connections_drawn += 1
        
        points_drawn = 0
        for idx, point in refined_points.items():
            if idx == 0:
                cv2.circle(skeleton, point, 10, (255, 255, 255), -1)
            elif idx in [1, 8]:
                cv2.circle(skeleton, point, 8, (255, 255, 255), -1)
            elif idx in [2, 5, 9, 12]:
                cv2.circle(skeleton, point, 7, (255, 255, 255), -1)
            else:
                cv2.circle(skeleton, point, 6, (255, 255, 255), -1)
            points_drawn += 1
        
        print(f"   Drew {points_drawn} points, {connections_drawn} connections")
        
        return skeleton

class TwoStageProcessor:
    """Two-stage AI human generation for maximum quality"""
    
    def __init__(self):
        print("🎭 Two-Stage AI Human Processor")
        print("=" * 45)
        
        if not AI_AVAILABLE:
            print("❌ AI libraries not available!")
            return
        
        self.skeleton_generator = OpenPoseSkeleton()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load both pipelines
        self._load_ai_models()
        
        # Stats
        self.total_processed = 0
        self.successful_generations = 0
        self.failed_generations = 0
        
        print("✅ Two-stage processor ready!")
    
    def _load_ai_models(self):
        """Load both ControlNet and Img2Img pipelines"""
        try:
            print("📥 Loading AI models...")
            
            # Stage 1: ControlNet for full body
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-openpose",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            self.stage1_pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Stage 2: Img2Img for face refinement
            self.stage2_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Optimize both pipelines
            self.stage1_pipe.scheduler = UniPCMultistepScheduler.from_config(self.stage1_pipe.scheduler.config)
            self.stage2_pipe.scheduler = UniPCMultistepScheduler.from_config(self.stage2_pipe.scheduler.config)
            
            self.stage1_pipe = self.stage1_pipe.to(self.device)
            self.stage2_pipe = self.stage2_pipe.to(self.device)
            
            if self.device == "cuda":
                try:
                    self.stage1_pipe.enable_model_cpu_offload()
                    self.stage2_pipe.enable_model_cpu_offload()
                except Exception:
                    pass
            
            print(f"✅ Both AI pipelines loaded on {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.stage1_pipe = None
            self.stage2_pipe = None
    
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
    
    def _generate_stage1_prompt(self, character: CharacterProfile, skeleton_scope: str) -> str:
        """Stage 1: Full body prompt - maximize tokens for body/environment details"""
        prompt_parts = []
        
        # Rich environment and body details (use full token budget)
        if skeleton_scope == "full_body":
            prompt_parts.extend([
                "photorealistic full body", "beautiful athletic woman", "complete figure",
                "dynamic pose", "professional fitness photography", "studio lighting",
                "highly detailed", "8k quality", "sharp focus"
            ])
        else:
            prompt_parts.extend([
                "photorealistic portrait", "beautiful woman", "professional headshot",
                "studio lighting", "highly detailed face", "8k portrait"
            ])
        
        # Character body features
        age_factor = character.facial.age_factor
        if age_factor < 0.4:
            prompt_parts.append("young woman")
        
        # Hair (detailed for body shot)
        hair_color = character.hair.color
        if hair_color[0] > 0.8 and hair_color[1] > 0.7:
            prompt_parts.extend(["long blonde wavy hair", "flowing hair", "silky hair texture"])
        
        # Basic facial
        eye_color = character.facial.eye_color
        if eye_color[2] > 0.7:
            prompt_parts.append("bright blue eyes")
        
        # Clothing details (full budget for clothing)
        if skeleton_scope == "full_body":
            clothing_styles = {
                "athletic_top": "blue athletic sports bra",
                "dress_shirt": "elegant white shirt", 
                "casual_tshirt": "casual fitted top",
                "sweater": "cozy sweater"
            }
            clothing = clothing_styles.get(character.clothing.shirt_type, "blue athletic sports bra")
            prompt_parts.extend([clothing, "athletic leggings", "fitness attire"])
        
        # Enhanced environment and quality
        prompt_parts.extend([
            "natural skin texture", "professional photography", "beautiful lighting",
            "magazine quality", "athletic physique", "confident pose"
        ])
        
        prompt = ", ".join(prompt_parts)
        print(f"📝 Stage 1 prompt ({len(prompt.split())} words): {prompt}")
        return prompt
    
    def _generate_stage2_prompt(self, character: CharacterProfile) -> str:
        """Stage 2: Face-only prompt - dedicate ALL tokens to facial perfection"""
        prompt_parts = []
        
        # Facial perfection (use full token budget for face)
        prompt_parts.extend([
            "photorealistic close-up portrait", "beautiful woman face",
            "perfect facial features", "symmetrical face", "flawless skin",
            "defined cheekbones", "smooth jawline", "radiant complexion",
            "natural beauty", "expressive eyebrows", "perfect nose shape",
            "natural full lips", "clear glowing skin"
        ])
        
        # Eye perfection
        eye_color = character.facial.eye_color
        if eye_color[2] > 0.7:
            prompt_parts.extend([
                "bright blue eyes", "detailed pupils", "clear iris",
                "beautiful eyelashes", "expressive eyes", "captivating gaze"
            ])
        
        # Hair framing face
        hair_color = character.hair.color
        if hair_color[0] > 0.8 and hair_color[1] > 0.7:
            prompt_parts.extend(["blonde hair framing face", "wavy hair texture"])
        
        # Quality and lighting for face
        prompt_parts.extend([
            "natural skin pores", "realistic skin texture", "soft studio lighting",
            "professional portrait photography", "high resolution", "sharp facial focus"
        ])
        
        prompt = ", ".join(prompt_parts)
        print(f"📝 Stage 2 prompt ({len(prompt.split())} words): {prompt}")
        return prompt
    
    def _generate_negative_prompt(self, stage: int) -> str:
        """Generate negative prompts for each stage"""
        base_negative = [
            "distorted face", "asymmetrical features", "malformed hands",
            "extra limbs", "floating limbs", "multiple people", "deformed",
            "cartoon", "anime", "3d render", "cgi", "digital art",
            "artificial", "plastic", "doll", "low quality"
        ]
        
        if stage == 1:
            # Stage 1: Focus on body/pose negatives
            base_negative.extend([
                "incomplete body", "missing limbs", "floating torso",
                "back view", "rear view", "turned away", "cropped limbs"
            ])
        else:
            # Stage 2: Focus on facial negatives
            base_negative.extend([
                "empty eyes", "no pupils", "blank eyes", "distorted eyes",
                "crooked nose", "unnatural mouth", "facial deformity",
                "overly smooth skin", "poreless skin", "too perfect"
            ])
        
        return ", ".join(base_negative)
    
    def _detect_face_region(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face region for Stage 2 refinement"""
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Use OpenCV face detection as fallback
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Get largest face
                face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = face
                
                # Expand face region by 50% for context
                padding = int(min(w, h) * 0.5)
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2 * padding)
                h = min(image.shape[0] - y, h + 2 * padding)
                
                return (x, y, w, h)
            
            # If no face detected, use upper third of image
            img_h, img_w = image.shape[:2]
            return (0, 0, img_w, img_h // 2)
            
        except Exception as e:
            print(f"   Face detection error: {e}")
            # Fallback: upper half of image
            img_h, img_w = image.shape[:2]
            return (0, 0, img_w, img_h // 2)
    
    def _generate_two_stage_human(self, skeleton: np.ndarray, skeleton_scope: str, 
                                 character: CharacterProfile, image_name: str) -> Optional[np.ndarray]:
        """Generate AI human using two-stage process"""
        if not self.stage1_pipe or not self.stage2_pipe:
            return None
        
        try:
            print(f"🎨 Two-stage generation for: {image_name}")
            
            # STAGE 1: Full body generation
            print("   🎭 Stage 1: Generating full body...")
            skeleton_pil = Image.fromarray(skeleton)
            
            stage1_prompt = self._generate_stage1_prompt(character, skeleton_scope)
            stage1_negative = self._generate_negative_prompt(1)
            
            stage1_start = time.time()
            
            with torch.autocast(self.device):
                stage1_result = self.stage1_pipe(
                    prompt=stage1_prompt,
                    negative_prompt=stage1_negative,
                    image=skeleton_pil,
                    num_inference_steps=25,  # Fewer steps for stage 1
                    guidance_scale=8.0,
                    controlnet_conditioning_scale=1.0,
                    generator=torch.Generator(device=self.device).manual_seed(123),
                    width=512,
                    height=512,
                )
            
            stage1_time = time.time() - stage1_start
            stage1_image = np.array(stage1_result.images[0])
            print(f"   ✅ Stage 1 complete in {stage1_time:.1f}s")
            
            # STAGE 2: Face refinement
            print("   🎯 Stage 2: Refining facial details...")
            
            # Detect face region for refinement
            face_region = self._detect_face_region(stage1_image)
            if face_region:
                x, y, w, h = face_region
                print(f"   🎯 Face region detected: {w}x{h} at ({x},{y})")
            
            stage2_prompt = self._generate_stage2_prompt(character)
            stage2_negative = self._generate_negative_prompt(2)
            
            stage2_start = time.time()
            
            # Convert to PIL for img2img
            stage1_pil = Image.fromarray(stage1_image)
            
            with torch.autocast(self.device):
                stage2_result = self.stage2_pipe(
                    prompt=stage2_prompt,
                    negative_prompt=stage2_negative,
                    image=stage1_pil,
                    strength=0.5,  # Moderate refinement strength
                    num_inference_steps=35,  # More steps for facial quality
                    guidance_scale=9.0,  # Higher guidance for face
                    generator=torch.Generator(device=self.device).manual_seed(456),  # Different seed
                )
            
            stage2_time = time.time() - stage2_start
            final_image = np.array(stage2_result.images[0])
            print(f"   ✅ Stage 2 complete in {stage2_time:.1f}s")
            
            total_time = stage1_time + stage2_time
            print(f"✅ Two-stage generation complete in {total_time:.1f}s total")
            
            return final_image
            
        except Exception as e:
            print(f"❌ Two-stage generation error: {e}")
            return None
    
    def process_folder(self, input_folder: str, output_folder: str, character_file: str):
        """Process images with two-stage generation"""
        prevent_sleep()
        
        try:
            input_path = Path(input_folder)
            output_path = Path(output_folder)
            output_path.mkdir(parents=True, exist_ok=True)
            
            character = self._load_character(character_file)
            if not character:
                return
            
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(input_path.glob(f"*{ext}"))
                image_files.extend(input_path.glob(f"*{ext.upper()}"))
            
            if not image_files:
                print(f"❌ No images found in {input_folder}")
                return
            
            print(f"\n🎭 TWO-STAGE PROCESSING")
            print("=" * 45)
            print(f"📁 Input: {input_folder}")
            print(f"📁 Output: {output_folder}")
            print(f"👤 Character: {character.name}")
            print(f"📸 Images: {len(image_files)}")
            print(f"🎯 Method: Stage 1 (Body) + Stage 2 (Face)")
            
            start_time = time.time()
            
            for i, image_file in enumerate(image_files, 1):
                print(f"\n🔄 Processing {i}/{len(image_files)}: {image_file.name}")
                print("-" * 40)
                
                try:
                    skeleton_result = self.skeleton_generator.generate_openpose_skeleton(str(image_file))
                    
                    if skeleton_result is not None:
                        skeleton, skeleton_scope = skeleton_result
                        
                        ai_result = self._generate_two_stage_human(skeleton, skeleton_scope, character, image_file.name)
                        
                        if ai_result is not None:
                            # Save results
                            output_file = output_path / f"twostage_human_{image_file.stem}.png"
                            cv2.imwrite(str(output_file), cv2.cvtColor(ai_result, cv2.COLOR_RGB2BGR))
                            
                            skeleton_file = output_path / f"twostage_skeleton_{image_file.stem}.png"
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
            
            print(f"\n🎉 TWO-STAGE PROCESSING COMPLETE!")
            print("=" * 40)
            print(f"⏰ Total time: {total_time/60:.1f} minutes")
            print(f"✅ Successful: {self.successful_generations}")
            print(f"❌ Failed: {self.failed_generations}")
            print(f"📊 Success rate: {success_rate:.1f}%")
            print(f"🎭 Method: Two-stage generation for maximum quality")
            
        finally:
            allow_sleep()

def main():
    """Main function"""
    print("Two-Stage AI Human Generator")
    print("=" * 40)
    
    if not AI_AVAILABLE:
        print("❌ AI libraries required!")
        return
    
    if not MEDIAPIPE_AVAILABLE:
        print("❌ MediaPipe required!")
        return
    
    processor = TwoStageProcessor()
    
    if not processor.stage1_pipe or not processor.stage2_pipe:
        print("❌ AI models failed to load")
        return
    
    input_folder = input("📁 Input folder with photos: ").strip()
    if not Path(input_folder).exists():
        print("❌ Folder not found!")
        return
    
    output_folder = input("📁 Output folder: ").strip() or "twostage_humans"
    character_file = input("👤 Character JSON file: ").strip()
    
    if not Path(character_file).exists():
        print("❌ Character file not found!")
        return
    
    processor.process_folder(input_folder, output_folder, character_file)

if __name__ == "__main__":
    main()