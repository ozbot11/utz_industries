"""
Three-Stage Ultra-Refined AI Human Processor
Stage 1: Full body generation with pose control
Stage 2: Upper body refinement (arms, hands, torso) 
Stage 3: Face and hair perfection
Maximum quality through specialized refinement layers!
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

# Sleep prevention (keep existing code)
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

# Keep the exact same working OpenPoseSkeleton class
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

class ThreeStageProcessor:
    """Three-stage ultra-refined AI human generation"""
    
    def __init__(self):
        print("🎭 Three-Stage Ultra AI Human Processor")
        print("=" * 50)
        
        if not AI_AVAILABLE:
            print("❌ AI libraries not available!")
            return
        
        self.skeleton_generator = OpenPoseSkeleton()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load all three pipelines
        self._load_ai_models()
        
        # Stats
        self.total_processed = 0
        self.successful_generations = 0
        self.failed_generations = 0
        
        print("✅ Three-stage processor ready!")
    
    def _load_ai_models(self):
        """Load ControlNet and multiple Img2Img pipelines"""
        try:
            print("📥 Loading AI models...")
            
            # Stage 1: ControlNet for full body structure
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
            
            # Stage 2: Img2Img for upper body refinement
            self.stage2_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Stage 3: Img2Img for face and hair perfection
            self.stage3_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Optimize all pipelines
            for pipe in [self.stage1_pipe, self.stage2_pipe, self.stage3_pipe]:
                pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
                pipe = pipe.to(self.device)
            
            if self.device == "cuda":
                try:
                    self.stage1_pipe.enable_model_cpu_offload()
                    self.stage2_pipe.enable_model_cpu_offload()
                    self.stage3_pipe.enable_model_cpu_offload()
                except Exception:
                    pass
            
            print(f"✅ All three AI pipelines loaded on {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.stage1_pipe = None
            self.stage2_pipe = None
            self.stage3_pipe = None
    
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
        """Stage 1: Foundation body structure - focus on pose, lighting, environment"""
        prompt_parts = []
        
        # Core foundation
        if skeleton_scope == "full_body":
            prompt_parts.extend([
                "photorealistic full body portrait", "beautiful woman",
                "complete figure", "dynamic athletic pose"
            ])
        else:
            prompt_parts.extend([
                "photorealistic portrait", "beautiful woman"
            ])
        
        # Character basics
        age_factor = character.facial.age_factor
        if age_factor < 0.4:
            prompt_parts.append("young woman")
        
        # Basic features
        hair_color = character.hair.color
        if hair_color[0] > 0.8 and hair_color[1] > 0.7:
            prompt_parts.append("blonde hair")
        
        eye_color = character.facial.eye_color
        if eye_color[2] > 0.7:
            prompt_parts.append("blue eyes")
        
        # Basic clothing
        if skeleton_scope == "full_body":
            prompt_parts.extend(["blue athletic wear", "fitness outfit"])
        
        # Environment and lighting (use remaining tokens)
        prompt_parts.extend([
            "professional photography", "studio lighting", "dramatic lighting",
            "highly detailed", "8k quality", "sharp focus"
        ])
        
        prompt = ", ".join(prompt_parts)
        print(f"📝 Stage 1 prompt ({len(prompt.split())} words): {prompt}")
        return prompt
    
    def _generate_stage2_prompt(self, character: CharacterProfile) -> str:
        """Stage 2: Upper body perfection - arms, hands, torso, shoulders"""
        prompt_parts = []
        
        # Upper body focus
        prompt_parts.extend([
            "perfect upper body", "beautiful arms", "elegant hands",
            "graceful arm position", "natural hand pose", "defined shoulders",
            "athletic torso", "muscular definition", "perfect anatomy"
        ])
        
        # Detailed arm and hand features
        prompt_parts.extend([
            "five fingers", "detailed hands", "natural finger position",
            "smooth arm muscles", "elegant arm curves", "perfect proportions",
            "realistic skin texture", "natural arm movement"
        ])
        
        # Character consistency
        hair_color = character.hair.color
        if hair_color[0] > 0.8 and hair_color[1] > 0.7:
            prompt_parts.append("flowing blonde hair")
        
        # Quality
        prompt_parts.extend([
            "professional photography", "natural lighting", "high resolution"
        ])
        
        prompt = ", ".join(prompt_parts)
        print(f"📝 Stage 2 prompt ({len(prompt.split())} words): {prompt}")
        return prompt
    
    def _generate_stage3_prompt(self, character: CharacterProfile) -> str:
        """Stage 3: Face and hair perfection - maximum facial beauty"""
        prompt_parts = []
        
        # Facial perfection
        prompt_parts.extend([
            "perfect facial features", "beautiful face", "symmetrical features",
            "defined cheekbones", "smooth jawline", "radiant complexion",
            "flawless skin", "natural beauty", "expressive eyebrows"
        ])
        
        # Eye perfection
        eye_color = character.facial.eye_color
        if eye_color[2] > 0.7:
            prompt_parts.extend([
                "bright blue eyes", "detailed pupils", "clear iris",
                "beautiful eyelashes", "expressive eyes", "captivating gaze"
            ])
        
        # Hair perfection
        hair_color = character.hair.color
        if hair_color[0] > 0.8 and hair_color[1] > 0.7:
            prompt_parts.extend([
                "gorgeous blonde hair", "silky hair texture", "flowing waves",
                "voluminous hair", "perfect hair styling"
            ])
        
        # Skin perfection
        prompt_parts.extend([
            "natural skin pores", "realistic skin texture", "healthy glow"
        ])
        
        # Quality
        prompt_parts.extend([
            "portrait photography", "soft lighting", "high resolution"
        ])
        
        prompt = ", ".join(prompt_parts)
        print(f"📝 Stage 3 prompt ({len(prompt.split())} words): {prompt}")
        return prompt
    
    def _generate_negative_prompt(self, stage: int) -> str:
        """Generate specialized negative prompts for each stage"""
        base_negative = [
            "distorted", "deformed", "malformed", "low quality",
            "cartoon", "anime", "3d render", "artificial"
        ]
        
        if stage == 1:
            # Stage 1: Body structure negatives
            base_negative.extend([
                "incomplete body", "missing limbs", "floating parts",
                "bad proportions", "cropped limbs"
            ])
        elif stage == 2:
            # Stage 2: Arms and hands negatives
            base_negative.extend([
                "bad hands", "malformed hands", "extra fingers", "missing fingers",
                "floating hands", "disconnected arms", "twisted arms",
                "unnatural arm position", "deformed fingers"
            ])
        else:  # Stage 3
            # Stage 3: Face and hair negatives
            base_negative.extend([
                "asymmetrical face", "distorted face", "bad eyes",
                "empty eyes", "no pupils", "bad hair", "floating hair",
                "overly smooth skin", "plastic skin"
            ])
        
        return ", ".join(base_negative)
    
    def _generate_three_stage_human(self, skeleton: np.ndarray, skeleton_scope: str, 
                                   character: CharacterProfile, image_name: str) -> Optional[np.ndarray]:
        """Generate AI human using three-stage process"""
        if not all([self.stage1_pipe, self.stage2_pipe, self.stage3_pipe]):
            return None
        
        try:
            print(f"🎨 Three-stage generation for: {image_name}")
            total_start = time.time()
            
            # STAGE 1: Foundation body structure
            print("   🏗️ Stage 1: Building foundation structure...")
            skeleton_pil = Image.fromarray(skeleton)
            
            stage1_prompt = self._generate_stage1_prompt(character, skeleton_scope)
            stage1_negative = self._generate_negative_prompt(1)
            
            stage1_start = time.time()
            
            with torch.autocast(self.device):
                stage1_result = self.stage1_pipe(
                    prompt=stage1_prompt,
                    negative_prompt=stage1_negative,
                    image=skeleton_pil,
                    num_inference_steps=20,  # Fewer steps for foundation
                    guidance_scale=7.5,
                    controlnet_conditioning_scale=1.0,
                    generator=torch.Generator(device=self.device).manual_seed(123),
                    width=512,
                    height=512,
                )
            
            stage1_time = time.time() - stage1_start
            stage1_image = np.array(stage1_result.images[0])
            print(f"   ✅ Stage 1 complete in {stage1_time:.1f}s")
            
            # STAGE 2: Upper body refinement (arms, hands, torso)
            print("   💪 Stage 2: Refining arms and hands...")
            
            stage2_prompt = self._generate_stage2_prompt(character)
            stage2_negative = self._generate_negative_prompt(2)
            
            stage2_start = time.time()
            
            stage1_pil = Image.fromarray(stage1_image)
            
            with torch.autocast(self.device):
                stage2_result = self.stage2_pipe(
                    prompt=stage2_prompt,
                    negative_prompt=stage2_negative,
                    image=stage1_pil,
                    strength=0.4,  # Moderate refinement to preserve structure
                    num_inference_steps=25,
                    guidance_scale=8.0,
                    generator=torch.Generator(device=self.device).manual_seed(456),  # Different seed
                )
            
            stage2_time = time.time() - stage2_start
            stage2_image = np.array(stage2_result.images[0])
            print(f"   ✅ Stage 2 complete in {stage2_time:.1f}s")
            
            # STAGE 3: Face and hair perfection
            print("   ✨ Stage 3: Perfecting face and hair...")
            
            stage3_prompt = self._generate_stage3_prompt(character)
            stage3_negative = self._generate_negative_prompt(3)
            
            stage3_start = time.time()
            
            stage2_pil = Image.fromarray(stage2_image)
            
            with torch.autocast(self.device):
                stage3_result = self.stage3_pipe(
                    prompt=stage3_prompt,
                    negative_prompt=stage3_negative,
                    image=stage2_pil,
                    strength=0.3,  # Light refinement to preserve previous work
                    num_inference_steps=30,  # More steps for final quality
                    guidance_scale=8.5,
                    generator=torch.Generator(device=self.device).manual_seed(789),  # Another different seed
                )
            
            stage3_time = time.time() - stage3_start
            final_image = np.array(stage3_result.images[0])
            print(f"   ✅ Stage 3 complete in {stage3_time:.1f}s")
            
            total_time = time.time() - total_start
            print(f"🌟 Three-stage generation complete in {total_time:.1f}s total")
            print(f"   Breakdown: S1:{stage1_time:.1f}s + S2:{stage2_time:.1f}s + S3:{stage3_time:.1f}s")
            
            return final_image
            
        except Exception as e:
            print(f"❌ Three-stage generation error: {e}")
            return None
    
    def process_folder(self, input_folder: str, output_folder: str, character_file: str):
        """Process images with three-stage generation"""
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
            
            print(f"\n🌟 THREE-STAGE ULTRA PROCESSING")
            print("=" * 50)
            print(f"📁 Input: {input_folder}")
            print(f"📁 Output: {output_folder}")
            print(f"👤 Character: {character.name}")
            print(f"📸 Images: {len(image_files)}")
            print(f"🎯 Method: Stage 1 (Structure) + Stage 2 (Arms/Hands) + Stage 3 (Face/Hair)")
            
            start_time = time.time()
            
            for i, image_file in enumerate(image_files, 1):
                print(f"\n🔄 Processing {i}/{len(image_files)}: {image_file.name}")
                print("-" * 50)
                
                try:
                    skeleton_result = self.skeleton_generator.generate_openpose_skeleton(str(image_file))
                    
                    if skeleton_result is not None:
                        skeleton, skeleton_scope = skeleton_result
                        
                        ai_result = self._generate_three_stage_human(skeleton, skeleton_scope, character, image_file.name)
                        
                        if ai_result is not None:
                            # Save results
                            output_file = output_path / f"threestage_human_{image_file.stem}.png"
                            cv2.imwrite(str(output_file), cv2.cvtColor(ai_result, cv2.COLOR_RGB2BGR))
                            
                            skeleton_file = output_path / f"threestage_skeleton_{image_file.stem}.png"
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
            
            print(f"\n🌟 THREE-STAGE PROCESSING COMPLETE!")
            print("=" * 45)
            print(f"⏰ Total time: {total_time/60:.1f} minutes")
            print(f"✅ Successful: {self.successful_generations}")
            print(f"❌ Failed: {self.failed_generations}")
            print(f"📊 Success rate: {success_rate:.1f}%")
            print(f"🎭 Method: Three-stage ultra-refinement")
            print(f"🎯 Specialized: Structure + Arms/Hands + Face/Hair")
            
        finally:
            allow_sleep()

def main():
    """Main function"""
    print("Three-Stage Ultra AI Human Generator")
    print("=" * 45)
    
    if not AI_AVAILABLE:
        print("❌ AI libraries required!")
        return
    
    if not MEDIAPIPE_AVAILABLE:
        print("❌ MediaPipe required!")
        return
    
    processor = ThreeStageProcessor()
    
    if not all([processor.stage1_pipe, processor.stage2_pipe, processor.stage3_pipe]):
        print("❌ AI models failed to load")
        return
    
    input_folder = input("📁 Input folder with photos: ").strip()
    if not Path(input_folder).exists():
        print("❌ Folder not found!")
        return
    
    output_folder = input("📁 Output folder: ").strip() or "threestage_humans"
    character_file = input("👤 Character JSON file: ").strip()
    
    if not Path(character_file).exists():
        print("❌ Character file not found!")
        return
    
    processor.process_folder(input_folder, output_folder, character_file)

if __name__ == "__main__":
    main()