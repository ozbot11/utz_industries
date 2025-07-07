"""
ENHANCED Five-Stage Face-Focused AI Human Processor
EXACT original code + ONLY improved face blending function
Stage 1: Foundation structure (UNCHANGED)
Stage 2: Arms refinement (UNCHANGED)
Stage 3: Hair perfection (UNCHANGED)
Stage 4: Body refinement (UNCHANGED)
Stage 5: Face crop + ENHANCED seamless blending
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

class EnhancedFiveStageProcessor:
    """ENHANCED Five-stage processor - EXACT original + better face blending ONLY"""
    
    def __init__(self):
        print("🎭 ENHANCED Five-Stage Face-Focused AI Human Processor")
        print("=" * 55)
        print("🎯 Original amazing code + ONLY improved face blending")
        
        if not AI_AVAILABLE:
            print("❌ AI libraries not available!")
            return
        
        self.skeleton_generator = OpenPoseSkeleton()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load all five pipelines - IDENTICAL TO ORIGINAL
        self._load_ai_models()
        
        # Stats
        self.total_processed = 0
        self.successful_generations = 0
        self.failed_generations = 0
        
        print("✅ Enhanced five-stage processor ready!")
    
    def _load_ai_models(self):
        """Load ControlNet and multiple Img2Img pipelines - IDENTICAL TO ORIGINAL"""
        try:
            print("📥 Loading AI models...")
            
            # Stage 1: ControlNet for foundation
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
            
            # Stages 2-5: Img2Img refinement
            self.stage2_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            self.stage3_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            self.stage4_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            self.stage5_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Optimize all pipelines
            for pipe in [self.stage1_pipe, self.stage2_pipe, self.stage3_pipe, self.stage4_pipe, self.stage5_pipe]:
                pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
                pipe = pipe.to(self.device)
            
            if self.device == "cuda":
                try:
                    self.stage1_pipe.enable_model_cpu_offload()
                    self.stage2_pipe.enable_model_cpu_offload()
                    self.stage3_pipe.enable_model_cpu_offload()
                    self.stage4_pipe.enable_model_cpu_offload()
                    self.stage5_pipe.enable_model_cpu_offload()
                except Exception:
                    pass
            
            print(f"✅ All five AI pipelines loaded on {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.stage1_pipe = None
            self.stage2_pipe = None
            self.stage3_pipe = None
            self.stage4_pipe = None
            self.stage5_pipe = None
    
    def _load_character(self, character_file: str) -> Optional[CharacterProfile]:
        """Load character profile - IDENTICAL TO ORIGINAL"""
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
        """Stage 1: Foundation - IDENTICAL TO ORIGINAL"""
        prompt_parts = []
        
        if skeleton_scope == "full_body":
            prompt_parts.extend([
                "photorealistic full body", "beautiful woman", "athletic pose"
            ])
        else:
            prompt_parts.extend([
                "photorealistic portrait", "beautiful woman"
            ])
        
        prompt_parts.extend([
            "young woman", "blonde hair", "blue eyes", "blue athletic wear"
        ])
        
        prompt_parts.extend([
            "professional photography", "studio lighting", "8k quality", "sharp focus"
        ])
        
        prompt = ", ".join(prompt_parts)
        word_count = len(prompt.split())
        print(f"📝 Stage 1 prompt ({word_count} words): {prompt}")
        return prompt
    
    def _generate_stage2_prompt(self, character: CharacterProfile) -> str:
        """Stage 2: Arms ONLY - IDENTICAL TO ORIGINAL"""
        prompt_parts = []
        
        # Enhanced arm anatomy
        prompt_parts.extend([
            "natural arm position", "graceful arms", "elegant arm curves",
            "anatomically correct arms", "smooth arm lines", "natural shoulder slope",
            "relaxed arm pose", "beautiful arm shape"
        ])
        
        prompt_parts.extend([
            "professional photography", "natural lighting", "high quality"
        ])
        
        prompt = ", ".join(prompt_parts)
        word_count = len(prompt.split())
        print(f"📝 Stage 2 prompt ({word_count} words): {prompt}")
        return prompt
    
    def _generate_stage3_prompt(self, character: CharacterProfile) -> str:
        """Stage 3: Hair perfection - IDENTICAL TO ORIGINAL"""
        prompt_parts = []
        
        hair_color = character.hair.color
        if hair_color[0] > 0.8 and hair_color[1] > 0.7:
            prompt_parts.extend([
                "gorgeous blonde hair", "silky hair texture", "flowing waves",
                "voluminous hair", "perfect hair styling", "hair highlights",
                "natural hair movement", "beautiful hair flow"
            ])
        
        prompt_parts.extend([
            "professional photography", "natural lighting", "high resolution"
        ])
        
        prompt = ", ".join(prompt_parts)
        word_count = len(prompt.split())
        print(f"📝 Stage 3 prompt ({word_count} words): {prompt}")
        return prompt
    
    def _generate_stage4_prompt(self, character: CharacterProfile) -> str:
        """Stage 4: Body refinement - IDENTICAL TO ORIGINAL"""
        prompt_parts = []
        
        prompt_parts.extend([
            "perfect body proportions", "athletic physique", "toned muscles",
            "natural skin texture", "healthy glow", "confident pose",
            "beautiful figure", "fitness model"
        ])
        
        prompt_parts.extend([
            "professional photography", "studio lighting", "high quality"
        ])
        
        prompt = ", ".join(prompt_parts)
        word_count = len(prompt.split())
        print(f"📝 Stage 4 prompt ({word_count} words): {prompt}")
        return prompt
    
    def _generate_stage5_face_prompt(self, character: CharacterProfile) -> str:
        """Stage 5: FACE ONLY - IDENTICAL TO ORIGINAL"""
        prompt_parts = []
        
        # Ultra-detailed face prompts
        prompt_parts.extend([
            "photorealistic human face", "perfect facial symmetry", "natural beauty",
            "realistic facial features", "human-like appearance"
        ])
        
        # Detailed eyes
        eye_color = character.facial.eye_color
        if eye_color[2] > 0.7:
            prompt_parts.extend([
                "bright blue human eyes", "realistic pupils", "natural iris texture",
                "human eyelashes", "lifelike gaze"
            ])
        
        # Human facial details
        prompt_parts.extend([
            "natural human skin", "real person", "portrait photography"
        ])
        
        prompt = ", ".join(prompt_parts)
        word_count = len(prompt.split())
        print(f"📝 Stage 5 FACE prompt ({word_count} words): {prompt}")
        return prompt
    
    def _generate_negative_prompt(self, stage: int) -> str:
        """Generate specialized negative prompts - IDENTICAL TO ORIGINAL"""
        base_negative = [
            "distorted", "deformed", "malformed", "low quality"
        ]
        
        if stage == 1:
            base_negative.extend([
                "incomplete body", "missing limbs", "bad proportions"
            ])
        elif stage == 2:
            base_negative.extend([
                "twisted arms", "unnatural arm position", "robotic arms",
                "stiff arms", "mechanical arms"
            ])
        elif stage == 3:
            base_negative.extend([
                "bad hair", "floating hair", "unnatural hair"
            ])
        elif stage == 4:
            base_negative.extend([
                "bad anatomy", "unrealistic proportions"
            ])
        else:  # Stage 5 - FACE
            base_negative.extend([
                "alien face", "non-human", "robotic face", "artificial face",
                "cartoon face", "anime face", "doll face", "plastic face",
                "empty eyes", "dead eyes", "artificial eyes", "fake eyes",
                "distorted facial features", "asymmetrical face"
            ])
        
        return ", ".join(base_negative)
    
    def _detect_and_crop_face(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple]]:
        """Detect face and crop it - IDENTICAL TO ORIGINAL"""
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Use OpenCV face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Get largest face
                face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = face
                
                # Expand face region by 30% for context - ORIGINAL VALUE
                padding = int(min(w, h) * 0.3)
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2 * padding)
                h = min(image.shape[0] - y, h + 2 * padding)
                
                # Crop face region
                face_crop = image[y:y+h, x:x+w]
                
                # Resize to 512x512 for processing
                face_resized = cv2.resize(face_crop, (512, 512))
                
                return face_resized, (x, y, w, h)
            
            return None, None
            
        except Exception as e:
            print(f"   Face detection error: {e}")
            return None, None
    
    def _paste_face_back_seamless(self, original_image: np.ndarray, new_face: np.ndarray, 
                                 face_coords: Tuple[int, int, int, int]) -> np.ndarray:
        """ENHANCED seamless face blending - THE ONLY CHANGED FUNCTION"""
        try:
            x, y, w, h = face_coords
            
            # Resize new face to match original crop size
            face_resized = cv2.resize(new_face, (w, h))
            
            # Convert to float for precise blending
            result = original_image.copy().astype(np.float32)
            face_float = face_resized.astype(np.float32)
            
            # Create sophisticated blending mask
            mask = np.zeros((h, w), dtype=np.float32)
            
            # Create elliptical mask centered on face
            center_x, center_y = w // 2, h // 2
            
            # Ellipse size - covers most of face but not edges
            ellipse_w = int(w * 0.75)  # 75% of width
            ellipse_h = int(h * 0.8)   # 80% of height
            
            # Create solid elliptical core
            cv2.ellipse(mask, (center_x, center_y), (ellipse_w // 2, ellipse_h // 2), 0, 0, 360, 1.0, -1)
            
            # Apply distance transform for smooth falloff
            mask_binary = mask.copy()
            mask_binary[mask_binary > 0] = 255
            mask_binary = mask_binary.astype(np.uint8)
            
            # Create smooth distance-based gradient
            dist_transform = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
            
            # Normalize distance transform and apply smooth falloff
            if dist_transform.max() > 0:
                dist_transform = dist_transform / dist_transform.max()
                
                # Apply smooth sigmoid function for natural falloff
                mask = np.tanh(dist_transform * 3.0)  # Tanh for smooth S-curve
                
                # Ensure mask values are between 0 and 1
                mask = np.clip(mask, 0, 1)
            
            # Apply additional Gaussian smoothing for ultra-smooth edges
            mask = cv2.GaussianBlur(mask, (15, 15), 0)
            
            # Expand mask to 3 channels
            mask_3d = np.stack([mask, mask, mask], axis=2)
            
            # Extract regions
            original_region = result[y:y+h, x:x+w]
            
            # Sophisticated blending with edge preservation
            # Use different blending modes for different areas
            
            # Core blending (where mask is strong)
            core_mask = mask_3d * 0.9  # Slightly reduce strength for naturalness
            edge_mask = 1 - core_mask
            
            # Blend with smooth transition
            blended_region = face_float * core_mask + original_region * edge_mask
            
            # Apply color correction for seamless integration
            # Match average color tone between regions
            face_mean = np.mean(face_float, axis=(0, 1))
            orig_mean = np.mean(original_region, axis=(0, 1))
            
            # Subtle color correction only in transition areas
            color_diff = orig_mean - face_mean
            color_correction = color_diff * (1 - mask_3d) * 0.3  # Gentle correction
            
            blended_region += color_correction
            
            # Paste back blended region
            result[y:y+h, x:x+w] = blended_region
            
            # Convert back to uint8
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            print(f"   ✨ Enhanced seamless blending applied with elliptical gradient mask")
            return result
            
        except Exception as e:
            print(f"   Enhanced blending error: {e}")
            # Fallback to original method
            result = original_image.copy()
            face_resized = cv2.resize(new_face, (w, h))
            result[y:y+h, x:x+w] = face_resized
            return result
    
    def _generate_five_stage_human(self, skeleton: np.ndarray, skeleton_scope: str, 
                                  character: CharacterProfile, image_name: str) -> Optional[np.ndarray]:
        """Generate AI human - ALL STAGES IDENTICAL TO ORIGINAL + enhanced blending"""
        if not all([self.stage1_pipe, self.stage2_pipe, self.stage3_pipe, self.stage4_pipe, self.stage5_pipe]):
            return None
        
        try:
            print(f"🎨 ENHANCED Five-stage generation for: {image_name}")
            total_start = time.time()
            
            # STAGE 1: Foundation structure - IDENTICAL TO ORIGINAL
            print("   🏗️ Stage 1: Foundation structure (original)...")
            skeleton_pil = Image.fromarray(skeleton)
            
            stage1_prompt = self._generate_stage1_prompt(character, skeleton_scope)
            stage1_negative = self._generate_negative_prompt(1)
            
            stage1_start = time.time()
            
            with torch.autocast(self.device):
                stage1_result = self.stage1_pipe(
                    prompt=stage1_prompt,
                    negative_prompt=stage1_negative,
                    image=skeleton_pil,
                    num_inference_steps=20,  # SAME
                    guidance_scale=7.5,      # SAME
                    controlnet_conditioning_scale=1.0,  # SAME
                    generator=torch.Generator(device=self.device).manual_seed(123),  # SAME
                    width=512,
                    height=512,
                )
            
            stage1_time = time.time() - stage1_start
            stage1_image = np.array(stage1_result.images[0])
            print(f"   ✅ Stage 1 complete in {stage1_time:.1f}s")
            
            # STAGE 2: Arms refinement - IDENTICAL TO ORIGINAL
            print("   💪 Stage 2: Natural arm curves (original)...")
            
            stage2_prompt = self._generate_stage2_prompt(character)
            stage2_negative = self._generate_negative_prompt(2)
            
            stage2_start = time.time()
            
            stage1_pil = Image.fromarray(stage1_image)
            
            with torch.autocast(self.device):
                stage2_result = self.stage2_pipe(
                    prompt=stage2_prompt,
                    negative_prompt=stage2_negative,
                    image=stage1_pil,
                    strength=0.4,  # SAME
                    num_inference_steps=25,  # SAME
                    guidance_scale=8.0,      # SAME
                    generator=torch.Generator(device=self.device).manual_seed(456),  # SAME
                )
            
            stage2_time = time.time() - stage2_start
            stage2_image = np.array(stage2_result.images[0])
            print(f"   ✅ Stage 2 complete in {stage2_time:.1f}s")
            
            # STAGE 3: Hair perfection - IDENTICAL TO ORIGINAL
            print("   💇 Stage 3: Hair perfection (original)...")
            
            stage3_prompt = self._generate_stage3_prompt(character)
            stage3_negative = self._generate_negative_prompt(3)
            
            stage3_start = time.time()
            
            stage2_pil = Image.fromarray(stage2_image)
            
            with torch.autocast(self.device):
                stage3_result = self.stage3_pipe(
                    prompt=stage3_prompt,
                    negative_prompt=stage3_negative,
                    image=stage2_pil,
                    strength=0.3,   # SAME
                    num_inference_steps=20,  # SAME
                    guidance_scale=8.0,      # SAME
                    generator=torch.Generator(device=self.device).manual_seed(789),  # SAME
                )
            
            stage3_time = time.time() - stage3_start
            stage3_image = np.array(stage3_result.images[0])
            print(f"   ✅ Stage 3 complete in {stage3_time:.1f}s")
            
            # STAGE 4: Body refinement - IDENTICAL TO ORIGINAL
            print("   🏃 Stage 4: Body refinement (original)...")
            
            stage4_prompt = self._generate_stage4_prompt(character)
            stage4_negative = self._generate_negative_prompt(4)
            
            stage4_start = time.time()
            
            stage3_pil = Image.fromarray(stage3_image)
            
            with torch.autocast(self.device):
                stage4_result = self.stage4_pipe(
                    prompt=stage4_prompt,
                    negative_prompt=stage4_negative,
                    image=stage3_pil,
                    strength=0.3,   # SAME
                    num_inference_steps=20,  # SAME
                    guidance_scale=7.5,      # SAME
                    generator=torch.Generator(device=self.device).manual_seed(888),  # SAME
                )
            
            stage4_time = time.time() - stage4_start
            stage4_image = np.array(stage4_result.images[0])
            print(f"   ✅ Stage 4 complete in {stage4_time:.1f}s")
            
            # STAGE 5: Face generation - SAME GENERATION + ENHANCED BLENDING
            print("   👤 Stage 5: Face generation + ENHANCED blending...")
            
            # Detect and crop face - SAME AS ORIGINAL
            face_crop, face_coords = self._detect_and_crop_face(stage4_image)
            
            if face_crop is not None and face_coords is not None:
                print(f"   🎯 Face detected and cropped: {face_coords[2]}x{face_coords[3]}")
                
                stage5_prompt = self._generate_stage5_face_prompt(character)
                stage5_negative = self._generate_negative_prompt(5)
                
                stage5_start = time.time()
                
                face_pil = Image.fromarray(face_crop)
                
                with torch.autocast(self.device):
                    stage5_result = self.stage5_pipe(
                        prompt=stage5_prompt,
                        negative_prompt=stage5_negative,
                        image=face_pil,
                        strength=0.7,  # SAME AS ORIGINAL
                        num_inference_steps=40,  # SAME AS ORIGINAL
                        guidance_scale=12.0,  # SAME AS ORIGINAL
                        generator=torch.Generator(device=self.device).manual_seed(999),  # SAME
                    )
                
                stage5_time = time.time() - stage5_start
                new_face = np.array(stage5_result.images[0])
                
                # ENHANCED seamless face blending - ONLY CHANGE!
                final_image = self._paste_face_back_seamless(stage4_image, new_face, face_coords)
                print(f"   ✅ Stage 5 + ENHANCED blending complete in {stage5_time:.1f}s")
                
            else:
                print("   ❌ No face detected, using Stage 4 result")
                final_image = stage4_image
                stage5_time = 0
            
            total_time = time.time() - total_start
            print(f"🌟 ENHANCED Five-stage complete in {total_time:.1f}s total")
            print(f"✨ Same amazing quality + seamless face blending!")
            
            return final_image
            
        except Exception as e:
            print(f"❌ Five-stage generation error: {e}")
            return None
    
    def process_folder(self, input_folder: str, output_folder: str, character_file: str):
        """Process images - IDENTICAL TO ORIGINAL"""
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
            
            print(f"\n🌟 ENHANCED FIVE-STAGE PROCESSING")
            print("=" * 60)
            print(f"📁 Input: {input_folder}")
            print(f"📁 Output: {output_folder}")
            print(f"👤 Character: {character.name}")
            print(f"📸 Images: {len(image_files)}")
            print(f"🎯 Method: Original amazing code + ENHANCED face blending")
            print(f"✨ Goal: Keep all quality + fix the seam!")
            
            start_time = time.time()
            
            for i, image_file in enumerate(image_files, 1):
                print(f"\n🔄 Processing {i}/{len(image_files)}: {image_file.name}")
                print("-" * 60)
                
                try:
                    skeleton_result = self.skeleton_generator.generate_openpose_skeleton(str(image_file))
                    
                    if skeleton_result is not None:
                        skeleton, skeleton_scope = skeleton_result
                        
                        ai_result = self._generate_five_stage_human(skeleton, skeleton_scope, character, image_file.name)
                        
                        if ai_result is not None:
                            # Save results
                            output_file = output_path / f"enhanced_human_{image_file.stem}.png"
                            cv2.imwrite(str(output_file), cv2.cvtColor(ai_result, cv2.COLOR_RGB2BGR))
                            
                            skeleton_file = output_path / f"enhanced_skeleton_{image_file.stem}.png"
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
            
            print(f"\n🌟 ENHANCED PROCESSING COMPLETE!")
            print("=" * 55)
            print(f"⏰ Total time: {total_time/60:.1f} minutes")
            print(f"✅ Successful: {self.successful_generations}")
            print(f"❌ Failed: {self.failed_generations}")
            print(f"📊 Success rate: {success_rate:.1f}%")
            print(f"🎭 Method: Original code + enhanced seamless blending")
            print(f"✨ Amazing quality preserved + no more face seams!")
            
        finally:
            allow_sleep()

def main():
    """Main function"""
    print("ENHANCED Five-Stage Face-Focused AI Human Generator")
    print("=" * 55)
    print("🎯 Original amazing code + ONLY better face blending")
    
    if not AI_AVAILABLE:
        print("❌ AI libraries required!")
        return
    
    if not MEDIAPIPE_AVAILABLE:
        print("❌ MediaPipe required!")
        return
    
    processor = EnhancedFiveStageProcessor()
    
    if not all([processor.stage1_pipe, processor.stage2_pipe, processor.stage3_pipe, processor.stage4_pipe, processor.stage5_pipe]):
        print("❌ AI models failed to load")
        return
    
    input_folder = input("📁 Input folder with photos: ").strip()
    if not Path(input_folder).exists():
        print("❌ Folder not found!")
        return
    
    output_folder = input("📁 Output folder: ").strip() or "enhanced_humans"
    character_file = input("👤 Character JSON file: ").strip()
    
    if not Path(character_file).exists():
        print("❌ Character file not found!")
        return
    
    print(f"\n✨ Running ENHANCED version - original quality + seamless blending...")
    processor.process_folder(input_folder, output_folder, character_file)

if __name__ == "__main__":
    main()