"""
Static Image Batch Processor for AI Human Generation
Processes folders of input photos overnight using ControlNet + Stable Diffusion
Perfect for generating AI humans from your existing poses!
"""

import cv2
import numpy as np
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import asdict
import threading
from datetime import datetime
import glob
import platform

# Prevent system sleep during processing
try:
    if platform.system() == "Windows":
        import ctypes
        from ctypes import wintypes
        
        # Windows constants for preventing sleep
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_DISPLAY_REQUIRED = 0x00000002
        
        def prevent_sleep():
            """Prevent Windows from going to sleep"""
            ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
            )
            print("🔒 Sleep prevention activated - laptop will stay awake")
        
        def allow_sleep():
            """Allow Windows to sleep again"""
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            print("🔓 Sleep prevention deactivated - laptop can sleep again")
    
    elif platform.system() == "Darwin":  # macOS
        import subprocess
        
        caffeinate_process = None
        
        def prevent_sleep():
            global caffeinate_process
            caffeinate_process = subprocess.Popen(['caffeinate', '-d'])
            print("🔒 Sleep prevention activated - Mac will stay awake")
        
        def allow_sleep():
            global caffeinate_process
            if caffeinate_process:
                caffeinate_process.terminate()
                caffeinate_process = None
            print("🔓 Sleep prevention deactivated - Mac can sleep again")
    
    else:  # Linux
        def prevent_sleep():
            print("🔒 Linux detected - please disable sleep manually or install 'systemd-inhibit'")
            print("   Alternative: Run with 'systemd-inhibit --what=sleep python batch_image_processor.py'")
        
        def allow_sleep():
            print("🔓 Sleep prevention not implemented for Linux")
            
except ImportError:
    def prevent_sleep():
        print("🔒 Sleep prevention not available - please keep computer awake manually")
    
    def allow_sleep():
        pass

# AI Generation imports
try:
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    from diffusers import UniPCMultistepScheduler
    from PIL import Image, ImageDraw
    import torch
    AI_AVAILABLE = True
    print("✅ AI libraries loaded successfully!")
except ImportError as e:
    AI_AVAILABLE = False
    print(f"⚠️ AI libraries missing: {e}")

# Import our existing systems
from complete_human_tracker import CompleteHumanTracker
from realistic_character_creator import CharacterProfile, FacialFeatures, BodyFeatures, HairFeatures, ClothingFeatures

class StaticPoseExtractor:
    """Extract poses from static images for ControlNet"""
    
    def __init__(self):
        print("📷 Initializing Static Pose Extractor...")
        self.tracker = CompleteHumanTracker()
        print("✅ Pose extractor ready!")
    
    def extract_pose_from_image(self, image_path: str) -> Optional[np.ndarray]:
        """Extract OpenPose-style skeleton from static image"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Could not load image: {image_path}")
                return None
            
            print(f"📸 Processing image: {Path(image_path).name}")
            
            # Process with tracker
            tracking_data = self.tracker.process_frame(image)
            
            if not tracking_data or not tracking_data.body_keypoints_2d:
                print(f"⚠️ No pose detected in: {Path(image_path).name}")
                return None
            
            # Convert to OpenPose format
            h, w = image.shape[:2]
            pose_image = self._convert_to_openpose_image(tracking_data, (h, w))
            
            print(f"✅ Pose extracted from: {Path(image_path).name}")
            return pose_image
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            return None
    
    def _convert_to_openpose_image(self, tracking_data, frame_shape: Tuple[int, int]) -> np.ndarray:
        """Convert tracking data to OpenPose image format"""
        h, w = frame_shape
        
        # Create blank pose image
        pose_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        if not tracking_data.body_keypoints_2d:
            return pose_image
        
        keypoints = tracking_data.body_keypoints_2d
        
        # COCO to OpenPose mapping (improved)
        coco_to_openpose = {
            0: 0,   # nose -> nose
            1: 15,  # left_eye -> left_eye  
            2: 14,  # right_eye -> right_eye
            3: 17,  # left_ear -> left_ear
            4: 16,  # right_ear -> right_ear
            5: 5,   # left_shoulder -> left_shoulder
            6: 2,   # right_shoulder -> right_shoulder
            7: 6,   # left_elbow -> left_elbow
            8: 3,   # right_elbow -> right_elbow
            9: 7,   # left_wrist -> left_wrist
            10: 4,  # right_wrist -> right_wrist
            11: 11, # left_hip -> left_hip
            12: 8,  # right_hip -> right_hip
            13: 12, # left_knee -> left_knee
            14: 9,  # right_knee -> right_knee
            15: 13, # left_ankle -> left_ankle
            16: 10  # right_ankle -> right_ankle
        }
        
        # Extract COCO keypoints with lower confidence threshold for head
        coco_points = {}
        for coco_idx in range(17):
            if coco_idx * 3 + 2 < len(keypoints):
                x = keypoints[coco_idx * 3]
                y = keypoints[coco_idx * 3 + 1]
                conf = keypoints[coco_idx * 3 + 2]
                
                # Lower threshold for head keypoints (0-4)
                threshold = 0.1 if coco_idx <= 4 else 0.3
                
                if conf > threshold:
                    coco_points[coco_idx] = (int(x), int(y))
        
        # If no head detected, estimate from shoulders
        if 0 not in coco_points and (5 in coco_points and 6 in coco_points):
            # Estimate head position from shoulders
            left_shoulder = coco_points[5]
            right_shoulder = coco_points[6] 
            
            # Calculate head position
            center_x = (left_shoulder[0] + right_shoulder[0]) // 2
            head_y = min(left_shoulder[1], right_shoulder[1]) - 60  # Above shoulders
            
            # Add estimated head points
            coco_points[0] = (center_x, head_y)  # nose
            coco_points[1] = (center_x - 15, head_y - 5)  # left eye
            coco_points[2] = (center_x + 15, head_y - 5)  # right eye
            
            print(f"📍 Estimated head position from shoulders")
        
        # Convert to OpenPose format
        openpose_points = {}
        for coco_idx, openpose_idx in coco_to_openpose.items():
            if coco_idx in coco_points:
                openpose_points[openpose_idx] = coco_points[coco_idx]
        
        # OpenPose connections (improved)
        openpose_connections = [
            # Head connections
            (0, 14), (14, 16), (0, 15), (15, 17),  # Face outline
            (0, 1),   # Nose to neck
            
            # Body connections  
            (1, 2), (1, 5), (1, 8),  # Neck to shoulders
            (2, 3), (3, 4),  # Right arm
            (5, 6), (6, 7),  # Left arm
            (1, 11), (11, 12), (12, 13),  # Left leg
            (1, 8), (8, 9), (9, 10),     # Right leg
            (8, 11), # Connect hips
        ]
        
        # Draw skeleton with better visibility
        # Draw connections first
        for start_idx, end_idx in openpose_connections:
            if start_idx in openpose_points and end_idx in openpose_points:
                start_point = openpose_points[start_idx]
                end_point = openpose_points[end_idx]
                cv2.line(pose_image, start_point, end_point, (255, 255, 255), 5)  # Thicker lines
        
        # Draw keypoints with emphasis on head
        for idx, point in openpose_points.items():
            if idx in [0, 14, 15, 16, 17]:  # Head keypoints
                cv2.circle(pose_image, point, 10, (255, 255, 255), -1)  # Larger head points
                cv2.circle(pose_image, point, 12, (0, 255, 0), 3)  # Green outline for head
            else:
                cv2.circle(pose_image, point, 8, (255, 255, 255), -1)
                cv2.circle(pose_image, point, 10, (0, 255, 0), 2)
        
        return pose_image

class BatchImageProcessor:
    """Process multiple images overnight with AI generation"""
    
    def __init__(self):
        print("🌙 Batch AI Image Processor")
        print("=" * 40)
        
        if not AI_AVAILABLE:
            print("❌ AI libraries not available!")
            return
        
        # Initialize components
        self.pose_extractor = StaticPoseExtractor()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Debug device detection
        print(f"🔍 Device detection:")
        print(f"   torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"   Selected device: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
        # Load AI models
        self._load_ai_models()
        
        # Processing stats
        self.total_processed = 0
        self.successful_generations = 0
        self.failed_generations = 0
        
        print("✅ Batch processor ready for overnight processing!")
        print(f"   Device: {self.device}")
        print("   Ready to process: Images → Poses → AI Humans")
    
    def _load_ai_models(self):
        """Load ControlNet and Stable Diffusion models"""
        try:
            print("📥 Loading AI models...")
            print(f"🎯 Target device: {self.device}")
            
            # Load ControlNet
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-openpose",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            print(f"✅ ControlNet loaded on {self.device}")
            
            # Load Stable Diffusion pipeline
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Optimize for speed
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
            self.pipe = self.pipe.to(self.device)
            print(f"✅ Pipeline moved to {self.device}")
            
            # Enable memory optimization for GPU (with fallback)
            if self.device == "cuda":
                try:
                    # Try memory optimizations with error handling
                    self.pipe.enable_model_cpu_offload()
                    print("✅ CPU offload enabled")
                except Exception as e:
                    print(f"⚠️ CPU offload failed: {e}")
                
                try:
                    # Skip xFormers if it's causing issues
                    self.pipe.enable_xformers_memory_efficient_attention()
                    print("✅ xFormers memory optimization enabled")
                except Exception as e:
                    print(f"⚠️ xFormers optimization failed (this is OK): {e}")
                    print("💡 Continuing without xFormers - will still be fast!")
            
            print("✅ AI models loaded successfully!")
            print(f"🚀 Ready for {self.device.upper()} accelerated generation!")
            
        except Exception as e:
            print(f"❌ Error loading AI models: {e}")
            print("💡 Trying simplified loading without optimizations...")
            
            # Fallback: Load without optimizations
            try:
                controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-openpose",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                
                self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    controlnet=controlnet,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                )
                
                self.pipe = self.pipe.to(self.device)
                print("✅ AI models loaded (simplified mode)")
                print(f"🚀 Ready for {self.device.upper()} generation!")
                
            except Exception as e2:
                print(f"❌ Fallback loading also failed: {e2}")
                self.pipe = None
    
    def process_image_folder(self, input_folder: str, output_folder: str, character_file: str, 
                           image_extensions: List[str] = None):
        """Process all images in a folder overnight"""
        
        # Prevent laptop from sleeping during processing
        prevent_sleep()
        
        try:
            if image_extensions is None:
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            
            # Setup folders
            input_path = Path(input_folder)
            output_path = Path(output_folder)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Load character
            character = self._load_character(character_file)
            if not character:
                print("❌ Could not load character file")
                return
            
            # Find all images
            image_files = []
            for ext in image_extensions:
                image_files.extend(input_path.glob(f"*{ext}"))
                image_files.extend(input_path.glob(f"*{ext.upper()}"))
            
            if not image_files:
                print(f"❌ No images found in {input_folder}")
                return
            
            print(f"\n🌙 OVERNIGHT BATCH PROCESSING")
            print("=" * 50)
            print(f"📁 Input folder: {input_folder}")
            print(f"📁 Output folder: {output_folder}")
            print(f"👤 Character: {character.name}")
            print(f"📸 Images found: {len(image_files)}")
            print(f"⏰ Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🔒 Sleep prevention: ACTIVE - laptop will not sleep")
            
            # Estimate time with realistic expectations
            if self.device == "cuda":
                estimated_minutes_per_image = 2  # GPU: 2 minutes per image
            else:
                estimated_minutes_per_image = 10  # CPU: 10 minutes per image
                
            estimated_total_minutes = len(image_files) * estimated_minutes_per_image
            estimated_hours = estimated_total_minutes // 60
            
            print(f"⏱️ Estimated completion time: {estimated_hours}h {estimated_total_minutes % 60}m")
            print(f"🔧 Performance mode: {'GPU Accelerated' if self.device == 'cuda' else 'CPU Only (Slower)'}")
            
            if self.device == "cpu":
                print(f"💡 For faster processing, install CUDA-compatible PyTorch")
                print(f"   Current speed: ~10 min/image | With GPU: ~2 min/image")
            
            input("Press Enter to start overnight processing...")
            
            # Process each image
            start_time = time.time()
            
            for i, image_file in enumerate(image_files, 1):
                print(f"\n🔄 Processing {i}/{len(image_files)}: {image_file.name}")
                print("-" * 40)
                
                try:
                    # Extract pose
                    pose_image = self.pose_extractor.extract_pose_from_image(str(image_file))
                    
                    if pose_image is not None:
                        # Generate AI human
                        ai_result = self._generate_ai_human(pose_image, character, image_file.name)
                        
                        if ai_result is not None:
                            # Save result
                            output_file = output_path / f"ai_{image_file.stem}.png"
                            cv2.imwrite(str(output_file), cv2.cvtColor(ai_result, cv2.COLOR_RGB2BGR))
                            
                            # Save pose for reference
                            pose_file = output_path / f"pose_{image_file.stem}.png"
                            cv2.imwrite(str(pose_file), pose_image)
                            
                            self.successful_generations += 1
                            print(f"✅ Generated: {output_file.name}")
                        else:
                            self.failed_generations += 1
                            print(f"❌ Failed to generate AI human")
                    else:
                        self.failed_generations += 1
                        print(f"❌ No pose detected")
                    
                    self.total_processed += 1
                    
                    # Progress update
                    elapsed_time = time.time() - start_time
                    avg_time_per_image = elapsed_time / i
                    remaining_images = len(image_files) - i
                    eta_seconds = remaining_images * avg_time_per_image
                    eta_time = datetime.fromtimestamp(time.time() + eta_seconds)
                    
                    print(f"📊 Progress: {i}/{len(image_files)} ({i/len(image_files)*100:.1f}%)")
                    print(f"⏰ ETA: {eta_time.strftime('%H:%M:%S')}")
                    print(f"✅ Success rate: {self.successful_generations}/{self.total_processed} ({self.successful_generations/self.total_processed*100:.1f}%)")
                    
                except Exception as e:
                    print(f"❌ Error processing {image_file.name}: {e}")
                    self.failed_generations += 1
                    self.total_processed += 1
            
            # Final report
            total_time = time.time() - start_time
            self._print_final_report(total_time, output_folder)
            
        finally:
            # Always re-enable sleep when done
            allow_sleep()
    
    def _load_character(self, character_file: str) -> Optional[CharacterProfile]:
        """Load character from JSON file"""
        try:
            with open(character_file, 'r') as f:
                character_data = json.load(f)
            
            # Handle different file formats
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
    
    def _generate_character_prompt(self, character: CharacterProfile) -> str:
        """Generate AI prompt from character profile"""
        prompt_parts = []
        
        # Base description
        prompt_parts.append("photorealistic portrait of a person")
        
        # Age estimation
        age_factor = character.facial.age_factor
        if age_factor < 0.3:
            prompt_parts.append("young adult")
        elif age_factor < 0.6:
            prompt_parts.append("middle-aged")
        else:
            prompt_parts.append("mature")
        
        # Body build
        if character.body.build > 0.7:
            prompt_parts.append("athletic muscular build")
        elif character.body.build < 0.3:
            prompt_parts.append("slim build")
        
        # Hair description
        hair_styles = {
            "short_professional": "short professional hair",
            "medium_wavy": "medium wavy hair",
            "long_straight": "long straight hair",
            "long_curly": "long curly hair",
            "ponytail": "hair in ponytail",
            "pixie_cut": "pixie cut hair",
            "bob_cut": "bob cut hair"
        }
        
        hair_desc = hair_styles.get(character.hair.style, "styled hair")
        
        # Hair color
        hair_color = character.hair.color
        if hair_color[0] > 0.8 and hair_color[1] > 0.8:  # Blonde
            hair_desc = f"blonde {hair_desc}"
        elif hair_color[0] < 0.3 and hair_color[1] < 0.3 and hair_color[2] < 0.3:  # Black
            hair_desc = f"black {hair_desc}"
        elif hair_color[0] > 0.5 and hair_color[1] < 0.4:  # Brown/Red
            hair_desc = f"brown {hair_desc}"
        
        prompt_parts.append(hair_desc)
        
        # Eye color
        eye_color = character.facial.eye_color
        if eye_color[2] > 0.7:  # Blue
            prompt_parts.append("bright blue eyes")
        elif eye_color[1] > 0.6:  # Green
            prompt_parts.append("green eyes")
        elif eye_color[0] > 0.5 and eye_color[1] > 0.3:  # Brown
            prompt_parts.append("brown eyes")
        
        # Clothing
        clothing_styles = {
            "athletic_top": "wearing athletic sports top",
            "dress_shirt": "wearing dress shirt",
            "casual_tshirt": "wearing casual t-shirt",
            "sweater": "wearing sweater"
        }
        
        clothing_desc = clothing_styles.get(character.clothing.shirt_type, "stylish clothing")
        prompt_parts.append(clothing_desc)
        
        # Quality modifiers
        prompt_parts.extend([
            "high quality", "detailed", "professional photography",
            "natural lighting", "sharp focus", "8k resolution"
        ])
        
        return ", ".join(prompt_parts)
    
    def _generate_negative_prompt(self) -> str:
        """Generate negative prompt"""
        return ("blurry, low quality, distorted, deformed, ugly, bad anatomy, "
                "extra limbs, missing limbs, floating limbs, disconnected limbs, "
                "malformed hands, extra fingers, mutated hands, poorly drawn hands, "
                "poorly drawn face, mutation, mutated, bad proportions, "
                "cropped, lowres, text, jpeg artifacts, signature, watermark")
    
    def _generate_ai_human(self, pose_image: np.ndarray, character: CharacterProfile, 
                          image_name: str) -> Optional[np.ndarray]:
        """Generate AI human from pose and character"""
        if not self.pipe:
            print("❌ AI models not loaded!")
            return None
        
        try:
            # Convert pose image to PIL
            pose_pil = Image.fromarray(pose_image)
            
            # Generate prompts with more specific character details
            prompt = self._generate_detailed_character_prompt(character)
            negative_prompt = self._generate_enhanced_negative_prompt()
            
            print(f"🎨 Generating AI human for: {image_name}")
            print(f"📝 Prompt: {prompt[:80]}...")
            print(f"🔧 Device: {self.device}")
            
            # Performance optimizations
            step_start_time = time.time()
            
            # Improved generation settings
            num_steps = 20 if self.device == "cuda" else 12  # More steps for better quality
            
            print(f"⚡ Using {num_steps} steps for quality optimization")
            
            # Generate image with improved settings
            with torch.autocast(self.device):
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=pose_pil,
                    num_inference_steps=num_steps,
                    guidance_scale=7.5,  # Standard guidance for better results
                    controlnet_conditioning_scale=1.0,  # Full pose control
                    generator=torch.Generator(device=self.device).manual_seed(42),
                    # Improved settings
                    eta=0.0,  # Deterministic
                )
            
            generation_time = time.time() - step_start_time
            
            # Convert back to numpy
            generated_image = np.array(result.images[0])
            
            print(f"✅ AI generation complete for: {image_name}")
            print(f"⏱️ Generation time: {generation_time:.1f} seconds")
            
            # Performance info
            if generation_time < 60:
                print(f"🚀 Great performance: {generation_time:.1f}s")
            elif generation_time < 300:
                print(f"⚡ Good performance: {generation_time/60:.1f}m")
            else:
                print(f"⚠️ Slow generation: {generation_time/60:.1f}m")
            
            return generated_image
            
        except Exception as e:
            print(f"❌ AI generation error for {image_name}: {e}")
            return None
    
    def _generate_detailed_character_prompt(self, character: CharacterProfile) -> str:
        """Generate detailed AI prompt for better results"""
        prompt_parts = []
        
        # More specific base description
        prompt_parts.append("full body portrait of a beautiful person")
        
        # Better age description
        age_factor = character.facial.age_factor
        if age_factor < 0.3:
            prompt_parts.append("young woman with youthful features")
        elif age_factor < 0.6:
            prompt_parts.append("adult woman with mature features")
        else:
            prompt_parts.append("elegant mature woman")
        
        # Detailed hair description
        hair_color = character.hair.color
        if hair_color[0] > 0.8 and hair_color[1] > 0.7:  # Blonde
            hair_desc = "beautiful blonde wavy hair"
        elif hair_color[0] < 0.3 and hair_color[1] < 0.3:  # Black
            hair_desc = "long black hair"
        else:
            hair_desc = "brown hair"
        
        prompt_parts.append(hair_desc)
        
        # Better eye description
        eye_color = character.facial.eye_color
        if eye_color[2] > 0.7:  # Blue
            prompt_parts.append("piercing blue eyes")
        elif eye_color[1] > 0.6:  # Green
            prompt_parts.append("bright green eyes") 
        else:
            prompt_parts.append("beautiful brown eyes")
        
        # Detailed clothing
        clothing_styles = {
            "athletic_top": "wearing blue athletic sports top",
            "dress_shirt": "wearing white dress shirt",
            "casual_tshirt": "wearing casual t-shirt",
            "sweater": "wearing cozy sweater"
        }
        
        clothing_desc = clothing_styles.get(character.clothing.shirt_type, "wearing stylish blue top")
        prompt_parts.append(clothing_desc)
        
        # Enhanced quality modifiers
        prompt_parts.extend([
            "professional photography", "studio lighting", "high resolution",
            "detailed facial features", "natural skin texture", "perfect anatomy",
            "photorealistic", "8k quality", "sharp focus"
        ])
        
        return ", ".join(prompt_parts)
    
    def _generate_enhanced_negative_prompt(self) -> str:
        """Generate enhanced negative prompt for better quality"""
        return ("blurry, low quality, distorted, deformed, ugly, bad anatomy, "
                "extra limbs, missing limbs, floating limbs, disconnected limbs, "
                "malformed hands, extra fingers, mutated hands, poorly drawn hands, "
                "poorly drawn face, mutation, mutated, bad proportions, cropped, "
                "lowres, text, jpeg artifacts, signature, watermark, username, "
                "duplicate, extra arms, extra legs, extra hands, poorly drawn eyes, "
                "cross-eyed, out of frame, disfigured, gross proportions, "
                "long neck, duplicate heads, multiple people")
    
    def _print_final_report(self, total_time: float, output_folder: str):
        """Print final processing report"""
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        success_rate = (self.successful_generations / self.total_processed * 100) if self.total_processed > 0 else 0
        
        print(f"\n🎉 OVERNIGHT PROCESSING COMPLETE!")
        print("=" * 50)
        print(f"⏰ Total time: {hours}h {minutes}m {seconds}s")
        print(f"📸 Images processed: {self.total_processed}")
        print(f"✅ Successful generations: {self.successful_generations}")
        print(f"❌ Failed generations: {self.failed_generations}")
        print(f"📊 Success rate: {success_rate:.1f}%")
        print(f"📁 Results saved to: {output_folder}")
        print(f"🔓 Sleep prevention deactivated - laptop can sleep again")
        print(f"🔍 Check your output folder for:")
        print(f"   • ai_*.png (Generated AI humans)")
        print(f"   • pose_*.png (Extracted poses)")
        
        if self.successful_generations > 0:
            avg_time = total_time / self.total_processed
            print(f"⚡ Average time per image: {avg_time:.1f} seconds")
            print(f"\n🎨 You now have {self.successful_generations} AI-generated humans!")

def main():
    """Main batch processing function"""
    print("Static Image Batch Processor for AI Human Generation")
    print("=" * 60)
    
    if not AI_AVAILABLE:
        print("❌ AI libraries required!")
        print("Install with:")
        print("  pip install diffusers transformers accelerate")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return
    
    processor = BatchImageProcessor()
    
    if not processor.pipe:
        print("❌ AI models failed to load. Run robust_ai_downloader.py first.")
        return
    
    print("\n🌙 Overnight Batch Processing Setup")
    print("=" * 40)
    
    # Get input parameters
    input_folder = input("📁 Enter input folder path (with your photos): ").strip()
    if not Path(input_folder).exists():
        print("❌ Input folder does not exist!")
        return
    
    output_folder = input("📁 Enter output folder path (where to save AI humans): ").strip()
    if not output_folder:
        output_folder = "ai_generated_humans"
    
    character_file = input("👤 Enter character JSON file: ").strip()
    if not Path(character_file).exists():
        print("❌ Character file does not exist!")
        return
    
    print(f"\n📋 Batch Processing Configuration:")
    print(f"   Input: {input_folder}")
    print(f"   Output: {output_folder}")
    print(f"   Character: {character_file}")
    print(f"   Supported formats: .jpg, .jpeg, .png, .bmp, .tiff")
    
    # Start processing
    processor.process_image_folder(input_folder, output_folder, character_file)

if __name__ == "__main__":
    main()