"""
Enhanced Single Human Batch Processor
Generates ONE complete human per pose instead of multiple fragments
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
        
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_DISPLAY_REQUIRED = 0x00000002
        
        def prevent_sleep():
            ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
            )
            print("🔒 Sleep prevention activated - laptop will stay awake")
        
        def allow_sleep():
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
            print("🔒 Linux detected - please disable sleep manually")
        
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
    from PIL import Image, ImageDraw, ImageFilter
    import torch
    AI_AVAILABLE = True
    print("✅ AI libraries loaded successfully!")
except ImportError as e:
    AI_AVAILABLE = False
    print(f"⚠️ AI libraries missing: {e}")

# Import our existing systems
from complete_human_tracker import CompleteHumanTracker
from realistic_character_creator import CharacterProfile, FacialFeatures, BodyFeatures, HairFeatures, ClothingFeatures

class EnhancedPoseExtractor:
    """Enhanced pose extraction for single human generation"""
    
    def __init__(self):
        print("📷 Initializing Enhanced Pose Extractor...")
        self.tracker = CompleteHumanTracker()
        print("✅ Enhanced pose extractor ready!")
    
    def extract_pose_from_image(self, image_path: str) -> Optional[np.ndarray]:
        """Extract optimized OpenPose skeleton for single human generation"""
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
            
            # Convert to enhanced OpenPose format
            h, w = image.shape[:2]
            pose_image = self._convert_to_enhanced_openpose(tracking_data, (h, w))
            
            print(f"✅ Enhanced pose extracted from: {Path(image_path).name}")
            return pose_image
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            return None
    
    def _convert_to_enhanced_openpose(self, tracking_data, frame_shape: Tuple[int, int]) -> np.ndarray:
        """Convert to enhanced OpenPose format for single human generation"""
        h, w = frame_shape
        
        # Create blank pose image with better quality
        pose_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        if not tracking_data.body_keypoints_2d:
            return pose_image
        
        keypoints = tracking_data.body_keypoints_2d
        
        # Enhanced COCO to OpenPose mapping for better results
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
        
        # Extract COCO keypoints with adaptive confidence threshold
        coco_points = {}
        for coco_idx in range(17):
            if coco_idx * 3 + 2 < len(keypoints):
                x = keypoints[coco_idx * 3]
                y = keypoints[coco_idx * 3 + 1]
                conf = keypoints[coco_idx * 3 + 2]
                
                # Very low threshold for head keypoints for better face generation
                if coco_idx <= 4:  # Head keypoints
                    threshold = 0.05
                else:
                    threshold = 0.2
                
                if conf > threshold:
                    coco_points[coco_idx] = (int(x), int(y))
        
        # CRITICAL: Ensure complete skeleton connectivity
        coco_points = self._ensure_skeleton_completeness(coco_points, w, h)
        
        # Convert to OpenPose format
        openpose_points = {}
        for coco_idx, openpose_idx in coco_to_openpose.items():
            if coco_idx in coco_points:
                openpose_points[openpose_idx] = coco_points[coco_idx]
        
        # Enhanced OpenPose connections for single human
        openpose_connections = [
            # CRITICAL: Neck-to-body connection (prevents multiple people)
            (1, 8), (1, 11),  # Neck to both hips - ESSENTIAL for single human
            
            # Head connections (enhanced for better face generation)
            (0, 1),   # Nose to neck - CRITICAL
            (0, 14), (14, 16), (0, 15), (15, 17),  # Face outline
            
            # Body structure (must be connected)
            (1, 2), (1, 5),    # Neck to shoulders
            (2, 3), (3, 4),    # Right arm
            (5, 6), (6, 7),    # Left arm
            (8, 9), (9, 10),   # Right leg
            (11, 12), (12, 13), # Left leg
            (8, 11),           # Hip connection
            
            # Shoulder-to-hip connections (creates torso unity)
            (2, 8), (5, 11),   # Shoulders to opposite hips
        ]
        
        # Draw enhanced skeleton for single human generation
        
        # Step 1: Draw thick connections first (creates unified body)
        for start_idx, end_idx in openpose_connections:
            if start_idx in openpose_points and end_idx in openpose_points:
                start_point = openpose_points[start_idx]
                end_point = openpose_points[end_idx]
                
                # Thicker lines for core body structure
                if (start_idx, end_idx) in [(1, 8), (1, 11), (0, 1)]:
                    cv2.line(pose_image, start_point, end_point, (255, 255, 255), 8)
                else:
                    cv2.line(pose_image, start_point, end_point, (255, 255, 255), 6)
        
        # Step 2: Draw keypoints with emphasis on unity
        for idx, point in openpose_points.items():
            if idx == 0:  # Nose/head - most important for face generation
                cv2.circle(pose_image, point, 15, (255, 255, 255), -1)
                cv2.circle(pose_image, point, 18, (200, 200, 200), 3)
            elif idx in [1, 8, 11]:  # Core structural points
                cv2.circle(pose_image, point, 12, (255, 255, 255), -1)
                cv2.circle(pose_image, point, 15, (200, 200, 200), 3)
            elif idx in [14, 15, 16, 17]:  # Face points
                cv2.circle(pose_image, point, 10, (255, 255, 255), -1)
                cv2.circle(pose_image, point, 12, (200, 200, 200), 2)
            else:  # Other body points
                cv2.circle(pose_image, point, 10, (255, 255, 255), -1)
                cv2.circle(pose_image, point, 12, (200, 200, 200), 2)
        
        # Step 3: Add subtle body outline to reinforce single human
        self._add_body_outline(pose_image, openpose_points)
        
        return pose_image
    
    def _ensure_skeleton_completeness(self, coco_points: Dict, w: int, h: int) -> Dict:
        """Ensure skeleton has all critical points for single human generation"""
        
        # If missing head but have shoulders, estimate head position
        if 0 not in coco_points and (5 in coco_points and 6 in coco_points):
            left_shoulder = coco_points[5]
            right_shoulder = coco_points[6] 
            
            # Calculate head position above shoulders
            center_x = (left_shoulder[0] + right_shoulder[0]) // 2
            head_y = min(left_shoulder[1], right_shoulder[1]) - 80  # Well above shoulders
            head_y = max(head_y, 20)  # Don't go off screen
            
            # Add complete head set
            coco_points[0] = (center_x, head_y)  # nose
            coco_points[1] = (center_x - 20, head_y - 5)  # left eye
            coco_points[2] = (center_x + 20, head_y - 5)  # right eye
            coco_points[3] = (center_x - 30, head_y)      # left ear
            coco_points[4] = (center_x + 30, head_y)      # right ear
            
            print(f"📍 Estimated complete head from shoulders")
        
        # If missing hips but have knees, estimate hips
        if 11 not in coco_points and 13 in coco_points:  # Missing left hip
            left_knee = coco_points[13]
            coco_points[11] = (left_knee[0], left_knee[1] - 100)
            
        if 12 not in coco_points and 14 in coco_points:  # Missing right hip
            right_knee = coco_points[14]
            coco_points[12] = (right_knee[0], right_knee[1] - 100)
        
        # Ensure hip connectivity for single human
        if 11 in coco_points and 12 not in coco_points:
            left_hip = coco_points[11]
            coco_points[12] = (left_hip[0] + 100, left_hip[1])  # Estimate right hip
        elif 12 in coco_points and 11 not in coco_points:
            right_hip = coco_points[12]
            coco_points[11] = (right_hip[0] - 100, right_hip[1])  # Estimate left hip
        
        return coco_points
    
    def _add_body_outline(self, pose_image: np.ndarray, openpose_points: Dict):
        """Add subtle body outline to reinforce single human perception"""
        # Get key points for body outline
        key_points = []
        outline_indices = [0, 14, 16, 2, 4, 8, 10, 11, 13, 5, 7, 15, 17]  # Head to body outline
        
        for idx in outline_indices:
            if idx in openpose_points:
                key_points.append(openpose_points[idx])
        
        if len(key_points) > 3:
            # Create a very subtle body silhouette
            points_array = np.array(key_points, dtype=np.int32)
            
            # Draw a very faint outline
            cv2.fillPoly(pose_image, [points_array], (30, 30, 30))  # Very dark fill
            cv2.polylines(pose_image, [points_array], True, (100, 100, 100), 2)  # Subtle outline

class EnhancedBatchProcessor:
    """Enhanced batch processor for single human generation"""
    
    def __init__(self):
        print("🌙 Enhanced Single Human Batch Processor")
        print("=" * 50)
        
        if not AI_AVAILABLE:
            print("❌ AI libraries not available!")
            return
        
        # Initialize components
        self.pose_extractor = EnhancedPoseExtractor()
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
        
        print("✅ Enhanced batch processor ready for single human generation!")
        print(f"   Device: {self.device}")
        print("   Ready to process: Images → Enhanced Poses → Single AI Humans")
    
    def _load_ai_models(self):
        """Load enhanced AI models for single human generation"""
        try:
            print("📥 Loading enhanced AI models...")
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
            
            # Enhanced scheduler for better single human generation
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
            self.pipe = self.pipe.to(self.device)
            print(f"✅ Pipeline moved to {self.device}")
            
            # Enable optimizations with fallback
            if self.device == "cuda":
                try:
                    self.pipe.enable_model_cpu_offload()
                    print("✅ CPU offload enabled")
                except Exception as e:
                    print(f"⚠️ CPU offload failed: {e}")
                
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    print("✅ xFormers memory optimization enabled")
                except Exception as e:
                    print(f"⚠️ xFormers optimization failed (this is OK): {e}")
                    print("💡 Continuing without xFormers - will still be fast!")
            
            print("✅ Enhanced AI models loaded successfully!")
            print(f"🚀 Ready for {self.device.upper()} accelerated single human generation!")
            
        except Exception as e:
            print(f"❌ Error loading AI models: {e}")
            print("💡 Trying simplified loading...")
            
            # Fallback loading
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
                
            except Exception as e2:
                print(f"❌ Fallback loading also failed: {e2}")
                self.pipe = None
    
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
    
    def _generate_single_human_prompt(self, character: CharacterProfile) -> str:
        """Generate enhanced prompt for single complete human"""
        prompt_parts = []
        
        # CRITICAL: Single person specification
        prompt_parts.append("single beautiful woman")
        prompt_parts.append("full body portrait")
        prompt_parts.append("complete person")
        prompt_parts.append("one person only")
        
        # Character details
        age_factor = character.facial.age_factor
        if age_factor < 0.3:
            prompt_parts.append("young woman with youthful features")
        elif age_factor < 0.6:
            prompt_parts.append("adult woman with mature features")
        else:
            prompt_parts.append("elegant mature woman")
        
        # Enhanced hair description
        hair_color = character.hair.color
        if hair_color[0] > 0.8 and hair_color[1] > 0.7:
            hair_desc = "beautiful blonde wavy hair"
        elif hair_color[0] < 0.3 and hair_color[1] < 0.3:
            hair_desc = "long black hair"
        else:
            hair_desc = "brown hair"
        
        prompt_parts.append(hair_desc)
        
        # Eye description
        eye_color = character.facial.eye_color
        if eye_color[2] > 0.7:
            prompt_parts.append("piercing blue eyes")
        elif eye_color[1] > 0.6:
            prompt_parts.append("bright green eyes") 
        else:
            prompt_parts.append("beautiful brown eyes")
        
        # Clothing
        clothing_styles = {
            "athletic_top": "wearing blue athletic sports top",
            "dress_shirt": "wearing white dress shirt",
            "casual_tshirt": "wearing casual t-shirt",
            "sweater": "wearing cozy sweater"
        }
        
        clothing_desc = clothing_styles.get(character.clothing.shirt_type, "wearing stylish blue athletic top")
        prompt_parts.append(clothing_desc)
        
        # Enhanced quality modifiers for single human
        prompt_parts.extend([
            "professional photography", "studio lighting", "high resolution",
            "detailed facial features", "natural skin texture", "perfect anatomy",
            "photorealistic", "8k quality", "sharp focus", "single subject",
            "complete body", "unified pose", "coherent human figure"
        ])
        
        return ", ".join(prompt_parts)
    
    def _generate_enhanced_negative_prompt(self) -> str:
        """Generate enhanced negative prompt for single human"""
        return ("multiple people, crowd, group, extra person, duplicate people, "
                "multiple faces, multiple heads, extra heads, floating heads, "
                "disconnected limbs, floating limbs, extra limbs, missing limbs, "
                "malformed hands, extra fingers, mutated hands, poorly drawn hands, "
                "poorly drawn face, mutation, mutated, bad proportions, cropped, "
                "lowres, text, jpeg artifacts, signature, watermark, username, "
                "duplicate, extra arms, extra legs, extra hands, poorly drawn eyes, "
                "cross-eyed, out of frame, disfigured, gross proportions, "
                "long neck, duplicate heads, fragmented body, incomplete person, "
                "blurry, low quality, distorted, deformed, ugly, bad anatomy")
    
    def _generate_single_human(self, pose_image: np.ndarray, character: CharacterProfile, 
                              image_name: str) -> Optional[np.ndarray]:
        """Generate single complete AI human from pose"""
        if not self.pipe:
            print("❌ AI models not loaded!")
            return None
        
        try:
            # Convert pose image to PIL
            pose_pil = Image.fromarray(pose_image)
            
            # Apply slight blur to pose for better single human generation
            pose_pil = pose_pil.filter(ImageFilter.GaussianBlur(radius=0.5))
            
            # Generate enhanced prompts
            prompt = self._generate_single_human_prompt(character)
            negative_prompt = self._generate_enhanced_negative_prompt()
            
            print(f"🎨 Generating single AI human for: {image_name}")
            print(f"📝 Prompt: {prompt[:80]}...")
            print(f"🔧 Device: {self.device}")
            
            step_start_time = time.time()
            
            # Enhanced generation settings for single human
            num_steps = 25 if self.device == "cuda" else 15
            
            print(f"⚡ Using {num_steps} steps for single human generation")
            
            # Generate with enhanced settings for single complete human
            with torch.autocast(self.device):
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=pose_pil,
                    num_inference_steps=num_steps,
                    guidance_scale=8.0,  # Higher guidance for better control
                    controlnet_conditioning_scale=1.2,  # Stronger pose control
                    generator=torch.Generator(device=self.device).manual_seed(42),
                    eta=0.0,
                    width=512,   # Standard size for better results
                    height=512,
                )
            
            generation_time = time.time() - step_start_time
            
            # Convert to numpy
            generated_image = np.array(result.images[0])
            
            print(f"✅ Single human generation complete for: {image_name}")
            print(f"⏱️ Generation time: {generation_time:.1f} seconds")
            
            return generated_image
            
        except Exception as e:
            print(f"❌ Single human generation error for {image_name}: {e}")
            return None
    
    def process_image_folder(self, input_folder: str, output_folder: str, character_file: str, 
                           image_extensions: List[str] = None):
        """Process all images for single human generation"""
        
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
            
            print(f"\n🌙 ENHANCED SINGLE HUMAN PROCESSING")
            print("=" * 55)
            print(f"📁 Input folder: {input_folder}")
            print(f"📁 Output folder: {output_folder}")
            print(f"👤 Character: {character.name}")
            print(f"📸 Images found: {len(image_files)}")
            print(f"🎯 Goal: ONE complete human per image")
            print(f"⏰ Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Process each image
            start_time = time.time()
            
            for i, image_file in enumerate(image_files, 1):
                print(f"\n🔄 Processing {i}/{len(image_files)}: {image_file.name}")
                print("-" * 50)
                
                try:
                    # Extract enhanced pose
                    pose_image = self.pose_extractor.extract_pose_from_image(str(image_file))
                    
                    if pose_image is not None:
                        # Generate single AI human
                        ai_result = self._generate_single_human(pose_image, character, image_file.name)
                        
                        if ai_result is not None:
                            # Save results
                            output_file = output_path / f"single_human_{image_file.stem}.png"
                            cv2.imwrite(str(output_file), cv2.cvtColor(ai_result, cv2.COLOR_RGB2BGR))
                            
                            # Save enhanced pose for reference
                            pose_file = output_path / f"enhanced_pose_{image_file.stem}.png"
                            cv2.imwrite(str(pose_file), pose_image)
                            
                            self.successful_generations += 1
                            print(f"✅ Generated single human: {output_file.name}")
                        else:
                            self.failed_generations += 1
                            print(f"❌ Failed to generate single human")
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
            allow_sleep()
    
    def _print_final_report(self, total_time: float, output_folder: str):
        """Print final processing report"""
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        success_rate = (self.successful_generations / self.total_processed * 100) if self.total_processed > 0 else 0
        
        print(f"\n🎉 SINGLE HUMAN PROCESSING COMPLETE!")
        print("=" * 55)
        print(f"⏰ Total time: {hours}h {minutes}m {seconds}s")
        print(f"📸 Images processed: {self.total_processed}")
        print(f"✅ Single humans generated: {self.successful_generations}")
        print(f"❌ Failed generations: {self.failed_generations}")
        print(f"📊 Success rate: {success_rate:.1f}%")
        print(f"📁 Results saved to: {output_folder}")
        print(f"🎯 Check output for:")
        print(f"   • single_human_*.png (Complete AI humans)")
        print(f"   • enhanced_pose_*.png (Enhanced pose skeletons)")
        
        if self.successful_generations > 0:
            avg_time = total_time / self.total_processed
            print(f"⚡ Average time per image: {avg_time:.1f} seconds")
            print(f"\n🎭 You now have {self.successful_generations} complete AI humans!")

def main():
    """Main enhanced batch processing function"""
    print("Enhanced Single Human Batch Processor")
    print("=" * 45)
    
    if not AI_AVAILABLE:
        print("❌ AI libraries required!")
        return
    
    processor = EnhancedBatchProcessor()
    
    if not processor.pipe:
        print("❌ AI models failed to load.")
        return
    
    print("\n🎯 Enhanced Single Human Processing Setup")
    print("=" * 45)
    
    # Get input parameters
    input_folder = input("📁 Enter input folder path (with your photos): ").strip()
    if not Path(input_folder).exists():
        print("❌ Input folder does not exist!")
        return
    
    output_folder = input("📁 Enter output folder path (where to save single humans): ").strip()
    if not output_folder:
        output_folder = "single_ai_humans"
    
    character_file = input("👤 Enter character JSON file: ").strip()
    if not Path(character_file).exists():
        print("❌ Character file does not exist!")
        return
    
    print(f"\n📋 Enhanced Processing Configuration:")
    print(f"   Input: {input_folder}")
    print(f"   Output: {output_folder}")
    print(f"   Character: {character_file}")
    print(f"   Goal: Generate ONE complete human per pose")
    
    # Start processing
    processor.process_image_folder(input_folder, output_folder, character_file)

if __name__ == "__main__":
    main()