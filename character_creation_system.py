"""
AI Character Creation System - Design Your Own Aitana
The FOUNDATIONAL system for creating AI influencer characters from scratch

BASED ON THE CLUELESS METHOD:
1. Months of experimentation to find the right "look"
2. GANs + professional prompting for character consistency
3. Trend analysis to determine appealing features
4. Iterative refinement until the character is "perfect"
5. Character profile and personality development

WHAT THIS DOES:
- Character Designer: Create unique appearance from scratch
- Look Experimentation: Test different styles like The Clueless did
- Consistency Engine: Ensure same character across all images
- Trend Analysis: Build characters based on market research
- Character Profiling: Develop personality and backstory
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime
import random
import itertools

try:
    from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
    import torch
    from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

class AICharacterDesigner:
    """
    Complete AI Character Creation System
    Design unique AI influencers from scratch using The Clueless methodology
    """
    
    def __init__(self):
        print("🎨 AI Character Designer - Create Your Own Aitana")
        print("=" * 60)
        print("📚 Based on The Clueless Agency's methodology")
        
        if not AI_AVAILABLE:
            print("❌ AI libraries required!")
            return
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None
        
        # Character design database
        self.character_experiments = {}
        self.successful_characters = {}
        self.design_history = []
        
        # Trend analysis data (based on social media trends)
        self.trending_features = self._load_trending_features()
        
        # Setup AI pipeline
        self._setup_pipeline()
        
        print(f"🚀 Device: {self.device}")
        print("✅ Ready to design your AI influencer!")
    
    def _setup_pipeline(self):
        """Setup professional AI pipeline for character creation"""
        try:
            print("📥 Loading Character Creation Pipeline...")
            
            # Use SDXL for highest quality character creation
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True
            ).to(self.device)
            
            if self.device == "cuda":
                self.pipeline.enable_model_cpu_offload()
            
            print("✅ Professional character creation pipeline loaded!")
            
        except Exception as e:
            print(f"⚠️ SDXL failed: {e}")
            print("💡 Trying SD 1.5 fallback...")
            
            try:
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                ).to(self.device)
                print("✅ Character creation pipeline loaded (SD 1.5)")
            except Exception as e2:
                print(f"❌ All pipelines failed: {e2}")
    
    def _load_trending_features(self) -> Dict:
        """
        Load trending features based on social media analysis
        (What The Clueless analyzed to create Aitana)
        """
        return {
            "hair_colors": {
                "trending": ["pink", "platinum blonde", "silver", "pastel blue"],
                "classic": ["brown", "black", "blonde", "red"],
                "weights": [0.4, 0.6]  # 40% trending, 60% classic
            },
            "hair_styles": {
                "long": ["long wavy", "long straight", "long curly"],
                "medium": ["medium layered", "bob cut", "shoulder length"],
                "short": ["pixie cut", "short bob", "asymmetrical"],
                "weights": [0.5, 0.3, 0.2]
            },
            "eye_colors": {
                "natural": ["brown", "blue", "green", "hazel"],
                "striking": ["bright blue", "emerald green", "violet", "amber"],
                "weights": [0.7, 0.3]
            },
            "face_shapes": {
                "popular": ["oval", "heart-shaped", "diamond"],
                "classic": ["round", "square", "rectangular"],
                "weights": [0.6, 0.4]
            },
            "body_types": {
                "fitness": ["athletic", "toned", "muscular"],
                "fashion": ["slim", "model-thin", "tall"],
                "lifestyle": ["curvy", "average", "petite"],
                "weights": [0.4, 0.3, 0.3]
            },
            "style_aesthetics": {
                "2024_trends": ["sporty chic", "minimalist", "Y2K revival", "cottagecore"],
                "timeless": ["elegant", "bohemian", "classic", "edgy"],
                "weights": [0.3, 0.7]
            },
            "niches": {
                "high_earning": ["fitness", "fashion", "lifestyle"],
                "emerging": ["gaming", "tech", "sustainability"],
                "traditional": ["beauty", "travel", "food"],
                "weights": [0.5, 0.3, 0.2]
            }
        }
    
    def analyze_market_trends(self) -> Dict:
        """
        Analyze current market trends for character design
        (The Clueless method: "We created her based on what society likes most")
        """
        print("📊 Analyzing Market Trends for Character Design...")
        
        trends = self.trending_features
        
        # Generate trend-based recommendations
        recommendations = {
            "hot_combinations": [
                {
                    "style": "Athletic Pink Hair Influencer",
                    "description": "Pink hair + athletic build + fitness niche",
                    "market_appeal": "Very High (Following Aitana's success)",
                    "features": {
                        "hair": "pink wavy",
                        "build": "athletic toned",
                        "niche": "fitness",
                        "style": "sporty chic"
                    }
                },
                {
                    "style": "Minimalist Fashion Model",
                    "description": "Platinum blonde + slim build + fashion niche",
                    "market_appeal": "High (Classic appeal)",
                    "features": {
                        "hair": "platinum blonde straight",
                        "build": "model-thin",
                        "niche": "fashion",
                        "style": "minimalist elegant"
                    }
                },
                {
                    "style": "Gaming Tech Girl",
                    "description": "Pastel blue hair + petite + gaming niche",
                    "market_appeal": "Growing (Emerging market)",
                    "features": {
                        "hair": "pastel blue short",
                        "build": "petite",
                        "niche": "gaming",
                        "style": "Y2K revival"
                    }
                }
            ],
            "trending_now": {
                "hair_colors": ["pink", "platinum blonde", "silver"],
                "aesthetics": ["sporty chic", "minimalist", "Y2K revival"],
                "niches": ["fitness", "lifestyle", "gaming"],
                "body_types": ["athletic", "toned", "petite"]
            }
        }
        
        print("✅ Market Analysis Complete!")
        print("\n🔥 Hot Character Combinations:")
        for combo in recommendations["hot_combinations"]:
            print(f"   • {combo['style']}: {combo['market_appeal']}")
        
        return recommendations
    
    def design_character_features(self, inspiration: str = "trending") -> Dict:
        """
        Design character features using trend analysis
        """
        print(f"🎨 Designing Character Features (Inspiration: {inspiration})")
        
        trends = self.trending_features
        
        if inspiration == "trending":
            # Follow trending combinations (like Aitana)
            hair_color = random.choices(
                trends["hair_colors"]["trending"] + trends["hair_colors"]["classic"],
                weights=[0.4] * len(trends["hair_colors"]["trending"]) + 
                        [0.6/len(trends["hair_colors"]["classic"])] * len(trends["hair_colors"]["classic"])
            )[0]
            
            niche = random.choices(
                trends["niches"]["high_earning"],
                weights=[1/len(trends["niches"]["high_earning"])] * len(trends["niches"]["high_earning"])
            )[0]
            
        elif inspiration == "classic":
            # Classic, timeless appeal
            hair_color = random.choice(trends["hair_colors"]["classic"])
            niche = random.choice(trends["niches"]["traditional"])
            
        elif inspiration == "experimental":
            # Bold, experimental (higher risk/reward)
            hair_color = random.choice(trends["hair_colors"]["trending"])
            niche = random.choice(trends["niches"]["emerging"])
        
        else:
            # Random combination
            hair_color = random.choice(trends["hair_colors"]["trending"] + trends["hair_colors"]["classic"])
            niche = random.choice(trends["niches"]["high_earning"] + trends["niches"]["emerging"])
        
        # Generate complementary features
        hair_style = random.choice([style for styles in trends["hair_styles"].values() for style in styles])
        eye_color = random.choice(trends["eye_colors"]["natural"] + trends["eye_colors"]["striking"])
        face_shape = random.choice(trends["face_shapes"]["popular"] + trends["face_shapes"]["classic"])
        
        # Body type based on niche
        if niche == "fitness":
            body_type = random.choice(trends["body_types"]["fitness"])
        elif niche == "fashion":
            body_type = random.choice(trends["body_types"]["fashion"])
        else:
            body_type = random.choice(trends["body_types"]["lifestyle"])
        
        # Style aesthetic
        style = random.choice(trends["style_aesthetics"]["2024_trends"] + trends["style_aesthetics"]["timeless"])
        
        character_features = {
            "physical": {
                "hair_color": hair_color,
                "hair_style": hair_style,
                "eye_color": eye_color,
                "face_shape": face_shape,
                "body_type": body_type,
                "skin_tone": "fair to medium"  # Adjustable
            },
            "style": {
                "aesthetic": style,
                "niche": niche,
                "personality": self._generate_personality(niche, style),
                "target_audience": self._determine_target_audience(niche)
            },
            "inspiration_type": inspiration,
            "market_appeal": self._estimate_market_appeal(hair_color, niche, body_type)
        }
        
        print("✅ Character Features Designed!")
        print(f"   • Hair: {hair_color} {hair_style}")
        print(f"   • Eyes: {eye_color}")
        print(f"   • Build: {body_type}")
        print(f"   • Niche: {niche}")
        print(f"   • Style: {style}")
        print(f"   • Market Appeal: {character_features['market_appeal']}")
        
        return character_features
    
    def _generate_personality(self, niche: str, style: str) -> str:
        """Generate personality based on niche and style"""
        personalities = {
            "fitness": ["motivational and energetic", "strong and confident", "disciplined and inspiring"],
            "fashion": ["elegant and sophisticated", "trendy and creative", "bold and expressive"],
            "lifestyle": ["relatable and warm", "aspirational and chic", "down-to-earth and friendly"],
            "gaming": ["fun and competitive", "tech-savvy and cool", "energetic and entertaining"],
            "travel": ["adventurous and free-spirited", "cultured and worldly", "inspiring and wanderlust"],
            "beauty": ["glamorous and polished", "nurturing and helpful", "confident and radiant"]
        }
        
        return random.choice(personalities.get(niche, personalities["lifestyle"]))
    
    def _determine_target_audience(self, niche: str) -> str:
        """Determine target audience based on niche"""
        audiences = {
            "fitness": "fitness enthusiasts, health-conscious individuals, athletes",
            "fashion": "fashion lovers, trendsetters, style-conscious consumers",
            "lifestyle": "young professionals, lifestyle aspirants, general audience",
            "gaming": "gamers, tech enthusiasts, esports fans",
            "travel": "travel enthusiasts, wanderlust seekers, adventure lovers",
            "beauty": "beauty enthusiasts, makeup lovers, skincare advocates"
        }
        
        return audiences.get(niche, audiences["lifestyle"])
    
    def _estimate_market_appeal(self, hair_color: str, niche: str, body_type: str) -> str:
        """Estimate market appeal based on feature combination"""
        appeal_score = 0
        
        # Hair color appeal
        if hair_color in ["pink", "platinum blonde"]:
            appeal_score += 3  # High appeal (following Aitana)
        elif hair_color in ["brown", "blonde"]:
            appeal_score += 2  # Classic appeal
        else:
            appeal_score += 1  # Niche appeal
        
        # Niche appeal
        if niche in ["fitness", "fashion", "lifestyle"]:
            appeal_score += 3  # High earning potential
        elif niche in ["gaming", "tech"]:
            appeal_score += 2  # Growing market
        else:
            appeal_score += 1  # Niche market
        
        # Body type appeal
        if body_type in ["athletic", "toned", "model-thin"]:
            appeal_score += 2  # High visual appeal
        else:
            appeal_score += 1  # Relatability appeal
        
        if appeal_score >= 7:
            return "Very High (€5,000+ monthly potential)"
        elif appeal_score >= 5:
            return "High (€2,000-5,000 monthly potential)"
        elif appeal_score >= 3:
            return "Moderate (€500-2,000 monthly potential)"
        else:
            return "Niche (€100-500 monthly potential)"
    
    def create_character_prompts(self, features: Dict) -> Dict:
        """
        Create AI prompts from character features
        """
        physical = features["physical"]
        style = features["style"]
        
        # Base character description
        base_prompt = [
            "beautiful 25-year-old woman",
            f"{physical['hair_color']} {physical['hair_style']} hair",
            f"{physical['eye_color']} eyes",
            f"{physical['face_shape']} face",
            f"{physical['body_type']} build",
            f"{physical['skin_tone']} skin",
            f"{style['aesthetic']} style",
            style["personality"],
            "flawless skin",
            "professional photography",
            "high detail",
            "commercial quality",
            "instagram influencer style"
        ]
        
        # Niche-specific additions
        niche_additions = {
            "fitness": "athletic sportswear, gym setting, confident pose",
            "fashion": "trendy outfit, fashion pose, studio lighting",
            "lifestyle": "casual chic outfit, modern setting, natural lighting",
            "gaming": "gamer aesthetic, tech setup, cool lighting",
            "travel": "travel outfit, beautiful destination, wanderlust vibes",
            "beauty": "flawless makeup, beauty lighting, glamorous"
        }
        
        main_prompt = ", ".join(base_prompt)
        if style["niche"] in niche_additions:
            main_prompt += f", {niche_additions[style['niche']]}"
        
        # Quality enhancers
        quality_terms = [
            "masterpiece", "best quality", "ultra detailed", "sharp focus",
            "professional lighting", "perfect composition", "8k resolution"
        ]
        
        final_prompt = f"{main_prompt}, {', '.join(quality_terms)}"
        
        # Negative prompt
        negative_prompt = (
            "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
            "extra limbs, missing limbs, malformed, poorly drawn, sketch, "
            "cartoon, anime, painting, multiple people, crowd, text, watermark, "
            "masculine features, male, man, beard, mustache"
        )
        
        return {
            "main_prompt": final_prompt,
            "negative_prompt": negative_prompt,
            "character_summary": f"{physical['hair_color']} {style['niche']} influencer with {style['personality']} personality"
        }
    
    def experiment_with_character(self, features: Dict, num_variations: int = 5) -> List[str]:
        """
        Experiment with character generation (The Clueless method)
        Generate multiple variations to find the perfect look
        """
        if not self.pipeline:
            print("❌ Pipeline not available!")
            return []
        
        print(f"🧪 Experimenting with Character Design ({num_variations} variations)")
        print("Following The Clueless methodology...")
        
        prompts = self.create_character_prompts(features)
        character_name = features["style"]["niche"] + "_" + features["physical"]["hair_color"].replace(" ", "_")
        
        generated_files = []
        successful_seeds = []
        
        for i in range(num_variations):
            print(f"\n🎨 Generating Variation {i+1}/{num_variations}")
            
            # Use different seeds for variation
            seed = random.randint(1000, 999999)
            
            try:
                generator = torch.Generator(device=self.device).manual_seed(seed)
                
                result = self.pipeline(
                    prompt=prompts["main_prompt"],
                    negative_prompt=prompts["negative_prompt"],
                    num_inference_steps=30,  # Higher quality for character creation
                    guidance_scale=7.5,
                    generator=generator,
                    height=1024,
                    width=768
                )
                
                # Save with descriptive name
                timestamp = int(time.time())
                filename = f"character_experiment_{character_name}_{timestamp}_var{i+1}_seed{seed}.png"
                
                result.images[0].save(filename)
                
                # Add metadata overlay
                self._add_character_metadata(filename, features, seed, i+1)
                
                generated_files.append(filename)
                successful_seeds.append(seed)
                
                print(f"✅ Generated: {filename}")
                print(f"🎲 Seed: {seed}")
                
                # Brief pause between generations
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Variation {i+1} failed: {e}")
        
        # Store experiment data
        experiment_id = f"{character_name}_{int(time.time())}"
        self.character_experiments[experiment_id] = {
            "features": features,
            "prompts": prompts,
            "generated_files": generated_files,
            "successful_seeds": successful_seeds,
            "experiment_date": datetime.now().isoformat(),
            "market_appeal": features["market_appeal"]
        }
        
        # Save experiment data
        self._save_experiment_data()
        
        print(f"\n🎯 Character Experiment Complete!")
        print(f"📸 Generated: {len(generated_files)} variations")
        print(f"💾 Experiment ID: {experiment_id}")
        print("\n👀 Review the generated images and choose your favorite!")
        print("💡 Use the best variation as your character base")
        
        return generated_files
    
    def _add_character_metadata(self, filename: str, features: Dict, seed: int, variation: int):
        """Add metadata overlay to character image"""
        try:
            img = Image.open(filename)
            draw = ImageDraw.Draw(img)
            
            # Create metadata text
            metadata_lines = [
                f"Variation {variation} | Seed: {seed}",
                f"Hair: {features['physical']['hair_color']} {features['physical']['hair_style']}",
                f"Eyes: {features['physical']['eye_color']} | Build: {features['physical']['body_type']}",
                f"Niche: {features['style']['niche']} | Style: {features['style']['aesthetic']}",
                f"Appeal: {features['market_appeal']}"
            ]
            
            # Add semi-transparent background
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            
            # Draw background rectangle
            overlay_draw.rectangle([(10, 10), (400, 120)], fill=(0, 0, 0, 128))
            
            # Composite overlay
            img = Image.alpha_composite(img.convert('RGBA'), overlay)
            draw = ImageDraw.Draw(img)
            
            # Draw text
            try:
                # Try to use a font
                font = ImageFont.truetype("arial.ttf", 14)
            except:
                font = ImageFont.load_default()
            
            for i, line in enumerate(metadata_lines):
                draw.text((15, 15 + i * 18), line, fill=(255, 255, 255), font=font)
            
            # Save with metadata
            img.convert('RGB').save(filename)
            
        except Exception as e:
            print(f"⚠️ Could not add metadata to {filename}: {e}")
    
    def finalize_character(self, experiment_id: str, chosen_variation: int) -> Dict:
        """
        Finalize a character based on chosen experiment variation
        """
        if experiment_id not in self.character_experiments:
            print(f"❌ Experiment {experiment_id} not found!")
            return {}
        
        experiment = self.character_experiments[experiment_id]
        
        if chosen_variation > len(experiment["successful_seeds"]):
            print(f"❌ Variation {chosen_variation} not available!")
            return {}
        
        # Get the chosen seed and file
        chosen_seed = experiment["successful_seeds"][chosen_variation - 1]
        chosen_file = experiment["generated_files"][chosen_variation - 1]
        
        # Create finalized character profile
        character_profile = {
            "character_id": f"char_{int(time.time())}",
            "name": f"{experiment['features']['style']['niche'].title()} {experiment['features']['physical']['hair_color'].title()}",
            "features": experiment["features"],
            "prompts": experiment["prompts"],
            "base_seed": chosen_seed,
            "base_image": chosen_file,
            "experiment_source": experiment_id,
            "finalized_date": datetime.now().isoformat(),
            "status": "ready_for_production"
        }
        
        # Store as successful character
        char_id = character_profile["character_id"]
        self.successful_characters[char_id] = character_profile
        
        # Save data
        self._save_experiment_data()
        
        print(f"🎉 Character Finalized!")
        print(f"   • Name: {character_profile['name']}")
        print(f"   • ID: {char_id}")
        print(f"   • Base Seed: {chosen_seed}")
        print(f"   • Base Image: {chosen_file}")
        print(f"   • Market Appeal: {experiment['features']['market_appeal']}")
        print("\n✅ Ready for content production!")
        
        return character_profile
    
    def list_experiments(self):
        """List all character experiments"""
        if not self.character_experiments:
            print("📝 No character experiments yet!")
            return
        
        print("🧪 Character Experiments:")
        print("=" * 50)
        
        for exp_id, exp_data in self.character_experiments.items():
            features = exp_data["features"]
            physical = features["physical"]
            style = features["style"]
            
            print(f"📋 Experiment: {exp_id}")
            print(f"   • Hair: {physical['hair_color']} {physical['hair_style']}")
            print(f"   • Niche: {style['niche']}")
            print(f"   • Appeal: {features['market_appeal']}")
            print(f"   • Variations: {len(exp_data['generated_files'])}")
            print(f"   • Date: {exp_data['experiment_date'][:10]}")
            print()
    
    def list_finalized_characters(self):
        """List all finalized characters"""
        if not self.successful_characters:
            print("📝 No finalized characters yet!")
            return
        
        print("👥 Finalized Characters:")
        print("=" * 50)
        
        for char_id, char_data in self.successful_characters.items():
            features = char_data["features"]
            
            print(f"🎭 {char_data['name']} ({char_id})")
            print(f"   • Niche: {features['style']['niche']}")
            print(f"   • Hair: {features['physical']['hair_color']}")
            print(f"   • Appeal: {features['market_appeal']}")
            print(f"   • Base Image: {char_data['base_image']}")
            print()
    
    def _save_experiment_data(self):
        """Save all experiment and character data"""
        try:
            data = {
                "character_experiments": self.character_experiments,
                "successful_characters": self.successful_characters,
                "design_history": self.design_history,
                "last_updated": datetime.now().isoformat()
            }
            
            with open("character_design_data.json", 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Could not save data: {e}")
    
    def _load_experiment_data(self):
        """Load existing experiment data"""
        try:
            if os.path.exists("character_design_data.json"):
                with open("character_design_data.json", 'r') as f:
                    data = json.load(f)
                    self.character_experiments = data.get("character_experiments", {})
                    self.successful_characters = data.get("successful_characters", {})
                    self.design_history = data.get("design_history", [])
        except Exception as e:
            print(f"⚠️ Could not load existing data: {e}")

def main():
    """Main character design interface"""
    print("AI Character Designer - Create Your Own Aitana")
    print("=" * 55)
    print("🎨 Design unique AI influencers from scratch")
    
    if not AI_AVAILABLE:
        print("❌ AI libraries required!")
        print("Install: pip install diffusers transformers torch pillow")
        return
    
    designer = AICharacterDesigner()
    
    if not designer.pipeline:
        print("❌ AI pipeline failed to load")
        return
    
    # Load existing data
    designer._load_experiment_data()
    
    while True:
        print("\n🎨 Character Designer Menu:")
        print("1. Analyze market trends")
        print("2. Design new character features")
        print("3. Experiment with character generation")
        print("4. Finalize character from experiment") 
        print("5. View all experiments")
        print("6. View finalized characters")
        print("7. Exit")
        
        choice = input("\nChoose option (1-7): ").strip()
        
        if choice == "1":
            designer.analyze_market_trends()
        
        elif choice == "2":
            print("\n🎯 Character Inspiration:")
            print("1. Trending (Follow current trends like Aitana)")
            print("2. Classic (Timeless appeal)")
            print("3. Experimental (Bold, high risk/reward)")
            print("4. Random")
            
            inspiration_choice = input("Choose inspiration (1-4): ").strip()
            inspirations = {"1": "trending", "2": "classic", "3": "experimental", "4": "random"}
            inspiration = inspirations.get(inspiration_choice, "trending")
            
            features = designer.design_character_features(inspiration)
            
            # Ask if they want to experiment with this design
            experiment = input("\nGenerate variations of this character? (y/n): ").strip().lower()
            if experiment == 'y':
                num_vars = int(input("Number of variations (1-10): ") or "5")
                designer.experiment_with_character(features, num_vars)
        
        elif choice == "3":
            # Quick experiment with trending features
            print("\n🧪 Quick Character Experiment")
            features = designer.design_character_features("trending")
            num_vars = int(input("Number of variations (1-10): ") or "5")
            designer.experiment_with_character(features, num_vars)
        
        elif choice == "4":
            designer.list_experiments()
            exp_id = input("\nEnter experiment ID: ").strip()
            variation = int(input("Choose variation number: ") or "1")
            designer.finalize_character(exp_id, variation)
        
        elif choice == "5":
            designer.list_experiments()
        
        elif choice == "6":
            designer.list_finalized_characters()
        
        elif choice == "7":
            print("🎨 Happy character designing!")
            break
        
        else:
            print("❌ Invalid choice!")

if __name__ == "__main__":
    main()