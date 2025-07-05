"""
Robust AI Model Downloader with Resume Capability
Handles interrupted downloads and prevents sleep during model downloading
"""

import os
import sys
import time
import requests
import json
from pathlib import Path
import platform

# Prevent system sleep during download
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
            print("🔒 Sleep prevention activated")
        
        def allow_sleep():
            """Allow Windows to sleep again"""
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            print("🔓 Sleep prevention deactivated")
    
    elif platform.system() == "Darwin":  # macOS
        import subprocess
        
        def prevent_sleep():
            subprocess.Popen(['caffeinate', '-d'])
            print("🔒 Sleep prevention activated (macOS)")
        
        def allow_sleep():
            # Kill caffeinate process
            os.system("pkill caffeinate")
            print("🔓 Sleep prevention deactivated (macOS)")
    
    else:  # Linux
        def prevent_sleep():
            print("💡 On Linux, please disable sleep manually during download")
        
        def allow_sleep():
            pass
            
except ImportError:
    def prevent_sleep():
        print("💡 Please keep your computer awake during download")
    
    def allow_sleep():
        pass

def setup_huggingface_cache():
    """Setup HuggingFace cache and authentication"""
    print("🔧 Setting up HuggingFace configuration...")
    
    # Set cache directory
    cache_dir = Path.home() / ".cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Set environment variables for better downloads
    os.environ['HF_HUB_CACHE'] = str(cache_dir)
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
    
    # Fix Windows symlink issues by disabling symlinks
    if platform.system() == "Windows":
        os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'
        print("🔧 Windows symlinks disabled for compatibility")
    
    print(f"✅ Cache directory: {cache_dir}")
    
    # Check for authentication token
    token_file = cache_dir / "token"
    if not token_file.exists():
        print("\n💡 Optional: For faster downloads, get a HuggingFace token:")
        print("   1. Go to https://huggingface.co/settings/tokens")
        print("   2. Create a 'READ' token")
        print("   3. Run: huggingface-cli login")
        print("   (This is optional but recommended)\n")
    
    return cache_dir

def check_disk_space():
    """Check available disk space"""
    cache_dir = Path.home() / ".cache" / "huggingface"
    
    if platform.system() == "Windows":
        free_bytes = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(
            ctypes.c_wchar_p(str(cache_dir.drive)), 
            ctypes.pointer(free_bytes), 
            None, None
        )
        free_gb = free_bytes.value / (1024**3)
    else:
        statvfs = os.statvfs(cache_dir)
        free_gb = (statvfs.f_frsize * statvfs.f_available) / (1024**3)
    
    print(f"💾 Available disk space: {free_gb:.1f} GB")
    
    if free_gb < 15:
        print("⚠️  Warning: You need at least 15GB free space for AI models")
        print("   Consider freeing up disk space before continuing")
        return False
    
    return True

def install_requirements():
    """Install required packages with compatible versions"""
    print("📦 Installing AI libraries...")
    
    # Install packages one by one for better error handling
    packages = [
        "torch",  # Latest stable version
        "torchvision", 
        "transformers",
        "diffusers",
        "accelerate",
        "safetensors",
        "Pillow",
        "requests",
        "huggingface-hub"
    ]
    
    for package in packages:
        print(f"📥 Installing {package}...")
        result = os.system(f"pip install {package}")
        if result != 0:
            print(f"❌ Failed to install {package}")
            return False
        else:
            print(f"✅ Installed {package}")
    
    return True

def download_with_resume():
    """Download AI models with resume capability"""
    print("🚀 Starting AI Model Download with Resume Support")
    print("=" * 60)
    
    # Prevent sleep during download
    prevent_sleep()
    
    try:
        # Setup environment
        cache_dir = setup_huggingface_cache()
        
        if not check_disk_space():
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                return False
        
        # Install requirements
        if not install_requirements():
            print("❌ Failed to install requirements")
            return False
        
        # Import after installation
        try:
            from huggingface_hub import snapshot_download
            import torch
            print("✅ Libraries imported successfully")
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("Try restarting your terminal and running again")
            return False
        
        # Download models with resume
        models_to_download = [
            {
                'repo_id': 'lllyasviel/sd-controlnet-openpose',
                'name': 'ControlNet OpenPose',
                'size': '1.5GB'
            },
            {
                'repo_id': 'runwayml/stable-diffusion-v1-5', 
                'name': 'Stable Diffusion 1.5',
                'size': '4.2GB'
            }
        ]
        
        for i, model_info in enumerate(models_to_download, 1):
            print(f"\n📥 Downloading {model_info['name']} ({model_info['size']}) [{i}/{len(models_to_download)}]")
            print("=" * 50)
            
            try:
                # Download with resume support
                model_path = snapshot_download(
                    repo_id=model_info['repo_id'],
                    cache_dir=cache_dir,
                    resume_download=True,  # KEY: Resume interrupted downloads
                    local_files_only=False,
                    force_download=False,  # Don't re-download existing files
                    token=None  # Use logged-in token if available
                )
                
                print(f"✅ Successfully downloaded {model_info['name']}")
                print(f"   Location: {model_path}")
                
            except Exception as e:
                print(f"❌ Error downloading {model_info['name']}: {e}")
                
                # Check if partial download exists
                partial_path = cache_dir / ("models--" + model_info['repo_id'].replace('/', '--'))
                if partial_path.exists():
                    print(f"💡 Partial download found at: {partial_path}")
                    print("   You can resume this download by running the script again")
                
                return False
        
        print("\n🎉 ALL MODELS DOWNLOADED SUCCESSFULLY!")
        print("✅ Ready to run photorealistic AI generation")
        
        return True
        
    finally:
        # Re-enable sleep
        allow_sleep()

def test_models():
    """Test if models are properly downloaded and working"""
    print("\n🧪 Testing AI models...")
    
    try:
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
        import torch
        
        print("📋 Checking model files...")
        
        # Check ControlNet
        try:
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-openpose",
                torch_dtype=torch.float32,
                local_files_only=True  # Only use downloaded files
            )
            print("✅ ControlNet model loads successfully")
        except Exception as e:
            print(f"❌ ControlNet error: {e}")
            return False
        
        # Check Stable Diffusion
        try:
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float32,
                local_files_only=True,  # Only use downloaded files
                safety_checker=None,
                requires_safety_checker=False
            )
            print("✅ Stable Diffusion pipeline loads successfully")
        except Exception as e:
            print(f"❌ Stable Diffusion error: {e}")
            return False
        
        print("🎉 ALL MODELS WORKING CORRECTLY!")
        print("🚀 Ready to generate photorealistic humans!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main download and test function"""
    print("Robust AI Model Downloader")
    print("=" * 40)
    
    print("\n🎯 This will download:")
    print("   • ControlNet OpenPose (~1.5GB)")
    print("   • Stable Diffusion 1.5 (~4.2GB)")
    print("   • Total: ~6GB download")
    
    print("\n🔒 Features:")
    print("   • Prevents computer sleep during download")
    print("   • Resumes interrupted downloads") 
    print("   • Won't re-download completed files")
    print("   • Robust error handling")
    
    response = input("\nStart download? (y/n): ")
    if response.lower() != 'y':
        print("Download cancelled")
        return
    
    # Download models
    success = download_with_resume()
    
    if success:
        # Test models
        test_success = test_models()
        
        if test_success:
            print("\n🎬 READY TO RUN AI HUMAN GENERATOR!")
            print("Run: python ai_human_generator.py")
        else:
            print("\n⚠️  Models downloaded but testing failed")
            print("Try running ai_human_generator.py anyway")
    else:
        print("\n❌ Download failed or incomplete")
        print("💡 You can run this script again to resume downloads")

if __name__ == "__main__":
    main()