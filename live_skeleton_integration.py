"""
Live Skeleton Integration System
Combines real-time tracking with 3D skeleton mapping
"""

import cv2
import time
from complete_human_tracker import CompleteHumanTracker
from skeleton_mapper import SkeletonMapper
import numpy as np
from dataclasses import asdict

class LiveSkeletonSystem:
    """Real-time tracking to 3D skeleton conversion"""
    
    def __init__(self):
        print("🎭 Live Skeleton Integration System")
        print("=" * 50)
        
        # Initialize components
        self.tracker = CompleteHumanTracker()
        self.mapper = SkeletonMapper()
        
        # Animation data storage
        self.skeleton_sequence = []
        self.frame_count = 0
        
        print("✅ Live skeleton system ready!")
        print("   • Real-time tracking: ON")
        print("   • 3D skeleton mapping: ON")
        print("   • Animation recording: ON")
    
    def start_live_mapping(self, camera_index=0):
        """Start live tracking and skeleton mapping"""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_index}")
            return
        
        print(f"🎥 Starting live skeleton mapping")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save skeleton animation")
        print("  'r' - Reset recording")
        print("  'SPACE' - Toggle skeleton overlay")
        
        show_skeleton_info = True
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process with tracker
            tracking_data = self.tracker.process_frame(frame)
            
            # Convert to 3D skeleton
            skeleton = None
            if tracking_data:
                try:
                    # Convert tracking data to dict format for mapper
                    tracking_dict = asdict(tracking_data)
                    
                    # Debug: Print what we're sending to mapper
                    if self.frame_count % 30 == 0:  # Every 30 frames
                        print(f"Debug: Tracking data has body_keypoints_2d: {tracking_dict.get('body_keypoints_2d') is not None}")
                        if tracking_dict.get('body_keypoints_2d'):
                            print(f"  Body keypoints length: {len(tracking_dict['body_keypoints_2d'])}")
                    
                    skeleton = self.mapper.map_tracking_to_skeleton(tracking_dict)
                    
                    if skeleton:
                        self.skeleton_sequence.append(skeleton)
                        self.frame_count += 1
                        
                        # Debug: Print skeleton info occasionally
                        if self.frame_count % 30 == 0:
                            print(f"Debug: Skeleton created successfully!")
                            print(f"  Root position: {skeleton.root.position}")
                            print(f"  Has spine: {skeleton.spine is not None}")
                    else:
                        if self.frame_count % 30 == 0:
                            print("Debug: No skeleton created from tracking data")
                            
                except Exception as e:
                    if self.frame_count % 30 == 0:
                        print(f"Debug: Error creating skeleton: {e}")
                    skeleton = None
            
            # Draw tracking overlay
            frame = self.tracker.draw_complete_tracking(frame, tracking_data)
            
            # Draw skeleton information
            if show_skeleton_info and skeleton:
                self._draw_skeleton_info(frame, skeleton)
            
            # Show frame
            cv2.imshow('Live Skeleton Mapping System', frame)
            
            # Handle controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self._save_skeleton_animation()
            elif key == ord('r'):
                self._reset_recording()
            elif key == ord(' '):
                show_skeleton_info = not show_skeleton_info
                print(f"Skeleton info: {'ON' if show_skeleton_info else 'OFF'}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Auto-save on exit
        if self.skeleton_sequence:
            self._save_skeleton_animation()
    
    def _draw_skeleton_info(self, frame, skeleton):
        """Draw minimal 3D skeleton information"""
        if not skeleton:
            return
            
        h, w = frame.shape[:2]
        
        # Much smaller, compact info in top-right corner
        x_start = w - 200
        y_start = 10
        
        # Small dark background
        cv2.rectangle(frame, (x_start - 5, y_start), 
                     (w - 5, y_start + 80), (0, 0, 0), -1)
        
        # Minimal info - just what's essential
        info_lines = [
            f"3D: {len(self.skeleton_sequence)}f",  # Frame count
            f"R: {skeleton.root.position[0]:.0f},{skeleton.root.position[1]:.0f}",  # Root position (simplified)
        ]
        
        # Quick status (one line)
        working_parts = 0
        if skeleton.spine: working_parts += 1
        if skeleton.left_shoulder and skeleton.right_shoulder: working_parts += 1
        if skeleton.left_hip and skeleton.right_hip: working_parts += 1
        if skeleton.left_hand_bones or skeleton.right_hand_bones: working_parts += 1
        
        info_lines.append(f"Parts: {working_parts}/4")
        
        # Draw compact info
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (x_start, y_start + 20 + i * 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    def _save_skeleton_animation(self):
        """Save recorded skeleton animation"""
        if not self.skeleton_sequence:
            print("❌ No skeleton data to save!")
            return
        
        # Export in multiple formats
        print(f"💾 Saving {len(self.skeleton_sequence)} skeleton frames...")
        
        try:
            # JSON format (for Unity/custom systems)
            json_file = self.mapper.export_skeleton_animation(self.skeleton_sequence, 'json')
            print(f"✅ JSON animation saved: {json_file}")
            
            # BVH format (for Blender/Maya)
            bvh_file = self.mapper.export_skeleton_animation(self.skeleton_sequence, 'bvh')
            print(f"✅ BVH animation saved: {bvh_file}")
            
            # Summary statistics
            duration = len(self.skeleton_sequence) / 30.0  # Assume 30 FPS
            print(f"📊 Animation stats:")
            print(f"   • Duration: {duration:.1f} seconds")
            print(f"   • Frames: {len(self.skeleton_sequence)}")
            print(f"   • Average FPS: {len(self.skeleton_sequence) / (time.time() - self._start_time) if hasattr(self, '_start_time') else 'N/A'}")
            
        except Exception as e:
            print(f"❌ Error saving animation: {e}")
    
    def _reset_recording(self):
        """Reset skeleton recording"""
        self.skeleton_sequence = []
        self.frame_count = 0
        self._start_time = time.time()
        print("🔄 Skeleton recording reset!")
    
    def process_video_file(self, video_path: str, output_path: str = None):
        """Process a video file and extract skeleton animation"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"🎬 Processing video: {video_path}")
        print(f"   • Total frames: {total_frames}")
        print(f"   • FPS: {fps}")
        
        skeleton_sequence = []
        frame_number = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            tracking_data = self.tracker.process_frame(frame)
            
            if tracking_data:
                tracking_dict = asdict(tracking_data)
                skeleton = self.mapper.map_tracking_to_skeleton(tracking_dict)
                
                if skeleton:
                    skeleton_sequence.append(skeleton)
            
            frame_number += 1
            if frame_number % 30 == 0:
                progress = (frame_number / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_number}/{total_frames})")
        
        cap.release()
        
        # Save results
        if skeleton_sequence:
            self.skeleton_sequence = skeleton_sequence
            self._save_skeleton_animation()
            print(f"✅ Video processing complete: {len(skeleton_sequence)} skeleton frames extracted")
        else:
            print("❌ No skeleton data extracted from video")

def main():
    """Main function"""
    print("Live Skeleton Integration System")
    print("=" * 40)
    
    system = LiveSkeletonSystem()
    
    print("\nChoose an option:")
    print("1. Live camera skeleton mapping")
    print("2. Process video file")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        camera_index = input("Enter camera index (0 for default): ").strip()
        try:
            camera_index = int(camera_index) if camera_index else 0
        except:
            camera_index = 0
        system.start_live_mapping(camera_index)
    
    elif choice == "2":
        video_path = input("Enter path to video file: ").strip()
        system.process_video_file(video_path)
    
    elif choice == "3":
        print("Goodbye!")
        return
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()