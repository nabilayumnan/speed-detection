import cv2
import numpy as np
import time
from ultralytics import YOLO
import torch
import json
import pandas as pd

# --- Configuration ---
VIDEO_PATH_IN = "FKH01/FKH01a.mp4"  # Path to your input video file
VIDEO_PATH_OUT = "output_video6.mp4"  # Path to save the processed video
YOLO_MODEL_PATH = "yolov8n.pt"  # Path to YOLOv8 model weights
CONFIDENCE_THRESHOLD = 0.25  # Confidence threshold for YOLO detections
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO class IDs (car, motorcycle, bus, truck)

# Process optimization settings
PROCESS_EVERY_N_FRAMES = 2  # Process detection every N frames for better performance
MAX_PROCESSING_WIDTH = 640  # Resize detection frames to this width
USE_CUDA = True  # Use CUDA if available

# Line-crossing configuration
LINE_1_Y = 600  # Y-coordinate of the first virtual line
LINE_2_Y = 700  # Y-coordinate of the second virtual line
PIXEL_TO_METER_RATIO = 0.05  # 1 pixel = 0.05 meters


# --- Simple SORT Tracker ---
class SimpleSORT:
    def __init__(self, max_age=8, min_hits=3, iou_threshold=0.3):
        self.tracks = []
        self.next_id = 0
        self.max_age = max_age  # Maximum frames to keep a track without detection
        self.min_hits = min_hits  # Minimum detections before track is confirmed
        self.iou_threshold = iou_threshold
        self.track_history = {}  # For velocity prediction
        self.age = {}  # Track age counter
        self.hits = {}  # Track detection counter
        self.smoothed_boxes = {}  # For smoothing box positions
        self.smoothing_factor = 0.7  # Smoothing factor for box positions
    
    def update(self, detections):
        # Age existing tracks
        for track_id, _ in list(self.tracks):
            self.age[track_id] = self.age.get(track_id, 0) + 1

        # Predict new positions using simple velocity model
        predicted_tracks = []
        for track_id, bbox in self.tracks:
            if track_id in self.track_history and len(self.track_history[track_id]) >= 2:
                # Calculate velocity from last positions
                prev_boxes = self.track_history[track_id][-2:]
                vel_x1 = prev_boxes[1][0] - prev_boxes[0][0]
                vel_y1 = prev_boxes[1][1] - prev_boxes[0][1]
                vel_x2 = prev_boxes[1][2] - prev_boxes[0][2]
                vel_y2 = prev_boxes[1][3] - prev_boxes[0][3]
                
                # Predict new position
                pred_x1 = bbox[0] + vel_x1
                pred_y1 = bbox[1] + vel_y1
                pred_x2 = bbox[2] + vel_x2
                pred_y2 = bbox[3] + vel_y2
                
                predicted_tracks.append((track_id, [pred_x1, pred_y1, pred_x2, pred_y2]))
            else:
                predicted_tracks.append((track_id, bbox))

        # Match tracks with detections
        matched_detections = np.zeros(len(detections), dtype=bool)
        updated_tracks = []
        
        # Keep only tracks that aren't too old
        active_tracks = [(t_id, t_bbox) for t_id, t_bbox in predicted_tracks 
                         if self.age.get(t_id, 0) <= self.max_age]
        
        # Match using IoU
        for i, (track_id, track_bbox) in enumerate(active_tracks):
            best_match_iou = self.iou_threshold
            best_match_idx = -1
            
            for j, det_bbox in enumerate(detections):
                if not matched_detections[j]:
                    iou = self._calculate_iou(track_bbox, det_bbox)
                    if iou > best_match_iou:
                        best_match_iou = iou
                        best_match_idx = j
            
            if best_match_idx != -1:
                # Update track with new detection
                matched_detections[best_match_idx] = True
                updated_tracks.append((track_id, detections[best_match_idx]))
                self.age[track_id] = 0  # Reset age
                self.hits[track_id] = self.hits.get(track_id, 0) + 1  # Increment hits
                
                # Update track history
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                self.track_history[track_id].append(detections[best_match_idx])
                if len(self.track_history[track_id]) > 10:  # Keep history limited
                    self.track_history[track_id].pop(0)

        # Create new tracks for unmatched detections
        new_tracks = []
        for i, det_bbox in enumerate(detections):
            if not matched_detections[i]:
                track_id = self.next_id
                new_tracks.append((track_id, det_bbox))
                self.track_history[track_id] = [det_bbox]
                self.age[track_id] = 0
                self.hits[track_id] = 1
                self.next_id += 1

        # Update tracks list
        self.tracks = [(t_id, t_bbox) for t_id, t_bbox in updated_tracks 
                      if self.hits.get(t_id, 0) >= self.min_hits]
        self.tracks.extend(new_tracks)
        
        return self.tracks

    def _calculate_iou(self, boxA, boxB):
        # Fast IoU calculation
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

# --- Speed Calculation ---
def calculate_speed_kmh(pixel_distance, time_seconds, pixel_to_meter):
    """Calculates speed in km/h given pixel distance, time, and conversion factor."""
    if time_seconds <= 0.05:  # Minimum time threshold
        return 0
    distance_meters = pixel_distance * pixel_to_meter
    speed_mps = distance_meters / time_seconds
    speed_kmh = speed_mps * 3.6
    
    # Add reasonable speed validation
    if 5 <= speed_kmh <= 120:  # Reasonable vehicle speed range
        return speed_kmh
    return 0

# --- Detection Processing ---
def process_frame(frame, model, classes, conf_thresh, max_width):
    """Process frame for detection with resizing for better performance"""
    h, w = frame.shape[:2]
    
    # Resize for faster processing while maintaining aspect ratio
    scale = max_width / w
    process_w = max_width
    process_h = int(h * scale)
    
    if w > max_width:  # Only resize if needed
        process_frame = cv2.resize(frame, (process_w, process_h))
    else:
        process_frame = frame
        scale = 1.0
    
    # Run detection
    results = model.predict(
        source=process_frame,
        conf=conf_thresh,
        classes=classes,
        verbose=False,
        device='cuda' if USE_CUDA and torch.cuda.is_available() else 'cpu'
    )
    
    # Format detections
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            
            # Scale back to original frame size
            if scale != 1.0:
                x1 /= scale
                y1 /= scale
                x2 /= scale
                y2 /= scale
            
            detections.append([int(x1), int(y1), int(x2), int(y2)])
    
    return detections

# --- Visualization Functions ---
def draw_visualizations(frame, tracked_vehicles, vehicle_speed_data):
    """Draw visualizations including bounding boxes, IDs, and speeds"""
    # Draw speed measurement lines
    h, w = frame.shape[:2]
    cv2.line(frame, (0, LINE_1_Y), (w, LINE_1_Y), (255, 0, 0), 2)
    cv2.putText(frame, "Line 1", (10, LINE_1_Y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
    cv2.line(frame, (0, LINE_2_Y), (w, LINE_2_Y), (0, 0, 255), 2)
    cv2.putText(frame, "Line 2", (10, LINE_2_Y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Draw bounding boxes and vehicle information
    for track_id, bbox in tracked_vehicles:
        x1, y1, x2, y2 = map(int, bbox)
        
        # Calculate centroid for visualization
        centroid_x = (x1 + x2) // 2
        centroid_y = (y1 + y2) // 2
        
        # Draw bounding box
        box_color = (0, 255, 0)  # Default green
        
        # Change color based on position relative to lines
        if centroid_y < LINE_1_Y:
            box_color = (255, 150, 0)  # Orange - before first line
        elif LINE_1_Y <= centroid_y <= LINE_2_Y:
            box_color = (0, 255, 255)  # Yellow - between lines (measuring)
        else:
            box_color = (0, 255, 0)  # Green - past second line (measured)
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.circle(frame, (centroid_x, centroid_y), 4, box_color, -1)
        
        # Draw vehicle ID and speed if available
        label = f"ID: {track_id}"
        if track_id in vehicle_speed_data:
            label += f" {vehicle_speed_data[track_id]} km/h"
        
        cv2.putText(frame, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
    
    # Show statistics
    valid_speeds = [speed for speed in vehicle_speed_data.values() if isinstance(speed, (int, float)) and speed > 0]
    avg_speed = sum(valid_speeds) / max(len(valid_speeds), 1) if valid_speeds else 0
    max_speed = max(valid_speeds) if valid_speeds else 0
    
    # Draw statistics background
    cv2.rectangle(frame, (10, 10), (250, 100), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (250, 100), (255, 255, 255), 1)
    
    # Draw statistics text
    cv2.putText(frame, f"Vehicles tracked: {len(vehicle_speed_data)}", 
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Avg speed: {avg_speed:.1f} km/h", 
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Max speed: {max_speed:.1f} km/h", 
                (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

# --- Main Processing ---
def main():
    print("Vehicle Speed Detection System Starting...")
    
    # Check CUDA availability
    device = 'cuda' if USE_CUDA and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load YOLOv8 model
    print(f"Loading YOLO model from {YOLO_MODEL_PATH}...")
    model = YOLO(YOLO_MODEL_PATH)
    print("Model loaded successfully.")

    # Initialize video capture
    print(f"Opening video file: {VIDEO_PATH_IN}")
    cap = cv2.VideoCapture(VIDEO_PATH_IN)
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH_IN}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
    out = cv2.VideoWriter(VIDEO_PATH_OUT, fourcc, fps, (frame_width, frame_height))

    # Initialize tracker with optimized parameters
    tracker = SimpleSORT(max_age=10, min_hits=2, iou_threshold=0.3)

    # Data structures for speed calculation
    vehicle_enter_time = {}  # When vehicle crosses line 1
    vehicle_speed_data = {}  # Calculated speeds

    frame_count = 0
    processing_times = []
    start_time = time.time()
    
    print("Starting video processing loop...")
    
    # Main processing loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time_sec = frame_count / fps

        # Process detection only every N frames for performance
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            process_start = time.time()
            
            # Detect vehicles
            detections = process_frame(
                frame, 
                model, 
                VEHICLE_CLASSES, 
                CONFIDENCE_THRESHOLD, 
                MAX_PROCESSING_WIDTH
            )
            
            # Update tracker
            tracked_vehicles = tracker.update(detections)
            
           # In the main loop, replace the speed calculation section with this:
            # Speed calculation
            for track_id, bbox in tracked_vehicles:
                centroid_x = (bbox[0] + bbox[2]) // 2
                centroid_y = (bbox[1] + bbox[3]) // 2
                
                # Wider detection window for line crossing (15 pixels)
                # Record time when vehicle crosses first line
                if track_id not in vehicle_enter_time and abs(centroid_y - LINE_1_Y) <= 15:
                    vehicle_enter_time[track_id] = current_time_sec
                    print(f"Vehicle {track_id} crossed Line 1 at {current_time_sec:.2f}s")
                
                # Calculate speed when vehicle crosses second line
                elif track_id in vehicle_enter_time and track_id not in vehicle_speed_data and abs(centroid_y - LINE_2_Y) <= 15:
                    time_diff = current_time_sec - vehicle_enter_time[track_id]
                    distance_pixels = abs(LINE_2_Y - LINE_1_Y)
                    
                    # Only calculate speed if sensible time difference (avoid false crossings)
                    if 0.1 <= time_diff <= 10:  # Reasonable time range (0.1-10 seconds)
                        speed = calculate_speed_kmh(distance_pixels, time_diff, PIXEL_TO_METER_RATIO)
                        if speed > 0:
                            vehicle_speed_data[track_id] = speed
                            print(f"Vehicle {track_id} speed: {speed:.1f} km/h (took {time_diff:.2f}s)")
            
            process_end = time.time()
            processing_times.append(process_end - process_start)
        
        # Draw visualization on every frame for smooth output
        frame = draw_visualizations(frame, tracker.tracks, vehicle_speed_data)
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
                    (frame_width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Write frame to output video
        out.write(frame)
        
        # Display frame if needed (comment out for faster processing)
        cv2.imshow('Vehicle Speed Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit early
            break
            
        # Print progress every 100 frames
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_count / elapsed
            print(f"Processed {frame_count}/{total_frames} frames ({(frame_count/total_frames*100):.1f}%) - {fps_actual:.1f} FPS")

    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Print processing statistics
    avg_proc_time = sum(processing_times) / len(processing_times) if processing_times else 0
    print(f"\nProcessing complete!")
    print(f"Total processing time: {time.time() - start_time:.1f} seconds")
    print(f"Average processing time per detection frame: {avg_proc_time*1000:.1f} ms")
    print(f"Vehicles tracked: {len(vehicle_speed_data)}")
    
    if vehicle_speed_data:
        valid_speeds = [s for s in vehicle_speed_data.values() if s > 0]
        print(f"Average vehicle speed: {sum(valid_speeds)/len(valid_speeds):.1f} km/h")
        print(f"Maximum vehicle speed: {max(valid_speeds):.1f} km/h")
    
    print(f"Output saved to: {VIDEO_PATH_OUT}")

if __name__ == "__main__":
    main()