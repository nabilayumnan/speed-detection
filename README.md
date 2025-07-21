# Vehicle Speed Detection System

A computer vision-based system for detecting and tracking vehicles in video footage while calculating their speeds using YOLOv8 object detection and SimpleSORT tracking algorithms.

## Overview

This project implements an automated vehicle speed detection system that processes video files to identify vehicles, track their movement across predefined virtual lines, and calculate their speeds. The system uses deep learning for object detection and computer vision techniques for speed measurement.

## Features

- Real-time vehicle detection using YOLOv8
- Multi-object tracking with SimpleSORT algorithm
- Speed calculation based on line-crossing methodology
- Visual output with bounding boxes, vehicle IDs, and speed annotations
- Support for multiple vehicle types (cars, motorcycles, buses, trucks)
- Performance optimization for efficient processing
- Comprehensive statistics and reporting

## Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for improved performance)
- YOLOv8 model weights (automatically downloaded)

### Dependencies

```bash
pip install ultralytics opencv-python numpy torch torchvision pandas
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

## Installation

1. Clone or download the project files
2. Install the required dependencies using pip
3. Ensure you have a video file for processing
4. Run the main script

## Usage

### Basic Usage

```python
python Speed_Detection_SORT.py
```

### Configuration

The main configuration parameters can be modified at the top of the script:

```python
# Video file paths
VIDEO_PATH_IN = "FKH01/FKH01a.mp4"      # Input video file
VIDEO_PATH_OUT = "output_video6.mp4"     # Output video file

# Detection parameters
CONFIDENCE_THRESHOLD = 0.25              # YOLO confidence threshold
VEHICLE_CLASSES = [2, 3, 5, 7]         # Vehicle class IDs (car, motorcycle, bus, truck)

# Speed measurement setup
LINE_1_Y = 600                          # First measurement line Y-coordinate
LINE_2_Y = 700                          # Second measurement line Y-coordinate
PIXEL_TO_METER_RATIO = 0.05            # Conversion factor (pixels to meters)

# Performance optimization
PROCESS_EVERY_N_FRAMES = 2              # Process every N frames
MAX_PROCESSING_WIDTH = 640              # Maximum width for processing
USE_CUDA = True                         # Enable GPU acceleration
```

## How It Works

The system operates through several key phases:

1. **Vehicle Detection**: Uses YOLOv8 to detect vehicles in each video frame
2. **Object Tracking**: Employs SimpleSORT algorithm to maintain consistent vehicle identities across frames
3. **Speed Measurement**: Calculates vehicle speeds when they cross predefined virtual lines
4. **Visualization**: Overlays detection results, tracking information, and speed data onto the video
5. **Output Generation**: Produces an annotated video file with all visualizations

### Speed Calculation Methodology

The system uses a line-crossing approach:
- Two virtual horizontal lines are placed across the roadway
- When a vehicle crosses the first line, the system records the timestamp
- When the same vehicle crosses the second line, it calculates the time difference
- Speed is computed using: `Speed = Distance / Time * Conversion_Factor`

## Output

The system generates:

- **Annotated Video**: Shows bounding boxes around detected vehicles with IDs and calculated speeds
- **Console Output**: Real-time processing statistics and detected vehicle information
- **Performance Metrics**: Processing time and frame rate information

### Visual Elements

- **Bounding Boxes**: Color-coded based on vehicle position relative to measurement lines
- **Vehicle IDs**: Persistent identification numbers for tracked vehicles  
- **Speed Display**: Real-time speed readings for measured vehicles
- **Statistics Panel**: Shows total vehicles tracked, average speed, and maximum speed
- **Measurement Lines**: Visual indicators of the speed measurement zone

## Project Structure

```
project/
├── Speed_Detection_SORT.py           # Main application script
├── yolov8n.pt                        # YOLOv8 model weights
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
├── FKH01/                           # Sample input video folder
│   └── FKH01a.mp4                   # Sample video file
├── output_video*.mp4                # Generated output videos
└── vehicle_speed_data.json          # Speed measurement data
```

## Configuration Parameters

### Detection Settings
- `CONFIDENCE_THRESHOLD`: Minimum confidence for vehicle detection (0.0-1.0)
- `VEHICLE_CLASSES`: COCO class IDs for vehicles to detect
- `YOLO_MODEL_PATH`: Path to YOLOv8 model weights

### Speed Measurement
- `LINE_1_Y`, `LINE_2_Y`: Y-coordinates of measurement lines
- `PIXEL_TO_METER_RATIO`: Conversion factor from pixels to real-world distance

### Performance Optimization
- `PROCESS_EVERY_N_FRAMES`: Frame processing interval for performance
- `MAX_PROCESSING_WIDTH`: Maximum width for detection processing
- `USE_CUDA`: Enable GPU acceleration if available

### Tracking Parameters
- `max_age`: Maximum frames to keep a track without detection
- `min_hits`: Minimum detections before confirming a track
- `iou_threshold`: Intersection over Union threshold for track matching

## Troubleshooting

### Common Issues

**Module Import Errors**
```bash
# Install missing dependencies
pip install ultralytics opencv-python numpy torch torchvision pandas
```

**CUDA Memory Issues**
- Set `USE_CUDA = False` to use CPU processing
- Reduce `MAX_PROCESSING_WIDTH` to decrease memory usage
- Increase `PROCESS_EVERY_N_FRAMES` to process fewer frames

**Video File Issues**
- Ensure video file path is correct
- Supported formats: .mp4, .avi, .mov, .mkv
- Verify file is not corrupted

**Inaccurate Speed Measurements**
- Adjust `PIXEL_TO_METER_RATIO` based on camera perspective
- Ensure measurement lines are positioned correctly
- Verify sufficient distance between measurement lines

## Performance Considerations

- **GPU Acceleration**: Enable CUDA for significantly faster processing
- **Frame Processing**: Adjust `PROCESS_EVERY_N_FRAMES` to balance speed vs accuracy
- **Resolution**: Lower `MAX_PROCESSING_WIDTH` for faster processing
- **Video Length**: Longer videos require more processing time and memory

## Technical Specifications

- **Supported Video Formats**: MP4, AVI, MOV, MKV
- **Minimum Python Version**: 3.8
- **Recommended Hardware**: GPU with CUDA support
- **Processing Speed**: 15-30 FPS (hardware dependent)
- **Detection Accuracy**: Optimized for traffic surveillance scenarios
- **Speed Measurement Range**: 5-120 km/h (configurable)

## Sample Output

The system provides detailed console output during processing:

```
Vehicle Speed Detection System Starting...
Using device: cuda
Loading YOLO model from yolov8n.pt...
Model loaded successfully.
Opening video file: FKH01/FKH01a.mp4
Video properties: 1920x1080, 25.0 FPS, 750 frames

Vehicle 1 crossed Line 1 at 5.32s
Vehicle 1 speed: 45.2 km/h (took 2.1s)
Vehicle 3 crossed Line 1 at 8.15s
Vehicle 3 speed: 52.7 km/h (took 1.9s)

Processing complete!
Total processing time: 45.3 seconds
Vehicles tracked: 15
Average vehicle speed: 48.5 km/h
Maximum vehicle speed: 67.3 km/h
Output saved to: output_video6.mp4
```

## Dependencies

The project requires the following Python packages:

- `ultralytics` - YOLOv8 object detection
- `opencv-python` - Computer vision operations
- `numpy` - Numerical computing
- `torch` - PyTorch deep learning framework
- `torchvision` - Computer vision for PyTorch
- `pandas` - Data manipulation and analysis

## License

This project is open source and available under standard licensing terms.

## Acknowledgments

- Ultralytics team for YOLOv8 implementation
- OpenCV community for computer vision tools
- PyTorch team for deep learning framework
