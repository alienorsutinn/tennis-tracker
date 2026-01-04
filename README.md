# 🎾 Tennis Tracker

A complete computer vision pipeline for tennis match analysis. Process any tennis video clip to detect court boundaries, track ball movement, detect bounces, and generate tactical 2D court views.

![Demo](https://github.com/AggieSportsAnalytics/CourtCheck/raw/main/images/game2_processed_10s.gif)

## 🌟 Features

- **Court Detection**: 17-keypoint detection for accurate court boundary mapping
- **Ball Tracking**: TrackNet-based ball trajectory tracking
- **Bounce Detection**: Automatic bounce detection with IN/OUT calls
- **Homography Transform**: Real-time perspective transformation to 2D tactical view
- **Player Tracking**: Multi-player detection and tracking
- **Speed Analysis**: Ball and player speed calculations
- **Video Export**: Side-by-side output with main view + 2D court

## 🏗️ Architecture

Based on these excellent open-source projects:

| Component | Source | Description |
|-----------|--------|-------------|
| Court Detection | [CourtCheck](https://github.com/AggieSportsAnalytics/CourtCheck) | Detectron2 keypoint detection |
| Ball Tracking | [yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet) | 3-frame CNN ball detection |
| Bounce Detection | [yastrebksv/TennisProject](https://github.com/yastrebksv/TennisProject) | CatBoost trajectory analysis |
| Court Reference | [TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector) | 14-point court model |
| Radar View | [roboflow/sports](https://github.com/roboflow/sports) | Homography & tactical view |

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/tennis-tracker.git
cd tennis-tracker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install Detectron2 (for court detection)
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## 🚀 Quick Start

### Process a Video

```bash
python tennis_pipeline.py --input match.mp4 --output analyzed.mp4
```

### With Custom Config

```bash
python tennis_pipeline.py --input match.mp4 --output analyzed.mp4 --config config.json
```

### Example Config

```json
{
  "ball_trail_length": 15,
  "keypoint_history_length": 10,
  "bounce_detection_threshold": 5.0,
  "court_2d_width": 300,
  "court_2d_height": 600
}
```

## 📁 Project Structure

```
tennis-tracker/
├── tennis_pipeline.py      # Main processing pipeline
├── requirements.txt        # Python dependencies
├── config.json            # Default configuration
├── models/                # Pre-trained model weights
│   ├── court_detection/   # Detectron2 court model
│   ├── tracknet/          # TrackNet ball detection
│   └── yolo/              # Player detection
├── scripts/
│   ├── process_video.py   # Video processing script
│   ├── train_court.py     # Court model training
│   └── annotate.py        # Annotation helper
└── data/
    ├── input/             # Input videos
    └── output/            # Processed videos
```

## 🔧 Pipeline Components

### 1. Court Keypoint Detection

Detects 17 keypoints on the tennis court:

| Keypoint | Description |
|----------|-------------|
| BTL, BTR | Top baseline corners (doubles) |
| BTLI, BTRI | Top baseline (singles) |
| ITL, ITM, ITR | Service line top |
| NL, NM, NR | Net line |
| IBL, IBM, IBR | Service line bottom |
| BBL, BBR | Bottom baseline corners |
| BBLI, BBRI | Bottom baseline (singles) |

### 2. Homography Transformation

Maps camera view to 2D court using 4-point perspective transform:

```python
# Source: Camera view corners
src_pts = [BTL, BTR, BBL, BBR]

# Destination: 2D reference court
dst_pts = court_reference.get_corner_points()

# Compute transformation matrix
matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

# Transform any point to 2D view
point_2d = cv2.perspectiveTransform(point_3d, matrix)
```

### 3. Ball Tracking (TrackNet)

Uses 3 consecutive frames for trajectory-based detection:

```python
# TrackNet takes 3 frames as input
x_pred, y_pred = detect_ball(model, frame, prev_frame, prev_prev_frame)

# Outputs gaussian heatmap centered on ball position
```

### 4. Bounce Detection

Detects bounces from vertical velocity sign changes:

```python
# Calculate vertical velocities
velocities = [pos[i].y - pos[i-1].y for i in range(1, len(positions))]

# Detect sign change (down → up = bounce)
if velocities[i-1] > threshold and velocities[i] < -threshold:
    bounce_detected = True
```

### 5. Keypoint Stabilization

Reduces jitter using temporal averaging:

```python
# Store last N positions for each keypoint
history[keypoint_name].append(current_position)

# Return average position
stabilized = np.mean(history[keypoint_name], axis=0)
```

## 🎯 Model Training

### Court Detection Model

```bash
# Download dataset
# https://drive.google.com/file/d/1lhAaeQCmk2y440PmagA0KmIVBIysVMwu

# Train with Detectron2
python scripts/train_court.py \
  --dataset data/court_keypoints \
  --output models/court_detection \
  --epochs 500
```

### Ball Detection Model (TrackNet)

```bash
# Download dataset from Roboflow
# https://universe.roboflow.com/tennisball-3eqxr/tennis-ball-detection

# Train TrackNet
python scripts/train_tracknet.py \
  --dataset data/ball_detection \
  --output models/tracknet \
  --epochs 100
```

## 📊 Output Format

### Video Output

- Left side: Original video with overlays
  - Green lines: Court boundaries
  - Yellow circles: Ball trail
  - Red/Blue circles: Players
  - Red markers: Bounce points

- Right side: 2D tactical view
  - Top-down court representation
  - Transformed player/ball positions

### JSON Analysis

```json
{
  "total_frames": 1800,
  "bounces": [
    {"frame": 45, "position": [320, 240], "confidence": 0.92},
    {"frame": 112, "position": [450, 380], "confidence": 0.87}
  ],
  "players": {
    "1": {"total_distance": 234.5, "final_position": [300, 400]},
    "2": {"total_distance": 198.2, "final_position": [320, 100]}
  }
}
```

## 🔬 Technical Details

### Supported Video Formats

- Input: MP4, AVI, MOV (any OpenCV-compatible format)
- Recommended: 1280x720 @ 30fps (as per training data)
- Output: MP4 (H.264)

### Performance

| Component | Time per Frame | GPU Memory |
|-----------|---------------|------------|
| Court Detection | ~30ms | 1.5GB |
| Ball Detection | ~15ms | 1GB |
| Player Detection | ~20ms | 1GB |
| Homography | ~1ms | - |
| **Total** | **~66ms (~15fps)** | **~3.5GB** |

### Hardware Requirements

- **Minimum**: NVIDIA GPU with 4GB VRAM
- **Recommended**: NVIDIA RTX 3060 or better
- **CPU-only**: Possible but slow (~2fps)

## 🤝 Contributing

Contributions welcome! Areas of improvement:

- [ ] Real-time processing (optimize for 30fps)
- [ ] Multi-camera support
- [ ] Automatic IN/OUT call accuracy
- [ ] Player identification (jersey numbers)
- [ ] Shot type classification
- [ ] Score tracking

## 📚 References

1. [CourtCheck - Aggie Sports Analytics](https://github.com/AggieSportsAnalytics/CourtCheck)
2. [TrackNet - Ball Tracking](https://github.com/yastrebksv/TrackNet)
3. [TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector)
4. [Roboflow Sports](https://github.com/roboflow/sports)
5. [Detectron2](https://github.com/facebookresearch/detectron2)

## 📄 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- [AggieSportsAnalytics](https://github.com/AggieSportsAnalytics) for CourtCheck
- [yastrebksv](https://github.com/yastrebksv) for TrackNet and TennisCourtDetector
- [Roboflow](https://github.com/roboflow) for sports analytics techniques
- [abdullahtarek](https://github.com/abdullahtarek) for tennis analysis concepts
