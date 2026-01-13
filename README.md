# MonkeySee

Combined AprilTag and Game Piece Detection system for FRC Team 846.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running

The system automatically detects your platform:
- **Windows/Mac**: Uses `config_dev.json` (single camera, ID 0)
- **Linux**: Uses `config.json` (full multi-camera setup)

Windows:
```bash
boot.bat
```

Linux/Mac:
```bash
chmod +x boot.sh
./boot.sh
```

Force development mode:
```bash
python main.py --dev
```

Override with specific config:
```bash
python main.py --config my_config.json
```

### Accessing the Dashboard

The dashboard port is determined by the lowest camera ID in your configuration:
- **Camera 0 (dev)**: http://localhost:5800
- **Camera 1**: http://localhost:5801
- **Camera 2**: http://localhost:5802
- **Camera 3**: http://localhost:5803
- **Camera 4**: http://localhost:5804

Formula: `port = 5800 + min(camera_ids)`

### Development vs Production Mode

**Development Mode (Windows/Mac)**
- Uses `config_dev.json`
- Single camera with ID 0
- Uses default webcam
- Perfect for testing without multiple cameras
- NetworkTables points to localhost

**Production Mode (Linux)**
- Uses `config.json`
- Up to 4 cameras
- Uses specific camera device IDs
- Full multi-camera support
- NetworkTables points to robot IP

### Configuration

Edit `config.json` (production) or `config_dev.json` (development) to configure cameras:

```json
{
  "cameras": [
    {
      "id": 1,
      "name": "Camera 1",
      "device_id": "ATCam1",
      "pipeline": "apriltag",
      "enabled": true
    }
  ],
  "network_tables": {
    "server": "10.8.46.2",
    "team_number": 846
  },
  "valid_apriltag_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
}
```

**Configuration Options:**
- `cameras`: Array of camera configurations
- `network_tables`: NetworkTables server and team number
- `valid_apriltag_ids`: List of valid AprilTag IDs (1-32) for detection filtering

## Camera Calibration

All camera calibrations are stored in a single `cal.json` file. Each camera ID has its own calibration data.

### Calibration File Format

```json
{
  "0": {
    "x": {
      "a": 0.0018803150360736042,
      "b": -1.4941116971361661e-09,
      "c": -4.5622203794325465e-15
    },
    "y": {
      "a": 0.0017131708207895275,
      "b": -1.8890823506313983e-09,
      "c": 7.353164417891053e-15
    },
    "meta": {
      "resolution": {
        "width": 800,
        "height": 600
      }
    }
  },
  "1": {
    "x": { "a": 0.00188, "b": -1.49e-09, "c": -4.56e-15 },
    "y": { "a": 0.00171, "b": -1.89e-09, "c": 7.35e-15 },
    "meta": { "resolution": { "width": 800, "height": 600 } }
  }
}
```

**Updating Calibrations:**
1. All camera calibrations are in one file: `cal.json`
2. Each camera ID (0, 1, 2, etc.) has its own entry
3. Use calibration tool to generate coefficients
4. Update the specific camera entry in `cal.json`
5. No need to manage multiple files

## Pipeline Types

### AprilTag Detection
- Polynomial calibration-based angle calculation
- Adaptive preprocessing with ASLC
- Dynamic frame correction
- Multi-tag simultaneous detection
- 3D position calculation

### Game Piece Detection (YOLO)
- NCNN-optimized YOLOv8 model
- Polynomial calibration for angle/distance
- Aspect ratio filtering
- On-table vs floor classification
- Real-time inference at 256x256

## Network Tables

### AprilTag Tables
- `AprilTagsCam{id}/tx`: Horizontal angles
- `AprilTagsCam{id}/distances`: Distances to tags
- `AprilTagsCam{id}/tags`: Detected tag IDs
- `AprilTagsCam{id}/tl`: Processing latency

### Game Piece Tables
- `GPDCam{id}/tx`: Horizontal angles
- `GPDCam{id}/distances`: Distances to pieces
- `GPDCam{id}/on_tops`: On table flags
- `GPDCam{id}/heights`: Heights above ground
- `GPDCam{id}/tl`: Processing latency


