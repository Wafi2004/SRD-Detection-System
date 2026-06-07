# Sahsiah Rupa Diri (SRD) Detection System

## Overview
**Sahsiah Rupa Diri (SRD)** is a dress ethics and personal appearance guideline set by Universiti Teknologi MARA (UiTM) for all students. This automated detection system uses computer vision to verify student compliance with UiTM's SRD requirements. It uses YOLO-based deep learning models to detect three key compliance items:
- **Collar** (proper neckline covering)
- **Lanyard** (UiTM ID card on lanyard)
- **Shoes** (appropriate footwear - no barefoot)

## Features
- **Real-time Webcam Detection**: Live stream monitoring with instant compliance feedback
- **Image Upload Processing**: Analyze individual student photos for compliance verification
- **Visual Annotations**: Color-coded bounding boxes and status indicators
- **Batch Processing Support**: Handle multiple images for automated verification

## System Requirements
- Python 3.8+
- Webcam (optional, for real-time mode)
- GPU recommended (NVIDIA CUDA) for faster inference
- At least 2GB RAM

## Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd srddetection
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Start the Web Application
```bash
python app.py
```
Then navigate to `http://localhost:5000` in your browser.

### Upload Image for SRD Compliance Check
1. Open the web interface
2. Click "Choose Image" and select a student photo
3. The system will analyze the image and display:
   - Detected items with bounding boxes (collar, lanyard, shoes)
   - Individual detection results
   - Overall SRD compliance status

### Real-time Webcam Monitoring
1. Open the web interface
2. Click "Start Webcam" or navigate to `/video_feed`
3. System displays live SRD compliance monitoring with real-time feedback

## Project Structure
```
srddetection/
├── app.py              # Flask web application
├── srd_infer.py        # YOLO inference engine
├── requirements.txt    # Python dependencies
├── static/
│   ├── uploads/        # Temporary uploaded images
│   └── results/        # Processed images with annotations
├── templates/
│   └── index.html      # Web interface
├── runs/
│   └── detect/
│       ├── train4/     # Collar model
│       ├── train7/     # Lanyard model
│       └── train8/     # Shoes model
└── README.md           # This file
```

## Configuration

### Detection Parameters (in `srd_infer.py`)
```python
COLLAR_CONF = 0.20      # Collar confidence threshold
LANYARD_CONF = 0.20     # Lanyard confidence threshold
SHOES_CONF = 0.40       # Shoes confidence threshold
IMGSZ = 1280            # Model input image size
```

### Shoe Region Filters
- `SHOE_MIN_Y_RATIO = 0.55`: Shoes detected in lower 45% of image
- `SHOE_MAX_AREA_RATIO = 0.12`: Max 12% of image area
- `SHOE_MAX_ASPECT_RATIO = 3.5`: Max width-to-height ratio

## Compliance Rules
A student is marked **SRD COMPLIANT** when ALL three UiTM dress code requirements are met:
1. ✅ **Collar** - Proper neckline covering (detected)
2. ✅ **Lanyard/ID Card** - UiTM student ID on lanyard (detected)
3. ✅ **Shoes** - Appropriate footwear worn (detected)

If any item is missing: ❌ **NOT COMPLIANT**

## Model Details
- **Purpose**: Detect UiTM SRD (Sahsiah Rupa Diri) compliance elements
- **Models Used**: YOLOv8 (Ultralytics)
- **Framework**: PyTorch
- **Input Resolution**: 1280x1280
- **Inference Framework**: OpenCV

| Model | SRD Item | Detection Target | Confidence |
|-------|----------|------------------|-----------|
| train4 | Collar | Proper neckline covering | 20% |
| train7 | Lanyard | UiTM ID card on lanyard | 20% |
| train8 | Shoes | Footwear / Barefoot detection | 40% |

## Troubleshooting

### Webcam not detected
- Try changing camera index in `gen_frames()`: `cv2.VideoCapture(1)`
- Ensure no other application is using the camera

### Models not loading
- Verify model files exist in `runs/detect/train{4,7,8}/weights/best.pt`
- Check GPU drivers if using CUDA acceleration

### Poor detection accuracy
- Adjust confidence thresholds in `srd_infer.py` to fine-tune detection
- Ensure adequate lighting conditions for image clarity
- Train models with more representative UiTM student photos
- Consider different angles and clothing styles

## File Upload Constraints
- **Supported formats**: JPG, JPEG, PNG, GIF, BMP
- **Max file size**: 50MB
- **Min resolution**: Recommended 640x480+

## Performance
- **GPU (RTX 3060)**: ~50-100 FPS per frame
- **CPU**: ~5-10 FPS per frame
- **Memory**: ~2-3GB during inference

## Limitations
- Works best in well-lit environments
- May struggle with partial/obscured views
- High false-positive rate with similar-looking objects
- Single-person detection per frame

## Future Enhancements
- [ ] Multi-person detection and tracking
- [ ] Database integration for attendance logging
- [ ] Mobile app support
- [ ] Edge deployment optimization
- [ ] Custom model training pipeline
- [ ] Compliance report generation

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss proposed changes.

## Contact
For questions or support, contact: [wafiaffandi2004@gmail.com]

## 📄 Documentation
- [View Project Report](REPORT%20PROJECT%20SPECIAL%20TOPIC%20CSC649.pdf)
  
## Acknowledgments
- YOLOv8 by Ultralytics
- Flask web framework
- OpenCV for computer vision
