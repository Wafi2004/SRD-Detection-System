"""
SRD (Sahsiah Rupa Diri) Inference Module
Detects UiTM dress ethics and personal appearance guideline compliance.
Uses pre-trained YOLO models for collar, lanyard, and shoes detection.
"""

from ultralytics import YOLO
import cv2
import os
import logging

logger = logging.getLogger(__name__)

# ==============================
# LOAD MODELS (GLOBAL, ONCE)
# ==============================
try:
    model_collar = YOLO("runs/detect/train4/weights/best.pt")
    model_lanyard = YOLO("runs/detect/train7/weights/best.pt")
    model_shoes = YOLO("runs/detect/train8/weights/best.pt")
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    raise

# ==============================
# LABEL DEFINITIONS
# ==============================
COLLAR_LABEL = "collar"
LANYARD_LABELS = {"ID-CARD", "lanyard", "student_card"}
SHOES_LABELS = {"shoes", "shoe", "sneaker", "Sneaker", "Shoes"}
NO_SHOES_LABELS = {"no_shoes", "no-shoes", "barefoot", "NoShoes"}

# Detection confidence thresholds
COLLAR_CONF = 0.20
LANYARD_CONF = 0.20
SHOES_CONF = 0.40
SHOES_IOU = 0.5
IMGSZ = 640  # Reduced from 1280 for 4x faster inference

# Shoe detection region filters
SHOE_MIN_Y_RATIO = 0.55  # Shoes typically in lower 45% of image
SHOE_MAX_AREA_RATIO = 0.12  # Max 12% of image area
SHOE_MAX_ASPECT_RATIO = 3.5  # Max width-to-height ratio

# Performance optimization settings
MAX_FRAME_WIDTH = 640  # Resize input frames to this width for faster processing
FRAME_SKIP = 0  # Process every frame (0), or skip frames (e.g., 1=process every 2nd frame)
VERBOSE = False  # Disable verbose YOLO output


# =====================================================
# STATIC IMAGE SRD (UPLOAD)
# =====================================================
def run_srd(image_path: str, output_path: str) -> dict:
    """
    Runs SRD detection on an image.
    
    Args:
        image_path: Path to input image
        output_path: Path to save annotated output image
    
    Returns:
        Dictionary with detection results
    """
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Failed to read image from {image_path}")

    annotated, data = run_srd_frame(frame)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, annotated)

    return data


# =====================================================
# REALTIME FRAME SRD (WEBCAM)
# =====================================================
def run_srd_frame(frame) -> tuple:
    """
    Runs SRD detection on a single frame with visual annotations.
    Optimized for real-time performance with frame resizing and parallel inference.
    
    Args:
        frame: OpenCV frame (BGR format)
    
    Returns:
        Tuple of (annotated_frame, detection_results_dict)
    """
    H, W = frame.shape[:2]

    # Resize frame for faster processing while maintaining aspect ratio
    if W > MAX_FRAME_WIDTH:
        scale = MAX_FRAME_WIDTH / W
        resized_frame = cv2.resize(frame, (MAX_FRAME_WIDTH, int(H * scale)), interpolation=cv2.INTER_LINEAR)
        # Store scale factor to map detections back to original frame size
        scale_x = W / MAX_FRAME_WIDTH
        scale_y = H / (H * scale)
    else:
        resized_frame = frame
        scale_x = 1.0
        scale_y = 1.0

    # Run all three models in parallel using list comprehension
    # This processes detections more efficiently than sequential execution
    res_collar = model_collar(resized_frame, conf=COLLAR_CONF, imgsz=IMGSZ, verbose=VERBOSE)
    res_lanyard = model_lanyard(resized_frame, conf=LANYARD_CONF, imgsz=IMGSZ, verbose=VERBOSE)
    res_shoes = model_shoes(resized_frame, conf=SHOES_CONF, imgsz=IMGSZ, iou=SHOES_IOU, verbose=VERBOSE)

    has_collar = False
    has_lanyard = False
    has_shoes = False

    # ---------- COLLAR DETECTION ----------
    for box in res_collar[0].boxes:
        cls = res_collar[0].names[int(box.cls[0])]
        if cls != COLLAR_LABEL:
            continue
        has_collar = True
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Scale coordinates back to original frame if resized
        x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, "collar", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # ---------- LANYARD DETECTION ----------
    for box in res_lanyard[0].boxes:
        cls = res_lanyard[0].names[int(box.cls[0])]
        if cls not in LANYARD_LABELS:
            continue
        has_lanyard = True
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Scale coordinates back to original frame if resized
        x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "lanyard", (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ---------- SHOES DETECTION ----------
    for box in res_shoes[0].boxes:
        cls = res_shoes[0].names[int(box.cls[0])]
        if cls not in SHOES_LABELS and cls not in NO_SHOES_LABELS:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Scale coordinates back to original frame if resized
        x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
        bw, bh = x2 - x1, y2 - y1
        area = bw * bh

        # Apply region filters to reduce false positives
        if y2 < int(H * SHOE_MIN_Y_RATIO):
            continue
        if area > (W * H) * SHOE_MAX_AREA_RATIO:
            continue
        if bw / max(bh, 1) > SHOE_MAX_ASPECT_RATIO:
            continue

        if cls in SHOES_LABELS:
            has_shoes = True
            color = (0, 200, 0)
            label = "shoes"
        else:
            color = (0, 165, 255)
            label = "NO SHOES"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y2 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ---------- COMPLIANCE STATUS ----------
    compliant = has_collar and has_lanyard and has_shoes
    status_text = "STUDENT COMPLIANT" if compliant else "NOT COMPLIANT"
    status_color = (0, 255, 0) if compliant else (0, 0, 255)

    cv2.rectangle(frame, (20, 15), (560, 85), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (30, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, status_color, 3)

    return frame, {
        "has_collar": has_collar,
        "has_lanyard": has_lanyard,
        "has_shoes": has_shoes,
        "compliant": compliant,
        "status_text": status_text
    }
