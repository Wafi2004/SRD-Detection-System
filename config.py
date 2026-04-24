"""
Configuration file for SRD Detection System
SRD = Sahsiah Rupa Diri (Universiti Teknologi MARA dress ethics guideline)

Centralize all configuration parameters here for easier management
"""

import os

# ==============================
# FLASK CONFIGURATION
# ==============================
DEBUG = True
THREADED = True
PORT = 5000

# ==============================
# FILE UPLOAD CONFIGURATION
# ==============================
UPLOAD_DIR = os.path.join("static", "uploads")
RESULT_DIR = os.path.join("static", "results")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "bmp"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# ==============================
# MODEL PATHS
# ==============================
MODEL_COLLAR_PATH = "runs/detect/train4/weights/best.pt"
MODEL_LANYARD_PATH = "runs/detect/train7/weights/best.pt"
MODEL_SHOES_PATH = "runs/detect/train8/weights/best.pt"

# ==============================
# DETECTION PARAMETERS
# ==============================
# Confidence thresholds
COLLAR_CONF = 0.20
LANYARD_CONF = 0.20
SHOES_CONF = 0.40
SHOES_IOU = 0.5
IMGSZ = 1280

# ==============================
# LABEL DEFINITIONS
# ==============================
COLLAR_LABEL = "collar"
LANYARD_LABELS = {"ID-CARD", "lanyard", "student_card"}
SHOES_LABELS = {"shoes", "shoe", "sneaker", "Sneaker", "Shoes"}
NO_SHOES_LABELS = {"no_shoes", "no-shoes", "barefoot", "NoShoes"}

# ==============================
# SHOE REGION FILTERS
# ==============================
SHOE_MIN_Y_RATIO = 0.55          # Shoes typically in lower 45% of image
SHOE_MAX_AREA_RATIO = 0.12       # Max 12% of image area
SHOE_MAX_ASPECT_RATIO = 3.5      # Max width-to-height ratio

# ==============================
# COLOR SCHEME (BGR format for OpenCV)
# ==============================
COLOR_COLLAR = (255, 0, 0)         # Blue
COLOR_LANYARD = (0, 255, 0)        # Green
COLOR_SHOES = (0, 200, 0)          # Dark Green
COLOR_NO_SHOES = (0, 165, 255)     # Orange
COLOR_COMPLIANT = (0, 255, 0)      # Green
COLOR_NOT_COMPLIANT = (0, 0, 255)  # Red
COLOR_TEXT_BG = (0, 0, 0)          # Black

# ==============================
# LOGGING
# ==============================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ==============================
# COMPLIANCE RULES
# ==============================
REQUIRED_ITEMS = {
    "collar": "Collar detected",
    "lanyard": "Lanyard/ID-card detected",
    "shoes": "Shoes detected"
}

# ==============================
# WEBCAM CONFIGURATION
# ==============================
WEBCAM_INDEX_PRIMARY = 0
WEBCAM_INDEX_SECONDARY = 1
WEBCAM_FPS = 30
WEBCAM_RESOLUTION = (1280, 720)
