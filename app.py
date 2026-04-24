"""
SRD (Sahsiah Rupa Diri) Detection Flask Application
Detects student compliance with UiTM dress ethics and personal appearance guidelines.
Verifies collar, lanyard, and shoes requirements.
"""

from flask import Flask, render_template, request, url_for, Response
import os
import cv2
import logging
from werkzeug.utils import secure_filename
from srd_infer import run_srd, run_srd_frame

app = Flask(__name__)

# ==============================
# LOGGING
# ==============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# CONFIGURATION
# ==============================
UPLOAD_DIR = os.path.join("static", "uploads")
RESULT_DIR = os.path.join("static", "results")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "bmp"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ==============================
# HOME PAGE (UPLOAD IMAGE)
# ==============================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    output_url = None
    error_msg = None

    if request.method == "POST":
        if "image" not in request.files:
            error_msg = "No image file provided"
        else:
            image = request.files["image"]
            if image.filename == "":
                error_msg = "No file selected"
            elif not allowed_file(image.filename):
                error_msg = f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            else:
                try:
                    filename = secure_filename(image.filename)
                    in_path = os.path.join(UPLOAD_DIR, filename)
                    image.save(in_path)

                    out_name = f"out_{filename}"
                    out_path = os.path.join(RESULT_DIR, out_name)

                    data = run_srd(in_path, out_path)
                    result = data
                    output_url = url_for("static", filename=f"results/{out_name}")
                    logger.info(f"SRD Detection: {result['status_text']} - Collar: {result['has_collar']}, Lanyard: {result['has_lanyard']}, Shoes: {result['has_shoes']}")
                except Exception as e:
                    error_msg = f"Error processing image: {str(e)}"
                    logger.error(error_msg)

    return render_template("index.html", result=result, output_url=output_url, error=error_msg)


# ==============================
# REALTIME WEBCAM STREAM (MJPEG)
# ==============================
def gen_frames():
    """Generator function for webcam frames with SRD detection overlay"""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logger.warning("Webcam not accessible. Attempting camera 1")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            logger.error("No webcam found")
            return

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame, _ = run_srd_frame(frame)

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
    finally:
        cap.release()


@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    logger.info("Starting SRD Detection Application")
    app.run(debug=True, use_reloader=False, threaded=True)
