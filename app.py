# app.py (FULL MODIFIED - includes upload + realtime webcam stream)

from flask import Flask, render_template, request, url_for, Response
import os
import cv2
from werkzeug.utils import secure_filename

from srd_infer import run_srd, run_srd_frame   # make sure you added run_srd_frame()

app = Flask(__name__)

# ==============================
# FOLDERS
# ==============================
UPLOAD_DIR = os.path.join("static", "uploads")
RESULT_DIR = os.path.join("static", "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ==============================
# HOME PAGE (UPLOAD IMAGE)
# ==============================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    output_url = None

    if request.method == "POST":
        if "image" in request.files:
            image = request.files["image"]
            if image and image.filename != "":
                filename = secure_filename(image.filename)

                # Save uploaded image
                in_path = os.path.join(UPLOAD_DIR, filename)
                image.save(in_path)

                # Output filename
                out_name = f"out_{filename}"
                out_path = os.path.join(RESULT_DIR, out_name)

                # Run SRD detection
                data = run_srd(in_path, out_path)

                result = data["status_text"]
                output_url = url_for("static", filename=f"results/{out_name}")

    return render_template("index.html", result=result, output_url=output_url)


# ==============================
# REALTIME WEBCAM STREAM (MJPEG)
# ==============================
def gen_frames():
    cap = cv2.VideoCapture(0)  # change to 1 if webcam not found

    if not cap.isOpened():
        # stream will just end if no camera
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run real-time SRD overlay on frame
        frame, _ = run_srd_frame(frame)

        # Encode as JPG
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

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
    # IMPORTANT: use_reloader=False so webcam doesn't open twice on Windows
    app.run(debug=True, use_reloader=False)
