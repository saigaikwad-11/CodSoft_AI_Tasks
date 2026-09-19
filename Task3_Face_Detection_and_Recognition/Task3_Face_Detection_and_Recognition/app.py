import cv2
import threading
from flask import Flask, Response, jsonify, render_template

from face_engine import FaceEngine


app = Flask(__name__)

engine = FaceEngine()

camera = cv2.VideoCapture(0)

camera_lock = threading.Lock()


def generate_frames():
    while True:
        with camera_lock:
            success, frame = camera.read()

        if not success:
            break

        frame = engine.process_frame(frame)

        success, buffer = cv2.imencode(".jpg", frame)

        if not success:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/status")
def status():
    return jsonify(engine.get_status())


@app.route("/api/reload")
def reload_faces():
    engine.load_known_faces()

    return jsonify({
        "success": True,
        "message": "Known faces reloaded successfully."
    })


if __name__ == "__main__":
    try:
        app.run(
            host="127.0.0.1",
            port=5000,
            debug=False,
            threaded=True
        )
    finally:
        camera.release()