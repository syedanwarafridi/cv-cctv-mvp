import cv2
from roboflow import Roboflow
from flask import Flask, Response, stream_with_context
from flask_cors import CORS
import threading
import base64
import datetime
import json
import time
import pandas as pd
from deepface import DeepFace

app = Flask(__name__)

CORS(app)

frame = None

import logging
logging.basicConfig(level=logging.DEBUG)

###----------> VIDEO FRAME <------------###
def capture_frames():
    global frame
    #cap = cv2.VideoCapture("rtsp://admin:L20372DC@39.37.177.3:554/cam/realmonitor?channel=1&subtype=0")
    cap = cv2.VideoCapture("rtsp://admin:Fbogarage2030@5.246.163.119:554/Streaming/Channels/101")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    while True:
        success, new_frame = cap.read()
        if not success:
            logging.debug("Failed to capture frame")
            break
        frame = new_frame
        logging.debug("Captured a frame")
        time.sleep(0.5)
    cap.release()


@app.route('/video_feed')
def video_feed():
    """Stream the video feed as a series of frames."""
    def generate_frames():
        while True:
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if not ret:
                    print("Failed to encode frame")
                    continue
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

###-----------> Face Detection API <--------------###

def stream_face_detection():
    """Stream face detection results as Server-Sent Events (SSE)."""
    global frame
    while True:
        current_frame = frame.copy() if frame is not None else None
        if current_frame is not None:
            try:
                temp_file = "temp_frame.jpg"
                cv2.imwrite(temp_file, current_frame)

                res = DeepFace.find(img_path=temp_file, db_path="./Database", enforce_detection=False, model_name="Facenet512", detector_backend="retinaface", threshold=0.98)

                face_data = []
                for df in res:
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        first_row = df.iloc[0]

                        if first_row['identity'] and isinstance(first_row['identity'], str):
                            parts = first_row['identity'].replace("\\", "/").split('/')
                            name = parts[-2] if len(parts) > 1 else "Unknown"

                            xmin = int(first_row['source_x'])
                            ymin = int(first_row['source_y'])
                            w = int(first_row['source_w'])
                            h = int(first_row['source_h'])
                            xmax = xmin + w
                            ymax = ymin + h

                            face_image = current_frame[ymin:ymax, xmin:xmax]

                            _, buffer = cv2.imencode('.jpg', face_image)
                            face_image_base64 = base64.b64encode(buffer).decode('utf-8')
                            current_datetime = datetime.datetime.now()
                            date = current_datetime.strftime("%d-%m-%Y")
                            time = current_datetime.strftime("%H:%M:%S") #timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                            face_data.append({
                                "name": name,
                                #"timestamp": timestamp,
                                "face_image": face_image_base64,
                                "time": time,
                                "date": date
                               })

                        break

                yield f"data: {json.dumps(face_data)}\n\n"

            except Exception as e:
                yield f"data: {{'error': 'An error occurred: {str(e)}'}}\n\n"

@app.route('/detect_faces', methods=['GET'])
def detect_faces_stream():
    """Route to stream face detection results as Server-Sent Events (SSE)."""
    return Response(stream_with_context(stream_face_detection()), mimetype="text/event-stream")

###---------------> PPE Detection API <--------------###
rf = Roboflow(api_key="l0fE2hSYgUwQjdtHMGlV")
project = rf.workspace().project("ppe-2p4ai")
model = project.version(1).model

required_ppe = ["Glove", "Helmet", "Safety_Harness", "Goggles", "Boots", "Vest"]
missing_ppe = ["No_Glove", "No_Helmet", "No_Harness", "No_Goggles", "No_Safety-Shoe", "No_Vest"]

def detect_ppe(image):
    result = model.predict(image, confidence=60, overlap=30).json()

    print("Full detection result:", result)

    detections = result["predictions"]

    return detections

def generate():
    cap = cv2.VideoCapture("rtsp://admin:L20372DC@39.37.177.3:554/cam/realmonitor?channel=1&subtype=0") #"emp.mp4") #"rtsp://admin:L20372DC@192.168.1.14:554/cam/realmonitor?channel=1&subtype=0")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        detections = detect_ppe(frame)

        for item in detections:
            xmin, ymin, width, height = int(item["x"]), int(item["y"]), int(item["width"]), int(item["height"])
            class_name = item["class"]
            confidence = item["confidence"]

            cv2.rectangle(frame, (xmin - width // 2, ymin - height // 2), (xmin + width // 2, ymin + height // 2), (0, 255, 0), 2)

            cv2.putText(frame, f"{class_name} ({confidence:.2f})",
                        (xmin - width // 2, ymin - height // 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/ppe_detection')
def ppe_detection():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    threading.Thread(target=capture_frames, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, ssl_context=('cert.pem', 'key.pem'))
