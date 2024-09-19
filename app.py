from flask import Flask, Response, stream_with_context
from flask_cors import CORS
import cv2
import threading
import base64
import datetime
import json
import time
import pandas as pd
from deepface import DeepFace

app = Flask(__name__)

# Initialize CORS
CORS(app)

# Global variables for the frame and face detection results
frame = None

def capture_frames():
    """Capture frames from the video stream and update the global `frame` variable."""
    global frame
    cap = cv2.VideoCapture("emp.mp4")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    while True:
        success, new_frame = cap.read()
        if not success:
            print("Failed to capture frame")
            break
        frame = new_frame
        time.sleep(0.1)  # Small delay to prevent too fast capturing
    cap.release()

@app.route('/video_feed')
def video_feed():
    """Stream the video feed as a series of frames."""
    def generate_frames():
        while True:
            if frame is not None:
                # Encode the frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    print("Failed to encode frame")
                    continue
                frame_bytes = buffer.tobytes()
                # Yield the frame in the proper format for streaming
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)  # Small delay to control frame rate
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def stream_face_detection():
    """Stream face detection results as Server-Sent Events (SSE)."""
    global frame
    while True:
        current_frame = frame.copy() if frame is not None else None
        if current_frame is not None:
            try:
                # Save the current frame temporarily as an image
                temp_file = "temp_frame.jpg"
                cv2.imwrite(temp_file, current_frame)

                # Perform face detection
                res = DeepFace.find(img_path=temp_file, db_path="./Database", enforce_detection=False, model_name="Facenet512", detector_backend="retinaface", threshold=0.98)

                face_data = []  # Store face data for the current frame

                for df in res:
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        # Process the first row of the detection result
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

                            # Crop the face from the current frame
                            face_image = current_frame[ymin:ymax, xmin:xmax]

                            # Convert the cropped face image to base64
                            _, buffer = cv2.imencode('.jpg', face_image)
                            face_image_base64 = base64.b64encode(buffer).decode('utf-8')
                            #pakistan_timezone = pytz.timezone('Asia/Karachi')
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

                        # Break after processing the first row to simplify the output
                        break

                # Yield the face detection result as a JSON string
                yield f"data: {json.dumps(face_data)}\n\n"

            except Exception as e:
                yield f"data: {{'error': 'An error occurred: {str(e)}'}}\n\n"

@app.route('/detect_faces', methods=['GET'])
def detect_faces_stream():
    """Route to stream face detection results as Server-Sent Events (SSE)."""
    return Response(stream_with_context(stream_face_detection()), mimetype="text/event-stream")

if __name__ == '__main__':
    # Start a thread to capture frames continuously
    threading.Thread(target=capture_frames, daemon=True).start()
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, ssl_context=('cert.pem', 'key.pem'))
