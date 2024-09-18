# from flask import Flask, Response, jsonify
# from flask_cors import CORS
# import cv2
# import numpy as np
# from deepface import DeepFace
# import datetime
# import threading
# import base64
# from io import BytesIO

# app = Flask(__name__)

# # Initialize CORS
# CORS(app)

# # Global variables for the frame and face detection results
# frame = None
# lock = threading.Lock()
# face_data = []

# # Function to capture frames from the camera
# def capture_frames():
#     global frame
#     cap = cv2.VideoCapture(0)  # Use 0 for the default camera
#     while True:
#         success, new_frame = cap.read()
#         if not success:
#             break
        
#         with lock:
#             frame = new_frame

#     cap.release()

# # Function to detect faces from the camera feed
# def detect_faces():
#     global frame, face_data
#     while True:
#         with lock:
#             current_frame = frame.copy() if frame is not None else None

#         if current_frame is not None:
#             try:
#                 # Save the frame temporarily as an image file
#                 temp_file = "temp_frame.jpg"
#                 cv2.imwrite(temp_file, current_frame)

#                 res = DeepFace.find(img_path=temp_file, db_path="./Database", enforce_detection=False, model_name="Facenet512", detector_backend="retinaface", threshold=0.98)
#                 face_data.clear()
#                 print(type(res))
#                 print(res)

#                 for res_item in res:
#                     if res_item["identity"]:
#                         prediction = str(res_item["identity"][0])
#                         parts = prediction.replace("\\", "/").split('/')
#                         name = parts[-2]
                        
#                         xmin = res_item['source_x'][0]
#                         ymin = res_item['source_y'][0]
#                         w = res_item['source_w'][0]
#                         h = res_item['source_h'][0]
#                         xmax = int(xmin + w)
#                         ymax = int(ymin + h)
                        
#                         face_image = current_frame[int(ymin):int(ymax), int(xmin):int(xmax)]

#                         # Convert face image to base64
#                         _, buffer = cv2.imencode('.jpg', face_image)
#                         face_image_base64 = base64.b64encode(buffer).decode('utf-8')

#                         timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
#                         face_data.append({
#                             "name": name,
#                             "timestamp": timestamp,
#                             "face_image": face_image_base64  # Use base64 string for JSON serialization
#                         })

#             except Exception as e:
#                 print(f"An error occurred: {e}")

# # Route to stream video feed
# @app.route('/video_feed')
# def video_feed():
#     def generate_frames():
#         while True:
#             with lock:
#                 if frame is not None:
#                     ret, buffer = cv2.imencode('.jpg', frame)
#                     if not ret:
#                         continue
#                     frame_bytes = buffer.tobytes()
#                     yield (b'--frame\r\n'
#                            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # Route to get face detection results
# @app.route('/detect_faces', methods=['GET'])
# def get_face_data():
#     return jsonify(face_data)

# if __name__ == '__main__':
#     # Start threads for capturing frames and detecting faces
#     threading.Thread(target=capture_frames, daemon=True).start()
#     threading.Thread(target=detect_faces, daemon=True).start()
#     app.run(host='0.0.0.0', port=5000)


# from flask import Flask, Response, jsonify
# from flask_cors import CORS
# import cv2
# import numpy as np
# from deepface import DeepFace
# import datetime
# import threading
# import base64
# import pandas as pd

# app = Flask(__name__)

# # Initialize CORS
# CORS(app)

# # Global variables for the frame and face detection results
# frame = None
# lock = threading.Lock()
# face_data = []

# # Function to capture frames from the camera
# def capture_frames():
#     global frame
#     cap = cv2.VideoCapture(0)  # Use 0 for the default camera
#     while True:
#         success, new_frame = cap.read()
#         if not success:
#             break
        
#         with lock:
#             frame = new_frame

#     cap.release()

# # Function to detect faces from the camera feed
# def detect_faces():
#     global frame, face_data
#     while True:
#         with lock:
#             current_frame = frame.copy() if frame is not None else None

#         if current_frame is not None:
#             try:
#                 # Save the frame temporarily as an image file
#                 temp_file = "temp_frame.jpg"
#                 cv2.imwrite(temp_file, current_frame)

#                 # Perform face detection
#                 res = DeepFace.find(img_path=temp_file, db_path="./Database", enforce_detection=False, model_name="Facenet512", detector_backend="retinaface", threshold=0.98)
                
#                 face_data.clear()  # Clear previous face data

#                 for df in res:
#                     if isinstance(df, pd.DataFrame):
#                         for _, row in df.iterrows():

#                             if row['identity'] and isinstance(row['identity'], str):
#                                 parts = row['identity'].replace("\\", "/").split('/')  
#                                 if len(parts) > 1: 
#                                     name = parts[-2]  
#                                     print(name)
#                                 else:
#                                     print("Unexpected format for identity:", row['identity'])

#                                 xmin = int(row['source_x'])
#                                 ymin = int(row['source_y'])
#                                 w = int(row['source_w'])
#                                 h = int(row['source_h'])
#                                 xmax = xmin + w
#                                 ymax = ymin + h
                                
#                                 face_image = current_frame[ymin:ymax, xmin:xmax]
#                                 # Convert face image to base64
#                                 _, buffer = cv2.imencode('.jpg', face_image)
#                                 face_image_base64 = base64.b64encode(buffer).decode('utf-8')

#                                 timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                
#                                 face_data.append({
#                                     "name": name,
#                                     "timestamp": timestamp,
#                                     "face_image": face_image_base64  # Use base64 string for JSON serialization
#                                 })

#             except Exception as e:
#                 print(f"An error occurred: {e}")

# # Route to stream video feed
# @app.route('/video_feed')
# def video_feed():
#     def generate_frames():
#         while True:
#             with lock:
#                 if frame is not None:
#                     ret, buffer = cv2.imencode('.jpg', frame)
#                     if not ret:
#                         continue
#                     frame_bytes = buffer.tobytes()
#                     yield (b'--frame\r\n'
#                            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # Route to get face detection results
# @app.route('/detect_faces', methods=['GET'])
# def get_face_data():
#     return jsonify(face_data)

# if __name__ == '__main__':
#     # Start threads for capturing frames and detecting faces
#     threading.Thread(target=capture_frames, daemon=True).start()
#     threading.Thread(target=detect_faces, daemon=True).start()
#     app.run(host='0.0.0.0', port=5000)




from flask import Flask, Response, jsonify, stream_with_context
from flask_cors import CORS
import cv2
import numpy as np
from deepface import DeepFace
import datetime
import threading
import base64
import pandas as pd
import json
import time

app = Flask(__name__)

# Initialize CORS
CORS(app)
#cv2.VideoCapture("rtsp://admin:L20372DC@192.168.1.39:554/cam/realmonitor?channel=1&subtype=0", cv2.CAP_FFMPEG)
# Global variables for the frame and face detection results
frame = None
lock = threading.Lock()

# Function to capture frames from the camera
# def capture_frames():
#     global frame
#     cap = cv2.VideoCapture(0, cv2.CAP_FFMPEG)
#     #rtsp://Beta:beta6337715%40!@192.168.1.2/cam/realmonitor?channel=1&subtype=00&authbasic=QmV0YTpiZXRhNjMzNzcxNSU0MCE=
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
#     while True:
#         success, new_frame = cap.read()
#         if not success:
#             break
        
#         with lock:
#             frame = new_frame

#     cap.release()


def capture_frames():
    global frame
    time.sleep(1)  # Small delay before starting capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    while True:
        success, new_frame = cap.read()
        if not success:
            print("Failed to capture frame")
            break
        
        with lock:
            frame = new_frame
            # print("Captured a frame")
    cap.release()


# Function to detect faces from the camera feed and stream the results
# def stream_face_detection():
#     global frame
#     while True:
#         with lock:
#             current_frame = frame.copy() if frame is not None else None

#         if current_frame is not None:
#             try:
#                 # Save the frame temporarily as an image file
#                 temp_file = "temp_frame.jpg"
#                 cv2.imwrite(temp_file, current_frame)

#                 # Perform face detection
#                 res = DeepFace.find(img_path=temp_file, db_path="./Database", enforce_detection=False, model_name="Facenet512", detector_backend="retinaface", threshold=0.98)
            
#                 face_data = []  # Store face data for the current frame

#                 for df in res:
#                     if isinstance(df, pd.DataFrame) and not df.empty:
#                         # Process only the first row
#                         first_row = df.iloc[0]

#                         if first_row['identity'] and isinstance(first_row['identity'], str):
#                             parts = first_row['identity'].replace("\\", "/").split('/')
#                             if len(parts) > 1:
#                                 name = parts[-2]
#                                 # print(name)
#                             else:
#                                 name = "Unknown"

#                             xmin = int(first_row['source_x'])
#                             ymin = int(first_row['source_y'])
#                             w = int(first_row['source_w'])
#                             h = int(first_row['source_h'])
#                             xmax = xmin + w
#                             ymax = ymin + h

#                             face_image = current_frame[ymin:ymax, xmin:xmax]

#                             # Convert face image to base64
#                             _, buffer = cv2.imencode('.jpg', face_image)
#                             face_image_base64 = base64.b64encode(buffer).decode('utf-8')

#                             timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#                             face_data.append({
#                                 "name": name,
#                                 "timestamp": timestamp,
#                                 "face_image": face_image_base64  # Use base64 string for JSON serialization
#                             })

#                         # Break after processing the first row
#                         break

#                 # Yield the face detection result for the current frame as a JSON string
#                 yield f"data: {json.dumps(face_data)}\n\n"

#             except Exception as e:
#                 yield f"data: {{'error': 'An error occurred: {str(e)}'}}\n\n"
def stream_face_detection():
    global frame
    while True:
        with lock:
            current_frame = frame.copy() if frame is not None else None

        if current_frame is not None:
            try:
                # Save the frame temporarily as an image file
                temp_file = "temp_frame.jpg"
                cv2.imwrite(temp_file, current_frame)

                # Perform face detection
                res = DeepFace.find(img_path=temp_file, db_path="./Database", enforce_detection=False, model_name="Facenet512", detector_backend="retinaface", threshold=0.98)

                face_data = []  # Store face data for the current frame

                for df in res:
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        # Process only the first row
                        first_row = df.iloc[0]

                        if first_row['identity'] and isinstance(first_row['identity'], str):
                            parts = first_row['identity'].replace("\\", "/").split('/')
                            if len(parts) > 1:
                                name = parts[-2]
                            else:
                                name = "Unknown"

                            xmin = int(first_row['source_x'])
                            ymin = int(first_row['source_y'])
                            w = int(first_row['source_w'])
                            h = int(first_row['source_h'])
                            xmax = xmin + w
                            ymax = ymin + h

                            face_image = current_frame[ymin:ymax, xmin:xmax]

                            # Convert face image to base64
                            _, buffer = cv2.imencode('.jpg', face_image)
                            face_image_base64 = base64.b64encode(buffer).decode('utf-8')

                            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                            face_data.append({
                                "name": name,
                                "timestamp": timestamp,
                                "face_image": face_image_base64  # Use base64 string for JSON serialization
                            })

                        # Break after processing the first row
                        break

                # Yield the face detection result for the current frame as a JSON string
                yield f"data: {json.dumps(face_data)}\n\n"

            except Exception as e:
                yield f"data: {{'error': 'An error occurred: {str(e)}'}}\n\n"



# Route to stream video feed
@app.route('/video_feed')
def video_feed():
    def generate_frames():
        while True:
            with lock:
                if frame is not None:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if not ret:
                        continue
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Streaming route for face detection results
@app.route('/detect_faces', methods=['GET'])
def detect_faces_stream():
    return Response(stream_with_context(stream_face_detection()), mimetype="text/event-stream")

if __name__ == '__main__':
    # Start threads for capturing frames
    threading.Thread(target=capture_frames, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)

