import cv2
import os
import datetime
from collections import deque
from threading import Timer
import webbrowser
from flask import Flask, render_template_string, Response, jsonify, send_from_directory
from model import Model
from database import init_db, log_incident, get_all_incidents

app = Flask(__name__)
init_db()

DETECTION_FOLDER = "detections"
if not os.path.exists(DETECTION_FOLDER):
    os.makedirs(DETECTION_FOLDER)

# FILTER: Only specific violence types
VIOLENT_LABELS = ['fight on a street', 'street violence', 'violence in office']

# --- CONFIGURATION FOR "SMART MERGING" ---
BUFFER_SECONDS = 3       # Pre-roll: Save 3 seconds BEFORE violence starts
POST_ROLL_SECONDS = 20   # Post-roll: Keep recording for 20s after violence stops (Merges short bursts)
FPS = 20.0

frame_buffer = deque(maxlen=int(FPS * BUFFER_SECONDS))
last_violence_state = False

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

def gen_frames():
    global last_violence_state
    camera = cv2.VideoCapture(0)
    model = Model()
    
    recording_until = None
    video_writer = None
    current_incident_data = {}

    while True:
        success, frame = camera.read()
        if not success: break

        # Always keep the pre-roll buffer updated
        frame_buffer.append(frame.copy())
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prediction = model.predict(image=rgb_frame)
        
        is_violence = prediction['label'] in VIOLENT_LABELS
        last_violence_state = is_violence
        timestamp = datetime.datetime.now()

        # LOGIC: If violence is detected, extend the recording timer
        if is_violence:
            # This "resets" the countdown, merging intermittent fights into one clip
            recording_until = timestamp + datetime.timedelta(seconds=POST_ROLL_SECONDS)
            
            # Start recording if not already doing so
            if video_writer is None:
                time_str = timestamp.strftime("%Y%m%d_%H%M%S")
                img_name = f"snap_{time_str}.jpg"
                vid_name = f"clip_{time_str}.mp4"  # Changed to .mp4
                img_path = os.path.join(DETECTION_FOLDER, img_name)
                vid_path = os.path.join(DETECTION_FOLDER, vid_name)
                
                # Save the snapshot
                cv2.imwrite(img_path, frame)
                
                # Setup MP4 Video Writer
                # 'avc1' (H.264) is best for HTML5. If it fails, try 'mp4v'
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'avc1') 
                    video_writer = cv2.VideoWriter(vid_path, fourcc, FPS, (frame.shape[1], frame.shape[0]))
                except:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(vid_path, fourcc, FPS, (frame.shape[1], frame.shape[0]))

                # Write the pre-roll (the 3 seconds BEFORE detection)
                for f in frame_buffer:
                    video_writer.write(f)
                
                # Store metadata for DB
                current_incident_data = {
                    "label": prediction['label'],
                    "conf": prediction['confidence'],
                    "img": img_name,
                    "vid": vid_name
                }

        # If we are currently recording...
        if video_writer:
            video_writer.write(frame)
            
            # Stop recording only if violence has stopped AND we passed the cooldown time (20s)
            if not is_violence and timestamp > recording_until:
                video_writer.release()
                video_writer = None
                
                # Save to Database
                log_incident(current_incident_data['label'], current_incident_data['conf'], 
                             current_incident_data['img'], current_incident_data['vid'])

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    with open('index.html', 'r') as f:
        return render_template_string(f.read())

# --- NEW: ROUTE TO SERVE THE ALARM AUDIO FILE ---
@app.route('/warning-alarm.wav')
def serve_audio():
    # Serves the file from the current directory
    return send_from_directory('.', 'warning-alarm.wav')

@app.route('/check_violence')
def check_violence():
    return jsonify({"violence": last_violence_state})

@app.route('/get_history')
def get_history():
    incidents = get_all_incidents()
    return jsonify({"incidents": incidents})

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(DETECTION_FOLDER, filename)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        Timer(1.5, open_browser).start()
    app.run(debug=True, port=5000)