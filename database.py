import sqlite3
import datetime

def init_db():
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS incidents
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  label TEXT,
                  confidence REAL,
                  image_path TEXT,
                  video_path TEXT)''')
    conn.commit()
    conn.close()

def log_incident(label, confidence, image_path, video_path):
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO incidents (timestamp, label, confidence, image_path, video_path) VALUES (?, ?, ?, ?, ?)",
              (timestamp, label, confidence, image_path, video_path))
    conn.commit()
    conn.close()

def get_all_incidents():
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute("SELECT * FROM incidents ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return rows