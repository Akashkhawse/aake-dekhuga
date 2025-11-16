# app.py
import os
import datetime
import psutil
import platform
from flask import Flask, render_template, jsonify, request, Response

# Optional imports (camera + wake-word / ai)
try:
    import cv2
    CAMERA_AVAILABLE = True
except Exception:
    CAMERA_AVAILABLE = False

# Optional: Picovoice Porcupine or other wake-word libraries not needed for web UI
# Optional: AI backends (OpenAI / Perplexity)
import requests
from dotenv import load_dotenv
load_dotenv()

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")  # optional
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # optional

app = Flask(__name__)

# ---------------------------------------------------------
# Helper: system uptime, processes, network
# ---------------------------------------------------------
def get_uptime():
    try:
        boot = datetime.datetime.fromtimestamp(psutil.boot_time())
        uptime = datetime.datetime.now() - boot
        # Format: days hh:mm:ss
        days = uptime.days
        hrs, rem = divmod(uptime.seconds, 3600)
        mins, secs = divmod(rem, 60)
        return f"{days}d {hrs:02d}:{mins:02d}:{secs:02d}"
    except Exception:
        return "N/A"

def get_network_usage_mb():
    try:
        net = psutil.net_io_counters()
        sent_mb = round(net.bytes_sent / (1024 * 1024), 2)
        recv_mb = round(net.bytes_recv / (1024 * 1024), 2)
        return sent_mb, recv_mb
    except Exception:
        return 0.0, 0.0

# ---------------------------------------------------------
# Global alert variable (camera/health)
# ---------------------------------------------------------
latest_alert = "✅ No alerts"

# ---------------------------------------------------------
# Home route - Dashboard
# ---------------------------------------------------------
@app.route("/")
def home():
    return render_template("dashboard.html")

# ---------------------------------------------------------
# Health route (auto-refresh + alerts)
# ---------------------------------------------------------
@app.route("/health")
def health():
    cpu = psutil.cpu_percent(interval=0.5)
    memory = psutil.virtual_memory().percent
    disk = psutil.disk_usage('/').percent

    # simple alert logic
    alert = "✅ Normal"
    if cpu > 85:
        alert = f"⚠️ High CPU usage: {cpu}%"
    elif memory > 90:
        alert = f"⚠️ High Memory usage: {memory}%"
    elif disk > 90:
        alert = f"⚠️ Low Disk Space: {disk}% used"

    sent_mb, recv_mb = get_network_usage_mb()

    data = {
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cpu_percent": cpu,
        "memory": memory,
        "disk": disk,
        "os": platform.platform(),
        "uptime": get_uptime(),
        "processes": len(psutil.pids()),
        "net_sent": sent_mb,
        "net_recv": recv_mb,
        "alert": alert
    }
    return jsonify(data)

# ---------------------------------------------------------
# Camera stream generator (optional: disabled if camera not available)
# ---------------------------------------------------------
def gen_empty_frame():
    # 1px black jpg to avoid UI broken image if camera disabled
    import io, base64
    import PIL.Image
    img = PIL.Image.new("RGB", (640, 480), (0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

if CAMERA_AVAILABLE and os.getenv("DISABLE_CAMERA") != "1":
    # attempt to open default camera index (0). If it fails, disable camera feed.
    try:
        camera_index = int(os.getenv("CAMERA_INDEX", "0"))
        camera = cv2.VideoCapture(camera_index)
        # try set resolution (optional)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # verify first frame
        ok, _ = camera.read()
        if not ok:
            CAMERA_AVAILABLE = False
            camera.release()
    except Exception:
        CAMERA_AVAILABLE = False

def generate_frames():
    global latest_alert
    if not CAMERA_AVAILABLE:
        empty = gen_empty_frame()
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + empty + b'\r\n')
    else:
        cam = cv2.VideoCapture(int(os.getenv("CAMERA_INDEX", "0")))
        face_cascade = None
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception:
            face_cascade = None

        while True:
            success, frame = cam.read()
            if not success:
                break

            # face detection (simple)
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if face_cascade is not None:
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    if len(faces) > 0:
                        latest_alert = f"⚠️ Person detected on camera ({len(faces)})"
            except Exception:
                pass

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        cam.release()

@app.route("/camera_feed")
def camera_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------------------------------------------------
# Endpoint to return latest camera alert (polled by UI)
# ---------------------------------------------------------
@app.route("/get_alert")
def get_alert():
    global latest_alert
    return jsonify({"alert": latest_alert})

# ---------------------------------------------------------
# Voice assistant endpoint (simple rule engine / AI proxy)
# ---------------------------------------------------------
def ask_perplexity(prompt):
    # optional: use Perplexity API if key provided
    key = PERPLEXITY_API_KEY
    if not key:
        return "AI backend not configured (PERPLEXITY_API_KEY missing)."

    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    data = {"model": "sonar-small-chat", "messages": [{"role": "user", "content": prompt}]}
    try:
        r = requests.post(url, headers=headers, json=data, timeout=12)
        r.raise_for_status()
        j = r.json()
        # be defensive about keys
        if "choices" in j and len(j["choices"]) > 0:
            return j["choices"][0]["message"].get("content", "No answer.")
        return j.get("answer", "No answer.")
    except Exception as e:
        return f"AI Error: {e}"

@app.route("/assistant", methods=["POST"])
def assistant():
    payload = request.get_json() or {}
    query = payload.get("query", "")
    query = str(query).strip()
    if not query:
        return jsonify({"reply": "Please send a query."})
    # Add small rule-based answers first
    q_lower = query.lower()
    if "time" in q_lower or "समय" in q_lower:
        return jsonify({"reply": f"The time is {datetime.datetime.now().strftime('%H:%M:%S')}"})
    if "cpu" in q_lower:
        return jsonify({"reply": f"CPU usage: {psutil.cpu_percent()}%"})
    # else use AI if configured
    if PERPLEXITY_API_KEY:
        return jsonify({"reply": ask_perplexity(query)})
    elif OPENAI_API_KEY:
        # optional: simple OpenAI call - keep small and safe (user must add OPENAI_API_KEY)
        try:
            import openai
            openai.api_key = OPENAI_API_KEY
            res = openai.ChatCompletion.create(
                model="gpt-4o-mini" if hasattr(openai, "ChatCompletion") else "gpt-3.5-turbo",
                messages=[{"role": "user", "content": query}],
                max_tokens=200
            )
            text = res.choices[0].message['content'] if "message" in res.choices[0] else res.choices[0].text
            return jsonify({"reply": text})
        except Exception as e:
            return jsonify({"reply": f"AI backend error: {e}"})
    else:
        return jsonify({"reply": "No AI backend configured. Add PERPLEXITY_API_KEY or OPENAI_API_KEY to .env."})

# ---------------------------------------------------------
# Dummy device controls (Light, Fan, AC, TV)
# ---------------------------------------------------------
device_state = {"light": "OFF", "fan": "OFF", "ac": "OFF", "tv": "OFF"}

@app.route("/toggle/<device>", methods=["POST"])
def toggle_device(device):
    if device not in device_state:
        return jsonify({"error": "Device not found"}), 404
    device_state[device] = "ON" if device_state[device] == "OFF" else "OFF"
    return jsonify({device: device_state[device]})

# ---------------------------------------------------------
# Start server
# ---------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    host = "0.0.0.0" if os.getenv("HOST_PUBLIC", "0") == "1" else "127.0.0.1"
    print(f"✅ Starting SmartAI Flask Server on {host}:{port} (camera available: {CAMERA_AVAILABLE})")
    # Use Flask debug server for local; for deployment use gunicorn (Procfile included)
    app.run(debug=True, host=host, port=port)