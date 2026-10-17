from flask import (
    Flask, render_template_string, request,
    redirect, url_for, flash, make_response, jsonify
)
import sqlite3
import csv
import io
import os
import base64
import hashlib
from datetime import datetime, date

import cv2
import numpy as np

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-this")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "attendance.db")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "faces")
os.makedirs(UPLOAD_DIR, exist_ok=True)

CLASS_START = "08:30"

# LBPH 距離越低通常代表越相似。
# 這是初始測試值，必須依實際照片調整，不能保證完全正確。
FACE_THRESHOLD = 65.0

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

recognizer = None
label_to_student = {}
model_signature = None


PAGE = """
<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>智慧人臉辨識出勤管理系統</title>
<style>
body{font-family:Arial,"Microsoft JhengHei",sans-serif;
background:#f3f5f9;margin:0;color:#202938}
header{background:#172554;color:white;padding:24px}
main{max-width:1100px;margin:24px auto;padding:0 16px}
.card{background:white;border-radius:12px;padding:20px;
margin-bottom:18px;box-shadow:0 2px 10px #0000000c}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}
input,button{padding:10px;margin:5px 0;max-width:100%;box-sizing:border-box}
input{border:1px solid #ccd2dc;border-radius:6px}
button,.btn{background:#1d4ed8;color:white;border:0;
border-radius:6px;padding:10px 14px;cursor:pointer;text-decoration:none}
button:disabled{opacity:.6;cursor:not-allowed}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:10px;border-bottom:1px solid #e5e7eb;text-align:left}
th{background:#f8fafc}
.table-wrap{overflow-x:auto}
.muted{color:#64748b;font-size:14px}
.msg{background:#e0f2fe;padding:10px;border-radius:6px;margin:8px 0}
.camera-wrap{position:relative;width:100%;max-width:720px;background:#111;
border-radius:8px;overflow:hidden}
.camera-wrap video{display:block;width:100%;height:auto}
.camera-wrap canvas{position:absolute;inset:0;width:100%;height:100%;
pointer-events:none}
#cameraStatus{margin-top:10px;white-space:pre-wrap}
</style>
</head>
<body>
<header>
<h1>智慧人臉辨識出勤管理系統</h1>
<div>學生建檔・即時人臉辨識・自動簽到・出勤報表</div>
</header>
<main>

{% with messages = get_flashed_messages() %}
{% for message in messages %}<div class="msg">{{ message }}</div>{% endfor %}
{% endwith %}

<div class="grid">
<div class="card"><div class="muted">學生總數</div><h2>{{ students|length }}</h2></div>
<div class="card"><div class="muted">今日簽到人數</div><h2>{{ today_count }}</h2></div>
<div class="card"><div class="muted">今日遲到人數</div><h2>{{ late_count }}</h2></div>
</div>

<div class="card">
<h2>一、即時人臉辨識與自動簽到</h2>
<p class="muted">
請先建立學生資料並上傳清楚、正面且經本人同意的照片。
按下開啟攝影機後，系統會嘗試比對學生照片，辨識成功時自動簽到。
辨識可能出錯，請確認結果並保留人工核對。
</p>
<button id="startCamera" type="button">開啟攝影機</button>
<button id="stopCamera" type="button" disabled>關閉攝影機</button>
<div class="camera-wrap">
<video id="video" autoplay playsinline muted></video>
<canvas id="overlay"></canvas>
</div>
<div id="cameraStatus" class="msg">攝影機尚未開啟。</div>
</div>

<div class="card">
<h2>二、學生註冊與建檔</h2>
<form method="post" action="{{ url_for('add_student') }}" enctype="multipart/form-data">
<label>學號</label><br>
<input name="student_id" required maxlength="50"><br>
<label>姓名</label><br>
<input name="name" required maxlength="100"><br>
<label>人臉照片（須取得本人同意）</label><br>
<input type="file" name="photo" accept="image/jpeg,image/png,image/webp"><br>
<button type="submit">新增學生</button>
</form>
<p class="muted">建議上傳單人、正面、清晰、光線充足的照片。</p>
</div>

<div class="card">
<h2>三、學生名單</h2>
<div class="table-wrap"><table>
<tr><th>學號</th><th>姓名</th><th>照片</th><th>操作</th></tr>
{% for s in students %}
<tr>
<td>{{ s['student_id'] }}</td><td>{{ s['name'] }}</td>
<td>{% if s['photo'] %}
<a href="{{ url_for('static', filename='faces/' + s['photo']) }}"
target="_blank" rel="noopener">查看照片</a>
{% else %}未上傳{% endif %}</td>
<td>
<form method="post" action="{{ url_for('delete_student', student_id=s['student_id']) }}" style="display:inline" onsubmit="return confirm('確定要刪除「{{ s["name"] }}（{{ s["student_id"] }}）」嗎？\n刪除後，該學生的出勤紀錄與人臉照片也會一併刪除。');">
<button type="submit" style="background:#dc2626">刪除</button>
</form>
</td>
</tr>
{% else %}
<tr><td colspan="4">目前沒有學生資料</td></tr>
{% endfor %}
</table></div>
</div>

<div class="card">
<h2>四、手動簽到</h2>
<p class="muted">供人工補登使用。每位學生每天只能簽到一次。</p>
<form method="post" action="{{ url_for('check_in') }}">
<label>學生學號</label><br>
<input name="student_id" required><br>
<button type="submit">登記簽到</button>
</form>
</div>

<div class="card">
<h2>五、出勤紀錄</h2>
<form method="get" action="/">
<label>日期篩選</label>
<input type="date" name="day" value="{{ selected_day }}">
<button type="submit">查詢</button>
<a class="btn" href="{{ url_for('export_csv', day=selected_day) }}">匯出 CSV</a>
</form>
<div class="table-wrap"><table>
<tr><th>日期</th><th>學號</th><th>姓名</th><th>簽到時間</th><th>狀態</th></tr>
{% for r in records %}
<tr>
<td>{{ r['day'] }}</td><td>{{ r['student_id'] }}</td>
<td>{{ r['name'] }}</td><td>{{ r['check_in_time'] }}</td>
<td>{{ r['status'] }}</td>
</tr>
{% else %}
<tr><td colspan="5">此日期沒有簽到紀錄</td></tr>
{% endfor %}
</table></div>
</div>
</main>

<script>
const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const statusBox = document.getElementById("cameraStatus");
const startButton = document.getElementById("startCamera");
const stopButton = document.getElementById("stopCamera");

let stream = null;
let timer = null;
let busy = false;
let lastMessages = {};

async function recognizeFrame() {
    if (!stream || busy || video.readyState < 2) return;

    busy = true;
    try {
        const capture = document.createElement("canvas");
        capture.width = video.videoWidth;
        capture.height = video.videoHeight;

        if (!capture.width || !capture.height) return;

        capture.getContext("2d").drawImage(
            video, 0, 0, capture.width, capture.height
        );

        const response = await fetch("/recognize_frame", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                image: capture.toDataURL("image/jpeg", 0.75)
            })
        });

        const data = await response.json();

        if (!response.ok || !data.ok) {
            statusBox.textContent = data.message || "辨識請求失敗。";
            return;
        }

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const messages = [];

        for (const face of data.faces) {
            const recognized = face.student_id !== null;
            ctx.lineWidth = 3;
            ctx.font = "18px Arial, sans-serif";
            ctx.strokeStyle = recognized ? "#22c55e" : "#f59e0b";
            ctx.fillStyle = recognized ? "#166534" : "#92400e";

            ctx.strokeRect(face.x, face.y, face.w, face.h);

            const label = recognized
                ? face.name + " (" + face.student_id + ")"
                : "未辨識";

            const textY = Math.max(22, face.y - 6);
            ctx.fillText(label, face.x, textY);

            if (recognized) {
                const msg = face.attendance_message;
                if (lastMessages[face.student_id] !== msg) {
                    messages.push(face.name + "：" + msg);
                    lastMessages[face.student_id] = msg;
                }
            }
        }

        if (messages.length) {
            statusBox.textContent = messages.join("\\n");
        } else if (data.faces.length === 0) {
            statusBox.textContent = "尚未偵測到人臉，請面向鏡頭。";
        } else if (data.faces.every(f => f.student_id === null)) {
            statusBox.textContent = "已偵測到人臉，但尚未比對成功。";
        } else if (!statusBox.textContent) {
            statusBox.textContent = "正在辨識中……";
        }
    } catch (error) {
        statusBox.textContent = "連線或辨識發生問題，請確認 Flask 正在執行。";
    } finally {
        busy = false;
    }
}

startButton.addEventListener("click", async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {width: {ideal: 640}, height: {ideal: 480}},
            audio: false
        });

        video.srcObject = stream;
        await video.play();

        startButton.disabled = true;
        stopButton.disabled = false;
        statusBox.textContent = "攝影機已開啟，正在辨識……";

        timer = setInterval(recognizeFrame, 1000);
    } catch (error) {
        statusBox.textContent =
            "無法開啟攝影機。請允許瀏覽器使用攝影機，並確認沒有其他程式占用鏡頭。";
    }
});

stopButton.addEventListener("click", () => {
    if (timer) {
        clearInterval(timer);
        timer = null;
    }
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    video.srcObject = null;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    startButton.disabled = false;
    stopButton.disabled = true;
    statusBox.textContent = "攝影機已關閉。";
});

window.addEventListener("beforeunload", () => {
    if (stream) stream.getTracks().forEach(track => track.stop());
});
</script>
</body>
</html>
"""


def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                photo TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                day TEXT NOT NULL,
                check_in_time TEXT NOT NULL,
                status TEXT NOT NULL,
                UNIQUE(student_id, day),
                FOREIGN KEY(student_id) REFERENCES students(student_id)
            )
        """)


def get_face_crop(image):
    """取得圖片中最大的人臉，回傳灰階裁切影像；找不到則回傳 None。"""
    if image is None or image.size == 0:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    crop = gray[y:y + h, x:x + w]

    if crop.size == 0:
        return None

    return cv2.resize(crop, (200, 200))


def get_model():
    """有學生照片變動時重新訓練，否則重用已建立的模型。"""
    global recognizer, label_to_student, model_signature

    with get_db() as conn:
        students = conn.execute("""
            SELECT student_id, name, photo
            FROM students
            WHERE photo IS NOT NULL AND photo != ''
            ORDER BY student_id
        """).fetchall()

    signature_items = []
    for student in students:
        path = os.path.join(UPLOAD_DIR, student["photo"])
        if os.path.isfile(path):
            signature_items.append((
                student["student_id"],
                student["name"],
                student["photo"],
                os.path.getmtime(path)
            ))

    signature = tuple(signature_items)

    if recognizer is not None and signature == model_signature:
        return recognizer, label_to_student

    faces_for_training = []
    labels = []
    new_label_to_student = {}
    next_label = 0

    for student in students:
        path = os.path.join(UPLOAD_DIR, student["photo"])
        if not os.path.isfile(path):
            continue

        image = cv2.imread(path)
        face = get_face_crop(image)

        if face is None:
            continue

        label = next_label
        next_label += 1

        faces_for_training.append(face)
        labels.append(label)
        new_label_to_student[label] = {
            "student_id": student["student_id"],
            "name": student["name"]
        }

    if not faces_for_training:
        recognizer = None
        label_to_student = {}
        model_signature = signature
        return None, {}

    new_recognizer = cv2.face.LBPHFaceRecognizer_create()
    new_recognizer.train(faces_for_training, np.array(labels, dtype=np.int32))

    recognizer = new_recognizer
    label_to_student = new_label_to_student
    model_signature = signature

    return recognizer, label_to_student


def record_attendance(student_id):
    """自動簽到；同一位學生同一天只記錄一次。"""
    now = datetime.now()
    day = now.date().isoformat()
    check_in_time = now.strftime("%H:%M:%S")
    status = "遲到" if check_in_time > CLASS_START + ":00" else "準時"

    with get_db() as conn:
        existing = conn.execute("""
            SELECT check_in_time, status
            FROM attendance
            WHERE student_id = ? AND day = ?
        """, (student_id, day)).fetchone()

        if existing:
            return f"今日已簽到（{existing['check_in_time']}）"

        try:
            conn.execute("""
                INSERT INTO attendance
                    (student_id, day, check_in_time, status)
                VALUES (?, ?, ?, ?)
            """, (student_id, day, check_in_time, status))
            return f"簽到成功 {check_in_time}（{status}）"
        except sqlite3.IntegrityError:
            return "今日已簽到"


@app.route("/")
def index():
    selected_day = request.args.get("day", date.today().isoformat())

    with get_db() as conn:
        students = conn.execute(
            "SELECT * FROM students ORDER BY student_id"
        ).fetchall()

        records = conn.execute("""
            SELECT a.*, s.name
            FROM attendance a
            JOIN students s ON s.student_id = a.student_id
            WHERE a.day = ?
            ORDER BY a.check_in_time DESC
        """, (selected_day,)).fetchall()

        today_count = conn.execute(
            "SELECT COUNT(*) FROM attendance WHERE day = ?",
            (date.today().isoformat(),)
        ).fetchone()[0]

        late_count = conn.execute("""
            SELECT COUNT(*) FROM attendance
            WHERE day = ? AND status = '遲到'
        """, (date.today().isoformat(),)).fetchone()[0]

    return render_template_string(
        PAGE,
        students=students,
        records=records,
        today_count=today_count,
        late_count=late_count,
        selected_day=selected_day
    )


@app.route("/add_student", methods=["POST"])
def add_student():
    student_id = request.form.get("student_id", "").strip()
    name = request.form.get("name", "").strip()
    photo = request.files.get("photo")
    filename = None

    if not student_id or not name:
        flash("請輸入學號和姓名。")
        return redirect(url_for("index"))

    if photo and photo.filename:
        raw = photo.read()
        image = cv2.imdecode(
            np.frombuffer(raw, np.uint8),
            cv2.IMREAD_COLOR
        )

        if image is None:
            flash("照片格式無法辨識，請上傳有效圖片。")
            return redirect(url_for("index"))

        if get_face_crop(image) is None:
            flash("照片中未偵測到清楚的人臉，請改用正面、清晰的單人照片。")
            return redirect(url_for("index"))

        filename = datetime.now().strftime("%Y%m%d%H%M%S%f") + ".jpg"
        saved_path = os.path.join(UPLOAD_DIR, filename)

        if not cv2.imwrite(saved_path, image):
            flash("照片儲存失敗。")
            return redirect(url_for("index"))

    try:
        with get_db() as conn:
            conn.execute("""
                INSERT INTO students (student_id, name, photo)
                VALUES (?, ?, ?)
            """, (student_id, name, filename))

        flash("學生建檔成功。")
    except sqlite3.IntegrityError:
        if filename:
            path = os.path.join(UPLOAD_DIR, filename)
            if os.path.exists(path):
                os.remove(path)
        flash("此學號已存在，請勿重複註冊。")

    return redirect(url_for("index"))


@app.route("/delete_student/<student_id>", methods=["POST"])
def delete_student(student_id):
    global recognizer, label_to_student, model_signature

    with get_db() as conn:
        student = conn.execute(
            "SELECT student_id, name, photo FROM students WHERE student_id = ?",
            (student_id,)
        ).fetchone()

        if not student:
            flash("找不到要刪除的學生。")
            return redirect(url_for("index"))

        # 先刪除出勤紀錄，避免外鍵限制，也確保不留下孤立資料。
        conn.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
        conn.execute("DELETE FROM students WHERE student_id = ?", (student_id,))

    if student["photo"]:
        photo_path = os.path.join(UPLOAD_DIR, student["photo"])
        if os.path.isfile(photo_path):
            try:
                os.remove(photo_path)
            except OSError:
                # 資料庫刪除已成功；照片清理失敗不應讓整個操作報錯。
                pass

    # 強制下次辨識時重新訓練模型，避免已刪學生仍留在記憶體。
    recognizer = None
    label_to_student = {}
    model_signature = None

    flash(f"已刪除學生：{student['name']}（{student['student_id']}）。")
    return redirect(url_for("index"))


@app.route("/check_in", methods=["POST"])
def check_in():
    student_id = request.form.get("student_id", "").strip()

    with get_db() as conn:
        student = conn.execute(
            "SELECT student_id FROM students WHERE student_id = ?",
            (student_id,)
        ).fetchone()

    if not student:
        flash("查無此學號，請先建立學生資料。")
        return redirect(url_for("index"))

    message = record_attendance(student_id)
    flash(f"{student_id}：{message}")
    return redirect(url_for("index"))


@app.route("/recognize_frame", methods=["POST"])
def recognize_frame():
    data = request.get_json(silent=True) or {}
    image_data = data.get("image", "")

    if not isinstance(image_data, str) or "," not in image_data:
        return jsonify(ok=False, message="影像資料格式不正確。"), 400

    try:
        encoded = image_data.split(",", 1)[1]
        raw = base64.b64decode(encoded, validate=True)
        image = cv2.imdecode(
            np.frombuffer(raw, np.uint8),
            cv2.IMREAD_COLOR
        )
    except Exception:
        return jsonify(ok=False, message="無法讀取攝影機影像。"), 400

    if image is None:
        return jsonify(ok=False, message="影像解碼失敗。"), 400

    model, student_map = get_model()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    detected = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    results = []

    for (x, y, w, h) in detected:
        result = {
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "student_id": None,
            "name": None,
            "attendance_message": None
        }

        if model is not None:
            face = gray[y:y + h, x:x + w]

            if face.size:
                face = cv2.resize(face, (200, 200))
                label, distance = model.predict(face)
                print(f"辨識診斷：label={label}, distance={distance:.2f}, threshold={FACE_THRESHOLD}")

                if distance <= FACE_THRESHOLD and label in student_map:
                    student = student_map[label]
                    result["student_id"] = student["student_id"]
                    result["name"] = student["name"]
                    result["attendance_message"] = record_attendance(student["student_id"])

        results.append(result)

    if model is None:
        message = "尚無可用的人臉照片，請先註冊學生並上傳正面照片。"
    else:
        message = "辨識完成。"

    return jsonify(ok=True, message=message, faces=results)


@app.route("/export.csv")
def export_csv():
    selected_day = request.args.get("day", date.today().isoformat())

    with get_db() as conn:
        rows = conn.execute("""
            SELECT a.day, a.student_id, s.name, a.check_in_time, a.status
            FROM attendance a
            JOIN students s ON s.student_id = a.student_id
            WHERE a.day = ?
            ORDER BY a.check_in_time
        """, (selected_day,)).fetchall()

    output = io.StringIO()
    output.write("\\ufeff")
    writer = csv.writer(output)
    writer.writerow(["日期", "學號", "姓名", "簽到時間", "出勤狀態"])

    for row in rows:
        writer.writerow([
            row["day"], row["student_id"], row["name"],
            row["check_in_time"], row["status"]
        ])

    response = make_response(output.getvalue())
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = (
        f"attachment; filename=attendance_{selected_day}.csv"
    )
    return response


if __name__ == "__main__":
    init_db()
    app.run(debug=True)