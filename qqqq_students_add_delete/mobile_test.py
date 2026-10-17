
from flask import Flask, request, render_template_string
import cv2
import numpy as np

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

HTML = """
<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>手機人臉打卡測試</title>
<style>
body {
    background: #f5f5f7;
    font-family: Arial, sans-serif;
    text-align: center;
    padding: 35px 18px;
}
.card {
    background: white;
    padding: 28px 20px;
    border-radius: 22px;
    max-width: 420px;
    margin: auto;
}
h2 { color: #1d1d1f; }
button {
    background: #007aff;
    color: white;
    padding: 15px;
    border: none;
    border-radius: 12px;
    width: 100%;
    font-size: 17px;
}
input { margin: 22px 0; max-width: 100%; }
.result { font-size: 20px; margin-top: 20px; }
</style>
</head>
<body>
<div class="card">
<h2>人臉打卡系統</h2>
<p>Android / iPhone 測試版</p>
<form method="POST" enctype="multipart/form-data">
<input type="file" name="photo"
       accept="image/*" capture="user" required>
<button type="submit">開始人臉偵測</button>
</form>
<div class="result">{{ result }}</div>
</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""

    if request.method == "POST":
        file = request.files.get("photo")

        if not file or not file.filename:
            result = "請先拍攝照片"
        else:
            data = file.read()

            if len(data) > 10 * 1024 * 1024:
                result = "照片不可超過 10MB"
            else:
                array = np.frombuffer(data, np.uint8)
                image = cv2.imdecode(array, cv2.IMREAD_COLOR)

                if image is None:
                    result = "無法讀取照片"
                else:
                    gray = cv2.cvtColor(
                        image, cv2.COLOR_BGR2GRAY
                    )
                    faces = face_cascade.detectMultiScale(
                        gray, 1.1, 5, minSize=(50, 50)
                    )
                    result = f"偵測到 {len(faces)} 張人臉"

    return render_template_string(HTML, result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
