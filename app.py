import requests
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, jsonify

app = Flask(__name__)

# =========================
# LOAD TFLITE MODEL
# =========================
interpreter = tf.lite.Interpreter(model_path="water_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = ["Drinking", "Irrigation", "Washing", "Unsafe"]


# =========================
# ML PREDICTION
# =========================
def predict(ph, temp, conductivity):
    try:
        input_data = np.array([[ph, temp, conductivity]], dtype=np.float32)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])

        probabilities = output[0].tolist()
        predicted_index = int(np.argmax(output))

        return labels[predicted_index], probabilities, predicted_index

    except Exception as e:
        print("Prediction Error:", e)
        return "Error", [0, 0, 0, 0], -1


# =========================
# SAFE FLOAT CONVERTER
# =========================
def safe_float(value):
    try:
        if value is None:
            return 0.0
        value = str(value).strip()
        if value.lower() in ["", "null", "nan"]:
            return 0.0
        return float(value)
    except:
        return 0.0


# =========================
# FETCH THINGSPEAK DATA
# =========================
def get_data():
    CHANNEL_ID = "3342088"
    READ_API_KEY = "A7M99B6I0LUM4T6M"

    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results=20"

    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        feeds = data.get("feeds", [])

        ph, temp, tds, turbidity, time = [], [], [], [], []

        for f in feeds:
            if not isinstance(f, dict):
                continue

            ph.append(safe_float(f.get("field1")))
            temp.append(safe_float(f.get("field2")))        # TEMP FIXED
            tds.append(safe_float(f.get("field3")))
            turbidity.append(safe_float(f.get("field4")))
            time.append(f.get("created_at", ""))

        return ph, temp, tds, turbidity, time

    except Exception as e:
        print("Data error:", e)
        return [0], [0], [0], [0], [""]


# =========================
# SAFE LAST VALUE
# =========================
def safe_last(arr):
    return arr[-1] if arr and len(arr) > 0 else 0


# =========================
# STATUS CHECK
# =========================
def get_status(ph, tds, temp, turbidity):
    issues = []
    if ph < 6.5 or ph > 8.5:
        issues.append("pH out of range")
    if tds > 500:
        issues.append("TDS too high")
    if temp > 35:
        issues.append("Temperature too high")
    if turbidity > 5:
        issues.append("High turbidity")
    return issues


# =========================
# HOME
# =========================
@app.route("/")
def home():
    ph, temp, tds, turbidity, time = get_data()

    prediction, _, _ = predict(
        safe_last(ph),
        safe_last(temp),
        safe_last(tds)
    )

    alert = "Unsafe Water Detected" if prediction == "Unsafe" else "System Normal"

    return render_template(
        "index.html",
        alert=alert,
        ph=safe_last(ph),
        temp=safe_last(temp),
        tds=safe_last(tds),
        turbidity=safe_last(turbidity),
        prediction=prediction
    )


# =========================
# LIVE PAGE
# =========================
@app.route("/live")
def live():
    ph, temp, tds, turbidity, time = get_data()

    prediction, _, _ = predict(
        safe_last(ph),
        safe_last(temp),
        safe_last(tds)
    )

    alert = "Unsafe Water Detected" if prediction == "Unsafe" else "System Normal"

    issues = get_status(
        safe_last(ph),
        safe_last(tds),
        safe_last(temp),
        safe_last(turbidity)
    )

    return render_template(
        "live.html",
        alert=alert,
        ph=safe_last(ph),
        temp=safe_last(temp),
        tds=safe_last(tds),
        turbidity=safe_last(turbidity),
        prediction=prediction,
        issues=issues
    )


# =========================
# CHARTS
# =========================
@app.route("/charts")
def charts():
    ph, temp, tds, turbidity, time = get_data()

    prediction, _, _ = predict(
        safe_last(ph),
        safe_last(temp),
        safe_last(tds)
    )

    alert = "Unsafe Water Detected" if prediction == "Unsafe" else "System Normal"

    return render_template(
        "charts.html",
        alert=alert,
        ph=ph,
        temp=temp,
        tds=tds,
        turbidity=turbidity,
        time=time
    )


# =========================
# HISTORY
# =========================
@app.route("/history")
def history():
    ph, temp, tds, turbidity, time = get_data()

    prediction, _, _ = predict(
        safe_last(ph),
        safe_last(temp),
        safe_last(tds)
    )

    alert = "Unsafe Water Detected" if prediction == "Unsafe" else "System Normal"

    return render_template(
        "history.html",
        alert=alert,
        ph=ph,
        temp=temp,
        tds=tds,
        turbidity=turbidity,
        time=time
    )


# =========================
# ML PAGE
# =========================
@app.route("/ml")
def ml():
    ph, temp, tds, turbidity, time = get_data()

    prediction, probabilities, pred_index = predict(
        safe_last(ph),
        safe_last(temp),
        safe_last(tds)
    )

    alert = "Unsafe Water Detected" if prediction == "Unsafe" else "System Normal"

    issues = get_status(
        safe_last(ph),
        safe_last(tds),
        safe_last(temp),
        safe_last(turbidity)
    )

    return render_template(
        "ml.html",
        alert=alert,
        prediction=prediction,
        probabilities=probabilities,
        labels=labels,
        pred_index=pred_index,
        ph=safe_last(ph),
        temp=safe_last(temp),
        tds=safe_last(tds),
        turbidity=safe_last(turbidity),
        issues=issues
    )


# =========================
# API
# =========================
@app.route("/api/live")
def api_live():
    ph, temp, tds, turbidity, time = get_data()

    prediction, probabilities, pred_index = predict(
        safe_last(ph),
        safe_last(temp),
        safe_last(tds)
    )

    issues = get_status(
        safe_last(ph),
        safe_last(tds),
        safe_last(temp),
        safe_last(turbidity)
    )

    return jsonify({
        "ph": safe_last(ph),
        "temp": safe_last(temp),
        "tds": safe_last(tds),
        "turbidity": safe_last(turbidity),
        "prediction": prediction,
        "probabilities": probabilities,
        "labels": labels,
        "issues": issues,
        "timestamp": time[-1] if time else ""
    })


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True, port=5001)