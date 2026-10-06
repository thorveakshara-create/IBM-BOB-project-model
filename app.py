from flask import Flask, request, jsonify, render_template_string
import joblib
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
MODEL_PATH = "model.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("model.pkl not found. Run train_model.py first.")
    return joblib.load(MODEL_PATH)

# ── HTML UI ──────────────────────────────────────────────────────────────────
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Salary Predictor</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
      background: #f0f4f8;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
    }
    .card {
      background: #fff;
      border-radius: 12px;
      padding: 40px 48px;
      box-shadow: 0 4px 24px rgba(0,0,0,0.10);
      width: 100%;
      max-width: 440px;
      text-align: center;
    }
    h1 { font-size: 1.5rem; color: #1f2328; margin-bottom: 6px; }
    p.sub { font-size: 0.9rem; color: #57606a; margin-bottom: 28px; }
    label { display: block; text-align: left; font-size: 0.85rem;
            font-weight: 600; color: #374151; margin-bottom: 6px; }
    input[type=number] {
      width: 100%; padding: 12px 14px; font-size: 1rem;
      border: 1.5px solid #d1d5db; border-radius: 8px; outline: none;
      transition: border-color .2s;
    }
    input[type=number]:focus { border-color: #3b82d4; }
    button {
      margin-top: 20px; width: 100%; padding: 13px;
      background: #3b82d4; color: #fff; border: none;
      border-radius: 8px; font-size: 1rem; font-weight: 600;
      cursor: pointer; transition: background .2s;
    }
    button:hover { background: #2563a8; }
    .result {
      margin-top: 24px; padding: 16px;
      background: #f0fdf4; border: 1.5px solid #bbf7d0;
      border-radius: 8px; font-size: 1.1rem; color: #166534;
      font-weight: 600; display: none;
    }
    .error {
      margin-top: 24px; padding: 12px;
      background: #fef2f2; border: 1.5px solid #fecaca;
      border-radius: 8px; font-size: 0.9rem; color: #991b1b;
      display: none;
    }
    .footer { margin-top: 32px; font-size: 0.75rem; color: #9ca3af; }
  </style>
</head>
<body>
  <div class="card">
    <h1>💼 Salary Predictor</h1>
    <p class="sub">Predict employee salary based on years of experience</p>
    <label for="exp">Years of Experience</label>
    <input type="number" id="exp" min="0" max="50" step="0.5" placeholder="e.g. 10" />
    <button onclick="predict()">Predict Salary</button>
    <div class="result" id="result"></div>
    <div class="error"  id="error"></div>
    <p class="footer">Model: Linear Regression · Dataset: salary_data.csv</p>
  </div>
  <script>
    async function predict() {
      const years = document.getElementById('exp').value;
      const res   = document.getElementById('result');
      const err   = document.getElementById('error');
      res.style.display = 'none';
      err.style.display = 'none';
      if (years === '' || isNaN(years) || Number(years) < 0) {
        err.textContent = 'Please enter a valid non-negative number.';
        err.style.display = 'block'; return;
      }
      try {
        const r = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ years_of_experience: Number(years) })
        });
        const data = await r.json();
        if (data.error) { err.textContent = data.error; err.style.display = 'block'; return; }
        res.textContent = `Predicted Salary: $${data.predicted_salary.toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2})}`;
        res.style.display = 'block';
      } catch(e) {
        err.textContent = 'Request failed. Is the server running?';
        err.style.display = 'block';
      }
    }
    document.getElementById('exp').addEventListener('keydown', e => { if (e.key === 'Enter') predict(); });
  </script>
</body>
</html>
"""

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if "years_of_experience" not in data:
        return jsonify({"error": "Missing field: years_of_experience"}), 400

    try:
        years = float(data["years_of_experience"])
    except (ValueError, TypeError):
        return jsonify({"error": "years_of_experience must be a number"}), 400

    if years < 0:
        return jsonify({"error": "years_of_experience cannot be negative"}), 400

    try:
        model = load_model()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    salary = model.predict(pd.DataFrame({"Experience": [years]}))[0]
    return jsonify({
        "years_of_experience": years,
        "predicted_salary": round(float(salary), 2)
    })


@app.route("/health")
def health():
    model_ready = os.path.exists(MODEL_PATH)
    return jsonify({"status": "ok", "model_loaded": model_ready})


if __name__ == "__main__":
    print("Starting Salary Predictor API...")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True, port=5000)
