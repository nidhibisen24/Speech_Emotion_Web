import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from predict import predict_emotion

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
UPLOAD_FOLDER = os.path.join(STATIC_DIR, "uploads")

app = Flask(
    __name__,
    static_folder=STATIC_DIR,
    template_folder=TEMPLATES_DIR
)

ALLOWED_EXTENSIONS = {"wav"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "audiofile" not in request.files:
        return render_template("result.html", error="No file part found.")

    file = request.files["audiofile"]

    if file.filename == "":
        return render_template("result.html", error="No file selected.")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            emotion, confidence, probabilities = predict_emotion(filepath)

            sorted_probs = dict(
                sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
            )

            audio_relative_path = os.path.join("static", "uploads", filename)

            return render_template(
                "result.html",
                emotion=emotion,
                confidence=round(confidence, 2),
                probabilities=sorted_probs,
                audio_path=audio_relative_path
            )

        except Exception as e:
            return render_template("result.html", error=str(e))

    return render_template("result.html", error="Only .wav files are allowed.")


if __name__ == "__main__":
    app.run(debug=True, port=5001)