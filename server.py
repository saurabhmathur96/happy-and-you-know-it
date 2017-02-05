from flask import Flask, jsonify, request, render_template
from io import BytesIO
from PIL import Image
import base64
from models import ResNet20


app = Flask(__name__)
model = ResNet20(input_shape=(48, 48))
model.load("models/resnet20.h5")
emotions = ("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/classify", methods=["POST"])
def classify ():
    body = request.get_json()
    image_base64 = base64.b64decode(body["image"].split(",")[1])
    image = Image.open(BytesIO(image_base64)).convert("L")
    prediction = model.predict(image)[0].tolist()
    
    return jsonify({ "prediction": dict(zip(emotions, prediction)) })


if __name__ == "__main__":
    app.run()
