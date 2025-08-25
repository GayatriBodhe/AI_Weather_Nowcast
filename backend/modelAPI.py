from flask import Flask, send_file, jsonify
import matplotlib
matplotlib.use("Agg")   # use non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
import os

# Config
IMG_SIZE = (128, 128)       # resize input images
SEQ_LEN = 5                 # number of frames in input sequence
OUTPUT_SIZE = (512, 512)    # upscale predicted output
DATA_DIR = "test_images"

# Load model once at startup
model = tf.keras.models.load_model("radar_nowcast_model.keras")

app = Flask(__name__)

def load_last_sequence():
    """Load the last SEQ_LEN frames from folder and preprocess."""
    files = sorted(os.listdir(DATA_DIR))[-SEQ_LEN:]
    frames = []
    for f in files:
        path = os.path.join(DATA_DIR, f)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {f}")
        img = cv2.resize(img, IMG_SIZE)
        img = img.astype("float32") / 255.0
        frames.append(img)
    arr = np.array(frames)[np.newaxis, ..., np.newaxis]  # (1, SEQ_LEN, H, W, 1)
    return arr, files

# ------------------------------
# a) Input sequence (5 images)
# ------------------------------
@app.route("/predict/input", methods=["GET"])
def predict_input():
    """Return the 5 input frames (Blues colormap)."""
    try:
        files = sorted(os.listdir(DATA_DIR))[-(SEQ_LEN+1):]  # last 6
        input_files = files[:SEQ_LEN]

        plt.figure(figsize=(15, 3))
        for i, f in enumerate(input_files):
            img = cv2.imread(os.path.join(DATA_DIR, f), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, IMG_SIZE)
            plt.subplot(1, SEQ_LEN, i+1)
            plt.imshow(img, cmap="Blues")
            plt.title(f"Input {i+1}")
            plt.axis("off")

        os.makedirs("static", exist_ok=True)
        out_path = "static/prediction_input.png"
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close()

        return send_file(out_path, mimetype="image/png")

    except Exception as e:
        import traceback
        print("🔥 Error in /predict/input:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# ------------------------------
# b) Actual output (6th frame)
# ------------------------------
@app.route("/predict/actual", methods=["GET"])
def predict_actual():
    """Return the actual 6th frame (Blues colormap)."""
    try:
        files = sorted(os.listdir(DATA_DIR))[-(SEQ_LEN+1):]
        actual_file = files[-1]

        img = cv2.imread(os.path.join(DATA_DIR, actual_file), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMG_SIZE)

        plt.figure(figsize=(5, 5))
        plt.imshow(img, cmap="Blues")
        plt.title("Actual Output")
        plt.axis("off")

        os.makedirs("static", exist_ok=True)
        out_path = "static/prediction_actual.png"
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close()

        return send_file(out_path, mimetype="image/png")

    except Exception as e:
        import traceback
        print("🔥 Error in /predict/actual:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# ------------------------------
# c) Predicted output (model)
# ------------------------------
@app.route("/predict/predicted", methods=["GET"])
def predict_predicted():
    """Return the predicted output frame (Blues colormap)."""
    try:
        files = sorted(os.listdir(DATA_DIR))[-(SEQ_LEN+1):]
        input_files = files[:SEQ_LEN]

        frames = []
        for f in input_files:
            img = cv2.imread(os.path.join(DATA_DIR, f), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, IMG_SIZE)
            frames.append(img.astype("float32") / 255.0)

        X_input = np.array(frames)[np.newaxis, ..., np.newaxis]

        pred = model.predict(X_input)
        pred_frame = pred[0, :, :, 0]

        plt.figure(figsize=(5, 5))
        plt.imshow(pred_frame, cmap="Blues")
        plt.title("Predicted Output")
        plt.axis("off")

        os.makedirs("static", exist_ok=True)
        out_path = "static/prediction_predicted.png"
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close()

        return send_file(out_path, mimetype="image/png")

    except Exception as e:
        import traceback
        print("🔥 Error in /predict/predicted:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)