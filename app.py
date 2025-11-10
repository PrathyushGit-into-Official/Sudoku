import os
import uuid
import logging
import cv2
import numpy as np
import requests
from flask import Flask, render_template, request, redirect, jsonify, url_for
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# Flask app config
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8 MB upload limit

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------- Auto-download model helper -----------------
MODEL_PATH = 'digit_model.h5'
MODEL_URL = os.environ.get('MODEL_URL')  # set this in Render / environment when deploying

def download_model_if_missing():
    """Download the model from MODEL_URL if it doesn't exist locally."""
    if os.path.exists(MODEL_PATH):
        logger.info("Model already exists at %s", MODEL_PATH)
        return
    if not MODEL_URL:
        logger.warning("MODEL_URL not set and %s missing — cannot download model.", MODEL_PATH)
        return

    logger.info("Downloading model from %s ...", MODEL_URL)
    try:
        resp = requests.get(MODEL_URL, stream=True, timeout=120)
        resp.raise_for_status()
        with open(MODEL_PATH, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info("Downloaded model to %s", MODEL_PATH)
    except Exception as e:
        logger.exception("Failed to download model: %s", e)

# Attempt to download the model before loading
download_model_if_missing()

# Load model if present
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        logger.info("Loaded model from %s", MODEL_PATH)
    except Exception as e:
        logger.exception("Failed to load model: %s", e)
else:
    logger.warning("Model file %s not found. Run train_model.py or set MODEL_URL to generate it.", MODEL_PATH)

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the uploaded image.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return img, thresh

def find_largest_contour(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)

def get_warped(img, contour):
    if contour is None:
        return None
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if approx is None or len(approx) < 4:
        return None
    try:
        pts = approx.reshape(4, 2).astype('float32')
    except Exception:
        return None

    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    side = int(max([
        np.linalg.norm(rect[0] - rect[1]),
        np.linalg.norm(rect[1] - rect[2]),
        np.linalg.norm(rect[2] - rect[3]),
        np.linalg.norm(rect[3] - rect[0])
    ]))
    if side <= 0:
        return None

    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (side, side))
    return warp

def predict_digit(cell_img):
    if model is None:
        return 0
    try:
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    except Exception:
        gray = cell_img.copy() if len(cell_img.shape) == 2 else cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    normalized = resized.reshape(1, 28, 28, 1).astype('float32') / 255.0
    pred = model.predict(normalized, verbose=0)
    probs = pred[0]
    best_idx = int(np.argmax(probs))
    conf = float(np.max(probs))

    # If low confidence treat as empty
    if conf < 0.55:
        return 0

    return best_idx

def extract_numbers(warped):
    if warped is None:
        return np.zeros((9, 9), dtype=int)
    side = warped.shape[0]
    cell_side = side // 9
    grid = np.zeros((9, 9), dtype=int)

    for i in range(9):
        for j in range(9):
            y1, y2 = i * cell_side, (i + 1) * cell_side
            x1, x2 = j * cell_side, (j + 1) * cell_side
            cell = warped[y1:y2, x1:x2]
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            nonzero = cv2.countNonZero(thresh)
            area = thresh.shape[0] * thresh.shape[1]
            if area == 0:
                continue
            fill_ratio = nonzero / float(area)
            if fill_ratio < 0.01:
                continue

            # Crop to bounding box
            x_coords = np.any(thresh, axis=0)
            y_coords = np.any(thresh, axis=1)
            if not x_coords.any() or not y_coords.any():
                continue
            x_min, x_max = np.where(x_coords)[0][[0, -1]]
            y_min, y_max = np.where(y_coords)[0][[0, -1]]
            pad = 2
            x_min = max(0, x_min - pad)
            x_max = min(thresh.shape[1] - 1, x_max + pad)
            y_min = max(0, y_min - pad)
            y_max = min(thresh.shape[0] - 1, y_max + pad)
            digit_roi = cell[y_min:y_max + 1, x_min:x_max + 1]

            digit = predict_digit(digit_roi)
            grid[i, j] = int(digit)

    return grid

def is_valid(board, row, col, num):
    if num == 0:
        return True
    if num in board[row, :]:
        return False
    if num in board[:, col]:
        return False
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    if num in board[start_row:start_row + 3, start_col:start_col + 3]:
        return False
    return True

def solve_sudoku(board):
    for i in range(9):
        for j in range(9):
            if board[i, j] == 0:
                for num in range(1, 10):
                    if is_valid(board, i, j, num):
                        board[i, j] = num
                        if solve_sudoku(board):
                            return True
                        board[i, j] = 0
                return False
    return True

def draw_solution(warped, original_grid, solved_grid):
    side = warped.shape[0]
    cell_side = side // 9
    font_scale = max(0.6, cell_side / 40.0)
    thickness = max(1, int(cell_side / 20))
    for i in range(9):
        for j in range(9):
            if original_grid[i, j] == 0:
                text = str(int(solved_grid[i, j]))
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                x = int(j * cell_side + (cell_side - w) / 2)
                y = int(i * cell_side + (cell_side + h) / 2)
                cv2.putText(warped, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 200, 0), thickness, cv2.LINE_AA)
    return warped

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "sudoku_image" not in request.files:
            return redirect(request.url)
        file = request.files["sudoku_image"]
        if file.filename == "":
            return redirect(request.url)
        if not allowed_file(file.filename):
            return render_template("index.html", error="Unsupported file type. Use PNG or JPG.")

        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        try:
            file.save(filepath)
            img, thresh = preprocess_image(filepath)
            largest_contour = find_largest_contour(thresh)
            if largest_contour is None:
                return render_template("index.html", error="Could not detect Sudoku grid. Try a clearer top-down photo.")
            warped = get_warped(img, largest_contour)
            if warped is None:
                return render_template("index.html", error="Failed to extract a square grid. Try a clearer top-down photo.")
            grid = extract_numbers(warped)
            solved_grid = grid.copy()
            if solve_sudoku(solved_grid):
                solved_img = draw_solution(warped.copy(), grid, solved_grid)
                solved_filename = f"solved_{unique_name}"
                solved_path = os.path.join(app.config['UPLOAD_FOLDER'], solved_filename)
                cv2.imwrite(solved_path, solved_img)
                solved_url = url_for('static', filename=f"uploads/{solved_filename}")
                return render_template("index.html", solved_image=solved_url)
            else:
                return render_template("index.html", error="No solution found for the detected grid.")
        except Exception as ex:
            logger.exception("Error processing uploaded image:")
            return render_template("index.html", error="An error occurred while processing the image.")
    return render_template("index.html")

@app.route("/solve_manual", methods=["POST"])
def solve_manual():
    try:
        data = request.get_json(force=True)
        grid = np.array(data.get('grid', []), dtype=int)
        if grid.shape != (9, 9):
            return jsonify({"error": "Grid must be a 9x9 array."})
        if solve_sudoku(grid):
            return jsonify({"solution": grid.tolist()})
        else:
            return jsonify({"error": "Unable to solve the provided grid."})
    except Exception:
        logger.exception("Error in solve_manual:")
        return jsonify({"error": "Server error while solving."})

if __name__ == "__main__":
    app.run(debug=True)
