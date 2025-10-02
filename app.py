import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load trained CNN model
model = load_model('digit_model.h5')

# ----------------- Helper Functions -----------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return img, thresh

def find_largest_contour(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    return largest

def get_warped(img, contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    pts = approx.reshape(4,2)
    rect = np.zeros((4,2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    side = max([
        np.linalg.norm(rect[0]-rect[1]),
        np.linalg.norm(rect[1]-rect[2]),
        np.linalg.norm(rect[2]-rect[3]),
        np.linalg.norm(rect[3]-rect[0])
    ])
    dst = np.array([[0,0],[side-1,0],[side-1,side-1],[0,side-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (int(side), int(side)))
    return warp

def predict_digit(cell_img):
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28,28))
    normalized = resized.reshape(1,28,28,1)/255.0
    pred = model.predict(normalized, verbose=0)
    return np.argmax(pred)

def extract_numbers(warped):
    side = warped.shape[0]
    cell_side = side // 9
    grid = np.zeros((9,9), dtype=int)

    for i in range(9):
        for j in range(9):
            y1, y2 = i*cell_side, (i+1)*cell_side
            x1, x2 = j*cell_side, (j+1)*cell_side
            cell = warped[y1:y2, x1:x2]
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            if cv2.countNonZero(thresh) < 50:
                continue
            grid[i,j] = predict_digit(cell)
    return grid

def is_valid(board, row, col, num):
    if num in board[row]: return False
    if num in board[:, col]: return False
    start_row, start_col = 3*(row//3), 3*(col//3)
    if num in board[start_row:start_row+3, start_col:start_col+3]: return False
    return True

def solve_sudoku(board):
    for i in range(9):
        for j in range(9):
            if board[i,j] == 0:
                for num in range(1,10):
                    if is_valid(board, i, j, num):
                        board[i,j] = num
                        if solve_sudoku(board): return True
                        board[i,j] = 0
                return False
    return True

def draw_solution(warped, original_grid, solved_grid):
    side = warped.shape[0]
    cell_side = side // 9
    for i in range(9):
        for j in range(9):
            if original_grid[i,j] == 0:
                x = j*cell_side + cell_side//4
                y = i*cell_side + (3*cell_side)//4
                cv2.putText(warped, str(solved_grid[i,j]), (x,y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    return warped

# ----------------- Routes -----------------
@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        if "sudoku_image" not in request.files:
            return redirect(request.url)
        file = request.files["sudoku_image"]
        if file.filename == "":
            return redirect(request.url)

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        img, thresh = preprocess_image(filepath)
        largest_contour = find_largest_contour(thresh)
        warped = get_warped(img, largest_contour)

        grid = extract_numbers(warped)
        solved_grid = grid.copy()

        if solve_sudoku(solved_grid):
            solved_img = draw_solution(warped, grid, solved_grid)
            solved_path = os.path.join(app.config["UPLOAD_FOLDER"], "solved_"+file.filename)
            cv2.imwrite(solved_path, solved_img)
            return render_template("index.html", solved_image=solved_path)
        else:
            return render_template("index.html", error="No solution found.")

    return render_template("index.html")

@app.route("/solve_manual", methods=["POST"])
def solve_manual():
    data = request.get_json()
    grid = np.array(data['grid'], dtype=int)
    original = grid.copy()
    if solve_sudoku(grid):
        return jsonify({"solution": grid.tolist()})
    else:
        return jsonify({"error":"Unable to solve the provided grid."})

if __name__=="__main__":
    app.run(debug=True)
