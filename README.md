Sudoku Solver – Intelligent Puzzle Solving Web Application
A complete Sudoku-solving system built using Flask, OpenCV, TensorFlow/Keras, and modern UI principles.

Overview
Sudoku Solver is a web application that can automatically detect, interpret, and solve Sudoku puzzles from either an uploaded image or manual input. The system uses computer vision for grid extraction and digit recognition, a trained deep-learning model for digit classification, and a backtracking algorithm for solving the puzzle. It provides both automatic and step-by-step visualization of the solution process, with a clean and responsive interface.

This project demonstrates skills in web development, computer vision, deep learning, algorithm design, and user-centric interface building.

Core Features

Image-Based Sudoku Solving
The user can upload a photo of a Sudoku puzzle.
OpenCV is used for grid detection, perspective correction, and digit extraction.
A TensorFlow/Keras digit classifier identifies the numbers.
The solver algorithm completes the puzzle and returns the solved Sudoku overlaid on the original grid.

Manual Input Mode
Users can manually enter numbers into a digital grid.
The solver can animate the entire backtracking process step by step.
The animation speed can be saved across sessions.

Modern UI with Theming
Supports light and dark themes with automatic persistence.
Includes keyboard shortcuts:
T for theme toggle
Spacebar for start/pause solving animation
The UI is responsive and accessible on all devices.

Solver Algorithm
Uses an optimized backtracking algorithm with constraint checks.
Handles invalid puzzles gracefully.
Provides visual feedback when solving step-by-step.

Fully Responsive Web App
Built with Flask for backend logic and HTML, CSS, and JavaScript for frontend interactions.
Works on desktops, tablets, and mobile devices.

Project Structure
The project contains the following main components:

app.py: Flask server and primary application logic

model/: trained TensorFlow/Keras digit recognition model

static/: CSS, JS, and theme-related assets

templates/: HTML templates for the user interface

utils/: OpenCV image processing utilities, solver logic, and helper functions

How It Works

When solving via an image, OpenCV preprocesses the uploaded picture, extracts the Sudoku grid, isolates each digit, and prepares it for classification.

The classifier predicts digits using a TensorFlow/Keras model.

The Sudoku grid is reconstructed internally.

The backtracking solver fills the puzzle.

The solved grid is projected back onto the original image.

The final solved Sudoku is returned to the user.

Usage Instructions

Create and activate a virtual environment.

Install dependencies using pip.

Ensure the TensorFlow/Keras model file is placed in the correct model directory.

Run the Flask application using python app.py.

Open the web interface in your browser.

Upload a Sudoku image or switch to manual mode to begin solving.

Skills Demonstrated
This project showcases knowledge and experience in the following areas:

Flask web development

Computer vision using OpenCV

Deep learning using TensorFlow and Keras

Backtracking algorithms

UI design with theme persistence

Client-server architecture

Image processing and OCR techniques

Responsive and interactive web design

Future Enhancements
Several improvements can be added:

Better digit recognition model trained on larger datasets

Real-time camera capture mode

Highlighting conflicting cells in manual mode

Touch-friendly interaction modes for mobile devices

Progressive Web App (PWA) version for offline use

Summary
Sudoku Solver is a complete, intelligent puzzle-solving application that combines computer vision, deep learning, backend logic, and an interactive user interface. It allows users to solve puzzles from images or manually, provides clear visualization of the solving process, and delivers a polished, responsive experience.