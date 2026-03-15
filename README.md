# Virtual Air Painter 🖌️

A computer vision application that allows users to draw in the air using their finger, powered by OpenCV and MediaPipe hand tracking. The system detects finger gestures in real time through a webcam and translates them into strokes on a digital canvas.

This project demonstrates how hand landmark detection and gesture recognition can be used to build interactive vision-based interfaces.

# Features

Real-time hand tracking using MediaPipe

Draw on screen using your index finger

Gesture-based controls

Color selection buttons

Canvas clearing

Persistent drawing overlay

Runs entirely using a webcam

# Demo Controls
Gesture	Action
Index finger up	Draw
Index + middle finger up	Selection mode
Move finger over color buttons	Change drawing color
Move finger over CLEAR	Clear canvas
Press Q	Quit application
UI Layout

At the top of the screen:

CLEAR | BLUE | GREEN | RED | YELLOW

When selection mode is active (index + middle finger up), move your finger over these buttons to select the action.

# How It Works

The pipeline of the application is:

Webcam → Hand Detection → Landmark Extraction → Gesture Detection
     → UI Interaction → Drawing Engine → Canvas Overlay
1. Hand Tracking

MediaPipe detects 21 hand landmarks per frame.

# Important landmarks used:

Landmark	Description
8	Index fingertip
12	Middle fingertip

These landmarks determine the cursor position and gesture state.

# 2. Gesture Detection

The system determines which fingers are raised by comparing fingertip positions with the finger joints.

Example:

Index up → drawing mode
Index + middle up → selection mode

# 3. Drawing System

Drawing is done using two layers:

Live camera frame

frame

Persistent drawing canvas

canvas

The canvas stores the strokes while the camera feed updates every frame.

# 4. Overlaying Drawing on Camera

The program combines both layers using bitwise masking:

canvas → grayscale
grayscale → inverse mask
frame AND mask → remove drawing area
frame OR canvas → overlay drawing

This ensures:

camera feed remains visible

drawing persists between frames

# Installation
1. Clone the repository
git clone https://github.com/Svamin-B/AIR-PAINTER.git
cd AIR-PAINTER
2. Create environment

Using Conda:

conda env create -f environment.yml
conda activate cvpaint
3. Install dependencies manually (optional)
pip install numpy opencv-python mediapipe==0.10.20
Running the Application
python air_painter.py

The webcam window will open and you can begin drawing.

Press Q to exit.

# Project Structure
AIR PAINTER/
│
├── air_painter.py
└── environment.yml

# Technologies Used

- Python

- OpenCV

- MediaPipe

- NumPy

# Possible Improvements

- Some potential enhancements include:

- Smoother drawing using point averaging

- Pinch gesture for drawing instead of finger state detection

- Adjustable brush thickness

- Eraser tool

- Save drawing to image file

- Multi-hand interaction

- Learning Outcomes

# This project demonstrates concepts including:

- Real-time computer vision pipelines

- Hand landmark detection

- Gesture recognition

- Image masking and compositing

- Interactive vision-based interfaces

# Author

Svamin Bhatnagar
