# Automated Soccer Match Analysis using Deep Learning

This repository contains the full implementation of a modular system for the automated analysis of soccer matches using deep learning and computer vision techniques. The system processes single-camera broadcast footage and extracts tactical and physical metrics from match play.

## 🔍 Project Overview

The system includes:
- **Object Detection** using YOLOv11 (players, ball, referees, goalkeepers)
- **Tracking** via ByteTrack for consistent object IDs
- **Team Assignment** based on jersey color clustering
- **Camera Stabilization** using optical flow and Lucas-Kanade feature tracking
- **View Transformation** from image space to top-down field coordinates
- **Statistical Analysis** including player speed, distance, and ball possession
- **Final Output** as both annotated video and structured machine-readable data

<p align="center">
  <img src="output_videos/pictures/annotated_video.png" width="600"/>
</p>

## 🧠 Technologies Used

- Python 3.10
- YOLOv11 (via [Ultralytics](https://github.com/ultralytics/ultralytics))
- OpenCV
- NumPy
- Scikit-learn (for KMeans clustering)
- Roboflow (for dataset creation and labeling)

## 📊 Dataset

The object detection model was trained on a custom Roboflow dataset containing annotated Bundesliga match footage:
[Football Players Detection Dataset – Roboflow Universe](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1)

- 612 training images
- 38 validation images
- 13 test images
- Resolution: 1920x1080
- Augmentations: horizontal flip, brightness/saturation adjustments
