# Non-destructive Egg Freshness Evaluation

This repository contains the core implementation of a physics-driven framework for evaluating egg freshness non-destructively using computer vision and hydrostatic equilibrium.

## Workflow

<img width="1624" height="823" alt="Figure" src="https://github.com/user-attachments/assets/bb8bf83d-84d7-4de4-86ef-eedd3c4ebc3f" />

## Core Features

* **Omnidirectional Image Acquisition:** Isolates the egg in 15°C ultra-pure water and employs a motorized camera system for 360° recording, capturing sequential frames at precise 0.5° angular increments.
* **Optimal Projection Selection:** Extracts binary segmentation masks to calculate pixel-wise projected areas, identifying the frame with the maximum projected area (with cross-validation for morphological consistency).
* **Angle Computation:** Utilizes SAM2 for precise segmentation. It fits an optimal contour model to the target using PCA initialization and Non-linear Least Squares (NLS) fitting to compute the acute tilt angle against a defined true horizontal plane.

## Requirements

* Python 3.x
* SAM2 ([Segment Anything Model 2](https://github.com/facebookresearch/sam2))
* Standard scientific and vision libraries: `numpy`, `scipy`, `opencv-python`

## Usage

Ensure your environment is set up with the required dependencies. The main entry point for the contour extraction and angle calculation pipeline is:

```bash
python3 egg_measure.py <image_path> 
