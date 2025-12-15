# CSC173 Deep Computer Vision Project Proposal
**Student:** Jace Vihzalel C. Sambilad, 2022-0144
**Date:** 12/15/2025

## 1. Project Title 
Real-Time Wildlife Detection System for Road Safety Using YOLOv8

## 2. Problem Statement
Wildlife crashes on the roads are a huge threat to human safety as well as to animal conservation. This is especially true in areas where roads intersect natural landscapes. In the Philippines, you have endangered species like the Philippine deer and wild boar crossing roads in areas with lots of forest - leading to accidents and animal deaths. We're going to develop a real-time computer vision system that can find wildlife on roads using dash camera footage. This system will then alert the driver before hand and potentially be integrated with intelligent transport systems to reduce accidents and protect wildlife.

## 3. Objectives
- Train a Yolo8 model that can detect multiple wildlife species on the road with at least an 85% mAP@0.5
- Get a system that can run real-time video streams at more than 20 frames per second on an NVIDIA RTX 3050TI (mobile)
- Create a solid pipeline to train our model on data that we've preprocessed, augmented, validated and tested to make sure it works well on all sorts of road conditions

## 4. Dataset Plan
- Source: https://www.kaggle.com/datasets/antoreepjana/animals-detection-images-dataset/data Wildlife Detection Dataset (Kaggle) ~5,000-10,000 images
- Classes: deer, boar, dog, cow, cat, chicken, monkey
- Acquisition: We'll download these straight from Kaggle/Roboflow

## 5. Technical Approach
#### Architecture Sketch
```
Input: Road Camera Feed
    ↓
Preprocessing: Resize, Normalization
    ↓
YOLOv8n Backbone: Feature extraction
    ↓
Neck: Multi-scale feature fusion
    ↓
Detection Head: Bounding boxes + confidence scores
    ↓
Post-processing: NMS, confidence filter
    ↓
Output: Wildlife detected → Alert + annotated frame
```
#### Approach
- Model: We'll be using Yolo8n (nano) for speed and accuracy
- Framework: PyTorch + the Ultralytics library
- Hardware: We'll train on Google Colab Pro (T4/A100 GPU)
- Training Strategy: We'll be using transfer learning from the COCO pre-trained weights

## 6. Expected Challenges & Mitigations
- Bad lighting: Train on lots of different lighting, and then use brightness/exposure to mix it up
- Wildlife is hidden: Get some animals in the data that are half-hidden; use multi-scale training
- Mistake for shadows: Get some bad examples; use a confidence threshold
