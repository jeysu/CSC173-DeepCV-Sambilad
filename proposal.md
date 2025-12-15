# CSC173 Deep Computer Vision Project Proposal
**Student:** Jace Vihzalel C. Sambilad, 2022-0144
**Date:** 12/15/2025

## 1. Project Title 
Real-Time Agricultural Pest Detection and Intrusion Alert System Using YOLOv8

## 2. Problem Statement
In the Philippines, farmers who are running small scale farms are losing out big when crop-raiding animals like Wild Boars (Baboy Damo) and Monkeys show up & start causing chaos with their raids. A single night, these pests can go through an entire harvest of corn or rice. At the moment, farmers have to fall back on manual guarding - which is a real hazard and way too tiring - to protect their crops. What this means is there's a desperate need for a surveillance system that is automated, doesn't hurt the wildlife and can pick up these critters in real-time so the farmers can get an alarm sounded and scare them off. This will cut down on crop wastage and help keep those farmers in work without putting the local wildlife in danger.

## 3. Objectives
- Train up a YOLOv8 model to specifically spot crop pests (Boars, Monkeys) and livestock (Cows, Chickens) and make sure its accuracy is at least 85% mAP@0.5.
- Build a detection system that can knock out frames at more than 60 FPS on an NVIDIA RTX 3050Ti (mobile) so farmers can get immediate warning of any pests turning up.
- Integrate an 'alarm logic' module that logs the detection of any critters and then simulates a trigger for some sort of audible alarm (like a buzzer or speaker) whenever a pest gets spotted.

## 4. Dataset Plan
- Source: https://www.kaggle.com/datasets/antoreepjana/animals-detection-images-dataset/data Wildlife Detection Dataset (Kaggle) ~5,000-10,000 images
- Pest Classes: Boar, Monkey, Deer, etc.
- Non-Pest Classes: Cow, Chicken, Mule, etc.
- Acquisition: We'll download these straight from Kaggle

## 5. Technical Approach
#### Architecture Sketch
```
Input: Farm/Field Camera Feed
    ↓
Preprocessing: Resize, Normalization
    ↓
Backbone: Feature extraction
    ↓
Neck: Multi-scale feature fusion
    ↓
Head: Bounding boxes + Class Classification
    ↓
Post-processing: Confidence Thresholding
    ↓
Logic-check: Class Identification
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
