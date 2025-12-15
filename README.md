# Real-Time Agricultural Pest Detection and Intrusion Alert System Using YOLOv8
**CSC173 Intelligent Systems Final Project**  
*Mindanao State University - Iligan Institute of Technology*  
**Student:** Jace Vihzalel C. Sambilad, 2022-0144
**Semester:** AY 2025-2026 Sem 1  
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org)

## Abstract
Crop raiding animals, especially wild boars and macaques, are wreaking havoc on agriculture in the Philippines that utterly destroying small-scale corn and rice crops all in one go. Its not uncommon to see farmers' livelihoods threatened by these masked critters. Traditional methods to counter this problem, like putting up high fences or having farmers keep watch at night, typically prove to be a lot of hard work, pose real risks to those doing the watching, and are rarely effective against a determined animal. We're proposing a real-time pest detection and alert system that makes use of the YOLOv8 deep learning concept, and is designed to get round the false alarms which can happen when harmless animals (think cows or chickens) get picked up by the system. We're going to train the model on the Kaggle Wildlife Detection dataset using transfer learning, and aim for an Average Precision of at least 85%. We're looking to get the system up to 20 frames per second on a standard piece of hardware and CPU for optimization. This will then kick in an alarm module straight away if the system detects an unwanted pest. This system offers an easy and cheap solution that will help protect crops without putting farmers through so much physical hardship.

## Table of Contents
- [Introduction](#introduction)
- [Related Work](#related-work)
- [Methodology](#methodology)
- [Experiments & Results](#experiments--results)
- [Discussion](#discussion)
- [Ethical Considerations](#ethical-considerations)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [References](#references)

## Introduction
### Problem Statement
Small scale farmers in the rural Philippines are in a pretty desperate situation because of all the economic uncertainty thats caused by the conflict between humans and wildlife. Particularly, crop raiding by wild boars and monkeys is a huge problem for these farmers. These critters, being nocturnal and crepuscular, can wipe out entire hectares of staple crops like rice & corn overnight, leaving farming families without any food. At the moment, the main thing farmers can do to try & stop these pests from raiding their crops is to have someone physically keep an eye on the fields overnight. And this isn't really working out too well.

This manual surveillance setup has a few pretty major problems:
- It's a nightmare for the farmer's safety - its dark and you cant see much, so running into a wild animal is a very real possibility.
- It's also really inefficient - people get tired and therefore the monitoring isnt very effective and pretty soon pests are slipping in under the radar.
- Not terribly reliable either - simple alarms that rely on motion sensors are often way too sensitive and end up triggering false alarms all the time because of wind, falling leaves or even stray livestock which just leads to alarm fatigue.

There is a pretty noticeable lack of cheap automated systems that can tell the difference between wild pests and other animals in real time. So we really need a computer vision solution that can sit on the field and autonomously monitor it all, be able to tell which animals are doing the damaging and then automatically sound the alarm to alert the farmer to the problem before all the crops get destroyed. No need for a person to be sitting there all the time.

### Objectives
- [Objective 1: e.g., Achieve >90% detection accuracy]
- [Objective 2: Integrate with decision logic]
- [Objective 3: Deploy on edge device]

![Problem Demo](images/problem_example.gif) [web:41]

## Related Work
- [Paper 1: YOLOv8 for real-time detection [1]]
- [Paper 2: Transfer learning on custom datasets [2]]
- [Gap: Your unique approach, e.g., Mindanao-specific waste classes] [web:25]

## Methodology
### Dataset
- Source: [e.g., Custom 5K images + COCO subset]
- Split: 70/15/15 train/val/test
- Preprocessing: Augmentation, resizing to 640x640 [web:41]

### Architecture
![Model Diagram](images/architecture.png)
- Backbone: [e.g., CSPDarknet53]
- Head: [e.g., YOLO detection layers]
- Hyperparameters: Table below

| Parameter | Value |
|-----------|-------|
| Batch Size | 16 |
| Learning Rate | 0.01 |
| Epochs | 100 |
| Optimizer | SGD |

### Training Code Snippet
train.py excerpt
model = YOLO('yolov8n.pt')
model.train(data='dataset.yaml', epochs=100, imgsz=640)


## Experiments & Results
### Metrics
| Model | mAP@0.5 | Precision | Recall | Inference Time (ms) |
|-------|---------|-----------|--------|---------------------|
| Baseline (YOLOv8n) | 85% | 0.87 | 0.82 | 12 |
| **Ours (Fine-tuned)** | **92%** | **0.94** | **0.89** | **15** |

![Training Curve](images/loss_accuracy.png)

### Demo
![Detection Demo](demo/detection.gif)
[Video: [CSC173_YourLastName_Final.mp4](demo/CSC173_YourLastName_Final.mp4)] [web:41]

## Discussion
- Strengths: [e.g., Handles occluded trash well]
- Limitations: [e.g., Low-light performance]
- Insights: [e.g., Data augmentation boosted +7% mAP] [web:25]

## Ethical Considerations
- Bias: Dataset skewed toward plastic/metal; rural waste underrepresented
- Privacy: No faces in training data
- Misuse: Potential for surveillance if repurposed [web:41]

## Conclusion
[Key achievements and 2-3 future directions, e.g., Deploy to Raspberry Pi for IoT.]

## Installation
1. Clone repo: `git clone https://github.com/yourusername/CSC173-DeepCV-YourLastName`
2. Install deps: `pip install -r requirements.txt`
3. Download weights: See `models/` or run `download_weights.sh` [web:22][web:25]

**requirements.txt:**
torch>=2.0
ultralytics
opencv-python
albumentations

## References
[1] Jocher, G., et al. "YOLOv8," Ultralytics, 2023.  
[2] Deng, J., et al. "ImageNet: A large-scale hierarchical image database," CVPR, 2009. [web:25]

## GitHub Pages
View this project site: [https://jjmmontemayor.github.io/CSC173-DeepCV-Montemayor/](https://jjmmontemayor.github.io/CSC173-DeepCV-Montemayor/) [web:32]
