# CSC173 Deep Computer Vision Project Progress Report
**Student:** Jace Vihzalel C. Sambilad, 2022-0144
**Date:** 12/23/2025
**Repository:** https://github.com/jeysu/CSC173-DeepCV-Sambilad


## 📊 Current Status
| Milestone | Status | Notes |
|-----------|--------|-------|
| Dataset Preparation | ✅ Completed | ~8000 images downloaded/preprocessed |
| Initial Training | ✅ Completed | 50 epochs completed |
| Baseline Evaluation | ⏳ Pending | Ongoing |
| Model Fine-tuning | ⏳ Pending | Ongoing |

## 1. Dataset Progress
- **Total images:** ~8000
- **Train/Val/Test split:** 80%/10%/10%
- **Classes implemented:** 4 classes: Cow, Pig, Monkey, Chicken
- **Preprocessing applied:** Auto-Orient(Applied), Resize(640), Augmentation (Brightness, Noise)

**Sample data preview:**
<img width="1028" height="681" alt="image" src="https://github.com/user-attachments/assets/7edb6300-4e35-416d-920b-814af18deac9" />

## 2. Training Progress

**Training Curves**
<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/d0c506eb-0748-4650-b8a9-34f2a5200af6" />

**Current Metrics:**
| Class | Images | Instances | Box (P) | Recall (R) | mAP@0.5 | mAP@0.5-95 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **all** | 707 | 1346 | 0.828 | 0.778 | 0.831 | 0.535 |
| **Chicken** | 149 | 184 | 0.803 | **0.897** | 0.876 | **0.634** |
| **Cow** | 158 | 647 | 0.743 | 0.612 | 0.697 | 0.453 |
| **Monkey** | 163 | 226 | 0.872 | 0.877 | **0.922** | 0.613 |
| **Pig** | 171 | 289 | **0.895** | 0.727 | 0.828 | 0.44 |

## 3. Challenges Encountered & Solutions
| Issue | Status | Resolution |
|-------|--------|------------|
| Colab runtime disconnect | ✅ Fixed | Paid for extra compute units |
| Roboflow dataset not exporting| ✅ Fixed | Paid for extra compute units |
| Class annotation imbalance | ⏳ Ongoing |  |

## 4. Next Steps (Before Final Submission)
- [ ] Complete training (50 more epochs)
- [ ] Hyperparameter tuning (learning rate, augmentations)
- [ ] Baseline comparison (vs. original pre-trained model)
- [ ] Record 5-min demo video
- [ ] Write complete README.md with results
