# 🚤 Autonomous Underwater Vehicle (AUV) Project

Welcome to the **Autonomous Underwater Vehicle (AUV)** project repository, developed for the **SAUVC 2024 Challenge**! This project focuses on building an underwater vehicle capable of autonomous navigation and performing tasks such as object detection and retrieval. 🌊

---
## 📷 Visual Overview
### Image inhancement
<p align="center">
  <img src="https://github.com/amandeep-yadav/Autonomous-Underwater-Vehicle-AUV-/blob/main/img/before.jpg" alt="Before Enhancement" width="40%">
  <img src="https://github.com/amandeep-yadav/Autonomous-Underwater-Vehicle-AUV-/blob/main/img/after.jpg" alt="After Enhancement" width="40%">
</p>

### Image detection
<p align="center">
  <img src="https://github.com/amandeep-yadav/Autonomous-Underwater-Vehicle-AUV-/blob/main/img/detect2.jpg" alt="Before Enhancement" width="40%">
  <img src="https://github.com/amandeep-yadav/Autonomous-Underwater-Vehicle-AUV-/blob/main/img/detect2.png " alt="After Enhancement" width="40%">
</p>

## ✨ Features

✅ Autonomous navigation in underwater environments.  
✅ Advanced object detection capabilities, including:
- **Gate detection**
- **Flare identification**
- **Colored bucket recognition**
- **Stand detection** (including stands with balls for retrieval)  
✅ Comprehensive data augmentation for robust performance.  
✅ End-to-end object detection pipeline using **YOLOv5**.

---

## 🛠 Workflow

![workflow chart](https://github.com/amandeep-yadav/Autonomous-Underwater-Vehicle-AUV-/blob/main/img/Untitled%20diagram-2024-12-22-124443.png)
---

## 📁 Directory Structure

```
AUV_Project/
├── data/                  # Dataset
│   ├── raw/              # Raw images and video frames
│   ├── augmented/        # Augmented images
│   └── annotations/      # YOLOv5 annotations
├── src/                  # Source code
│   ├── data_processing.py # Data preprocessing and augmentation scripts
│   ├── training.py       # YOLOv5 training pipeline
│   ├── inference.py      # Inference script
│   └── utils.py          # Utility functions
├── models/               # Pretrained and fine-tuned YOLOv5 models
├── results/              # Outputs and evaluation results
└── README.md             # Project documentation (this file)
```

---

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/AUV_Project.git
   cd AUV_Project
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLOv5**:
   ```bash
   git clone https://github.com/ultralytics/yolov5.git
   cd yolov5
   pip install -r requirements.txt
   cd ..
   ```

---

## 🎯 Usage

### 1️⃣ Data Preprocessing and Augmentation
Run the preprocessing script to augment your dataset:
```bash
python src/data_processing.py
```

### 2️⃣ Model Training
Train the YOLOv5 model:
```bash
python src/training.py --data data.yaml --epochs 50 --batch-size 16
```

### 3️⃣ Inference
Run inference on test images:
```bash
python src/inference.py --image test_image.jpg
```

---

## 🌟 Challenges and Solutions

### 🚧 Challenges
1. **Data Collection**:
   - Underwater conditions with varying lighting and visibility.
2. **Data Quality**:
   - Ensuring dataset diversity for robust detection.
3. **Model Generalization**:
   - Handling different underwater environments effectively.

### 🛠 Solutions
- Enhanced image clarity using OpenCV.  
- Applied extensive data augmentation for variability.  
- Fine-tuned YOLOv5 for better generalization.

---

## 🏆 Results

🎉 High accuracy achieved in:
- Detecting gates, flares, buckets, and stands with balls.  
- Successfully integrating the detection pipeline into the AUV system.

---

## 🔮 Future Improvements

🚀 **Planned Enhancements**:
- Real-time object tracking module.  
- Improved navigation algorithms for obstacle avoidance.  
- Extended detection capabilities for additional underwater objects and tasks.

---

## 👨‍💻 Contributors

- **AMANDEEP YADAV** (Project Lead)  

---

## 📜 License

This project is licensed under the **MIT License**. See the LICENSE file for details.

---


### Data Augmentation Examples
![Data Augmentation Examples](https://github.com/amandeep-yadav/Autonomous-Underwater-Vehicle-AUV-/blob/main/img/augmentatation.PNG)

### Task of AUV
**Detection and differention of object task (hit or avoid)**
<table>
  <tr>
    <td>
      <img src="https://github.com/amandeep-yadav/Autonomous-Underwater-Vehicle-AUV-/blob/main/img/1.PNG" alt="Image 1" width="200">
    </td>
    <td>
      <img src="https://github.com/amandeep-yadav/Autonomous-Underwater-Vehicle-AUV-/blob/main/img/2.PNG" alt="Image 2" width="200">
    </td>
    <td>
      <img src="https://github.com/amandeep-yadav/Autonomous-Underwater-Vehicle-AUV-/blob/main/img/3.PNG" alt="Image 3" width="200">
    </td>
    <td>
      <img src="https://github.com/amandeep-yadav/Autonomous-Underwater-Vehicle-AUV-/blob/main/img/4.PNG" alt="Image 4" width="200">
    </td>
  </tr>
</table>


