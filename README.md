# Image-Matching-Challenge-3D-Reconstruction

This repository contains an end‑to‑end TensorFlow pipeline developed for the “Image Matching Challenge 2025” Kaggle competition. The goal is to cluster mixed test images into their underlying scenes (or flag them as outliers) and reconstruct each camera’s 6‑DOF pose (rotation + translation) to build accurate 3D maps across diverse environments—from phototourism and historical sites to night‑time and aerial imagery.

## Objectives

- **Scene Clustering:** Group images into clusters corresponding to real scenes and detect outliers.
- **Pose Estimation:** Predict a rotation matrix and translation vector for each image.
- **Robust Evaluation:** Train with custom loss functions for quaternion, translation, and camera‑center errors.
- **Submission Preparation:** Generate a competition‑ready CSV file with the required metrics.

## Repository Structure
```bash
root/
├── config.py             # Global variables and hardware configuration
├── requirements.txt      # Project dependencies
├── main.py               # Main script of project for preprocessing, training, infrence and clustering
├── src/                  # Folder with modules of project
│   ├── preprocessing.py  # Utils: format adaptation, quaternions, statistics
│   ├── dataset.py        # tf.data.Dataset creation and data augmentation
│   ├── model_architecture.py # Definition of backbone and heads, Model subclass
│   ├── losses.py         # Custom loss implementations (Quaternion, Translation, CameraCenter)
│   ├── metrics.py        # Custom metrics (cosine similarity, MAE)
│   ├── training.py       # Callbacks, schedulers, distributed training logic
│   ├── inference.py      # Inference functions and prediction postprocessing
│   ├── clustering.py     # DBSCAN algorithm over features and poses
│   ├── utils_io.py       # File I/O and submission CSV creation
│   └── plot_results.py   # Curve and cluster visualization functions
└── outputs/              # Output folder: exported models, plots, and CSV
```

## Main Features

- **Modular Design:** Each component (preprocessing, model, training) is isolated to facilitate testing and maintenance.
- **TensorFlow Subclassing:** Model subclass customizes train_step and test_step, with advanced metrics and losses.
- **Distributed Training:** CPU, GPU, and TPU compatibility via tf.distribute.
- **Adaptive Losses:** Custom loss functions that filter NaNs and normalize across global batches and replicas.
- **Callbacks & Schedulers:** EarlyStopping, CosineDecay with warm-up, and two fine‑tuning phases.
- **Reproducibility:** requirements.txt with exact versions and fixed random seeds.


## Features

- **Data Preprocessing & Augmentation**
  - Parses CSV labels & thresholds into rotation matrices, translation vectors, camera centers and difficulty scores
  - Stratified train/validation split (80/20) and “easy” subset definition for scenes with low difficulty (for curriculum learning)
  - Real‑time `tf.data` augmentation: brightness, contrast, saturation, hue
  
- **Custom Multi‑Head ConvNeXt Architecture**
  - **Backbone**: ConvNeXt-Large (ImageNet‐pretrained, fine‑tuned in stages)
  - **Heads**:
    - Quaternion regressor
    - Translation regressor
    - Outlier classifier

- **Adaptive Training Schedule**
  - Warm‑up phase on “easy” scenes → full dataset training → two stages of backbone fine‑tuning (20% and 40% of layers)
  - Cosine‐decay learning rate with configurable warm‑up
  - EarlyStopping to restore best weights

- **Custom Losses**  
  - **QuaternionAngularLoss** (1 – |y_true·y_pred|) (For unitary vectors, Cosine Similarity is equivalent to dot product)
  - **TranslationLoss** (MSE on denormalized translation)
  - **CameraCenterLoss** (MSE on camera centers C = –Rᵀ·T)

- **Custom Metrics**  
  - **QuaternionCosineSimilarityMetric** (|y_true·y_pred|) (For unitary vectors, Cosine Similarity is equivalent to dot product)
  - **TranslMAEMetric** (MAE on translation)
  - **BinaryAccuracy** (Binary Accuracy on outliers)

- **Clustering & Inference**
  - Extracts global image features via ConvNeXt backbone
  - DBSCAN over normalized [quaternion, translation, visual] vectors
  - Retains model‑predicted outliers

- **Output Files**
  - Outputs `submission.csv` for Kaggle
  - Generates training‐history and loss‑curve PNGs
  - Saves a copy of:
    - StandardScaler applied to translation vectors
    - Backbone
    - Full model
    - Predictions
  

## Technologies Used

- **Python 3.10+**  
- **TensorFlow & Keras** (v2.16) for model building, custom training loops and mixed precision  
- **TensorFlow Graphics** for quaternion→rotation conversion  
- **Scikit‑learn** for data splits, standard scaling & DBSCAN  
- **SciPy** for rotation/quaternion utilities  
- **OpenCV & Matplotlib** for image I/O and plotting  
- **Pandas & NumPy** for data manipulation  
- **Joblib** for scaler persistence  


## Prerequisites

1. **Download Competition Data**  
   - Place `train_labels.csv`, `train_thresholds.csv` and `train/`, `test/` folders under `INPUT_PATH` (default: `./input/`)  

2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

## Installation & Execution
```bash
# Clone and enter repo
git clone https://github.com/your‑username/Image‑Matching‑Challenge‑3D‑Reconstruction.git
cd Image‑Matching‑Challenge‑3D‑Reconstruction


# Install required packages
pip install -r requirements.txt

# Run end‑to‑end pipeline
python main.py
```

The script will:
1. Configure TPU/GPU/CPU strategy
2. Load & preprocess labels + images
3. Build & train the multi‑head ConvNeXt model
4. Save training history plots under ./working/ directory
5. Run inference, cluster test images, and generate submission.csv

## Important Considerations
- Scaling & Normalization: Translation vectors are standardized per‑inlier then inverted at inference
- Outliers Handling: Images labeled as outliers by the model are force‑assigned cluster -1
- Reproducibility: Random seeds set for data splits; mixed‑precision policy may introduce minor variance

## Acknowledgments
This implementation was created for the “Image Matching Challenge 2025” Kaggle competition. 
Further improvements such as RANSAC + Horn’s alignment, and bundle adjustment, are expected to be applied for next versions.

## License
Distributed under the MIT License. See the LICENSE file for more information.

## Author
José Andrés Lorenzo. https://github.com/joseandres94
