# Machine Learning Practices

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A structured, production-ready repository containing curated machine learning implementations, mathematical foundations, data preprocessing recipes, and deep learning architectures.

---

## Repository Structure

```text
Machine-Learning-Practices/
├── 01_eda_and_feature_engineering/   # Data profiling, missing value imputation, outlier handling, scaling
│   ├── eda_pipeline.py
│   ├── feature_engineering_utils.py
│   └── README.md
├── 02_supervised_learning/           # Classical ML: Regression, classification, ensemble models
│   ├── regression_models.py          # OLS, Ridge, Lasso, ElasticNet
│   ├── classification_models.py      # Logistic Regression, Decision Tree, SVM, KNN
│   ├── ensemble_methods.py           # Random Forest, Extra Trees, Gradient Boosting
│   └── README.md
├── 03_unsupervised_learning/         # Clustering & dimensionality reduction
│   ├── clustering_algorithms.py      # K-Means (Elbow & Silhouette), DBSCAN
│   ├── pca_dimensionality_reduction.py # PCA explained variance & 2D projection
│   └── README.md
├── 04_deep_learning/                 # PyTorch architectures & neural network training
│   ├── pytorch_mlp.py                # Multi-Layer Perceptron with custom DataLoader
│   ├── pytorch_cnn.py                # 2D Convolutional Neural Network
│   └── README.md
├── utils/                            # Shared utilities
│   ├── __init__.py
│   ├── metrics.py                    # Classification & regression evaluation metrics
│   └── visualization.py              # Confusion matrix & loss curve plotting
├── notebooks/                        # Interactive Jupyter notebooks
│   └── quickstart_exploration.ipynb
├── data/                             # Dataset storage
│   ├── raw/                          # Untouched raw data (.gitignored)
│   └── processed/                    # Cleaned and transformed datasets (.gitignored)
├── models/                           # Serialized model checkpoints (.gitignored)
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git exclusion rules
└── README.md
```

---

## Getting Started

### 1. Clone or Open the Repository
```bash
git clone https://github.com/yadavshrey681-cell/Machine-Learning-Practices.git
cd Machine-Learning-Practices
```

### 2. Set Up a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running Practice Modules

### 📊 1. Exploratory Data Analysis & Feature Engineering
```bash
# Run automated EDA report on sample dataset
python 01_eda_and_feature_engineering/eda_pipeline.py

# Test missing value imputation and IQR outlier handling
python 01_eda_and_feature_engineering/feature_engineering_utils.py
```

### 🎯 2. Supervised Learning
```bash
# Compare OLS, Ridge, Lasso, and ElasticNet
python 02_supervised_learning/regression_models.py

# Compare Logistic Regression, Decision Trees, SVM, and KNN
python 02_supervised_learning/classification_models.py

# Compare Random Forest vs Gradient Boosting feature importances
python 02_supervised_learning/ensemble_methods.py
```

### 🔍 3. Unsupervised Learning
```bash
# K-Means clustering (elbow & silhouette analysis) and DBSCAN
python 03_unsupervised_learning/clustering_algorithms.py

# Principal Component Analysis (PCA) dimensionality reduction
python 03_unsupervised_learning/pca_dimensionality_reduction.py
```

### 🧠 4. Deep Learning (PyTorch)
```bash
# Train a Multi-Layer Perceptron (MLP) with validation tracking
python 04_deep_learning/pytorch_mlp.py

# Inspect and test Convolutional Neural Network (CNN) architecture
python 04_deep_learning/pytorch_cnn.py
```

### 📓 5. Interactive Notebooks
Launch Jupyter Lab or Notebook to run interactive experiments:
```bash
jupyter notebook notebooks/quickstart_exploration.ipynb
```

---

## Syncing with GitHub

To push changes to your GitHub repository:
```bash
git add .
git commit -m "feat: complete initial machine learning practice scaffold"
git push -u origin main
```

---

## License
This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).
