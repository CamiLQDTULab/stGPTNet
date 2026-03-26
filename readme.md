# 🧬 stGPTNet

## 1. Overview

**stGPTNet** is a supervised spatial representation learning framework that integrates:

- Pretrained transcriptomic foundation models (scGPT)  
- Spatial graph convolution  

to enable:

- Accurate spatial domain identification  
- Robust cross-dataset niche transfer  

---

## 2. Installation

### 🔧 Requirements

Make sure you have:

- Python ≥ 3.9  
- PyTorch  
- PyTorch Geometric  
- Transformers  
- Scikit-learn  
- NumPy, Pandas  

---

## 3. Dataset & Experiments

### 📊 Datasets

| Dataset | Description | Link |
|--------|------------|------|
| Maynard | Human spatial transcriptomics slices (benchmark for domain detection) | [Link](https://figshare.com/articles/dataset/10x_visium_datasets/22548901) |
| Xenium Breast Cancer | Human breast cancer (2 replicates) | [Link](https://drive.google.com/drive/folders/1fnSfIHzmOm-oiY2UVVLpM0xYDsvzaU5y) |
| CosMx NSCLC | Lung cancer dataset (2 donors) | [Link](https://drive.google.com/drive/folders/1YmBgICIxo6lf0bTiRvLBsBUTnVAIJAwd) |
| MOSTA Mouse | Mouse organogenesis (Stereo-seq) | [Link](https://slat.readthedocs.io/en/latest/tutorials/basic_usage.html) |

---

### 🧪 Experiments

| Task | Dataset | Train | Test | Section |
|------|--------|-------|------|--------|
| Spatial Domain Identification | Maynard | 1 slice (single) / multiple slices (full) in each group (Ex: train on slice 1515107) | Remaining slices in each group (Ex: test on slice 151508, 151509, 151510) | 2.2, 2.3 |
| Spatial Niche Transfer | Xenium | Replicate 1 | Replicate 2 | 2.4 |
| Spatial Niche Transfer | CosMx NSCLC | obs.batch['lung5_rep1'] | obs.batch['lung6'] | 2.4 |
| Spatial Domain Identification | MOSTA Mouse | E15.5-s1 | E15.5-s2 | 2.5 |

---

### 📌 Training Settings

- **stGPTNet-single**: train on 1 slice → test on remaining slices  
- **stGPTNet-full**: train on multiple slices → test on held-out slice  

---
## 💻 Code

This repository includes implementations of both **classical machine learning methods** and **graph neural network (GNN) models** used in our experiments.

---

### 🔹 Classical Machine Learning Methods

**File:** `ML_spot_classification.ipynb`

The following methods are implemented:

- K-Nearest Neighbors (**KNN**)  
- Decision Tree  
- Naive Bayes  
- Stochastic Gradient Descent (**SGD**)  
- XGBoost  
- LightGBM  

---

### 🔹 Graph Neural Network Methods

**File:** `GNN_spot_classification.ipynb`

The following GNN models are implemented:

- APPNP  
- GCN  
- GAT  
- GIN  
- GraphSAGE  

---

### 📌 Notes

- All methods are evaluated under the same experimental settings for fair comparison  
- Evaluation metrics include:
  - **ARI, NMI**
  - **F1-score, Accuracy, Precision, Recall**  
- The code reproduces results corresponding to Sections **2.2 → 2.5**  

