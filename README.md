# Blood Smear Sickle Cell Classification

Classical machine learning pipeline for automated sickle cell disease 
detection from peripheral blood smear images, applied to Ugandan clinical 
microscopy data.

**Author:** Okidi Patrovas Gaabriel | 2025/HD07/26020U
**Institution:** Makerere University, Kampala, Uganda
**Course:** MSB7215: Machine Learning in Biomedicine

## Research Gap

Sickle cell disease affects 13.3% of the Ugandan population, yet diagnosis 
currently requires trained laboratory scientists, specialist microscopes, 
and manual microscopic examination — resources that are largely unavailable 
in rural Ugandan settings. Patients in rural areas may go years without a 
formal diagnosis, missing early treatments that prevent life-threatening 
crises.

While deep learning has been applied to blood smear image classification, 
no prior study had applied classical machine learning to the Tushabe et al. 
(2024) Ugandan clinical mobile-phone microscopy dataset. Furthermore, no 
study had examined whether classical ML — which requires no GPU 
infrastructure — could produce clinically useful results on this dataset, 
making it accessible to health facilities that cannot support deep learning 
deployment.

## Research Objective

This project applies classical machine learning with HOG feature extraction 
to automate sickle cell detection from peripheral blood smear images captured 
using mobile phone cameras on basic microscopes in Uganda. The objective is 
to determine whether classical ML can produce clinically meaningful results 
on this task and to compare its performance against deep learning models 
applied to the same dataset in a prior study.

The specific objectives are:

1. Extract numerical features from blood smear images using HOG feature 
   extraction to create a tabular features matrix suitable for classical ML
2. Train and evaluate SVM and Random Forest classifiers using standard 
   supervised learning practice including preprocessing pipelines, 
   stratified train-test splitting, and cross-validation
3. Optimise both models using GridSearchCV and analyse overfitting
4. Compare classical ML performance against deep learning models from a 
   prior study on the same dataset
5. Assess the viability of classical ML as a low-resource alternative 
   to deep learning for sickle cell screening in Uganda

## Project Overview

This project applies classical machine learning — SVM and Random Forest — 
to classify peripheral blood smear images as sickle cell positive or normal, 
using HOG feature extraction to convert images into a tabular features matrix 
of 8100 numerical features per image.

Deep learning significantly outperforms classical ML on this task, with 
ResNet-50 achieving 98.57% accuracy compared to SVM at 91.98%. However 
classical ML remains a viable alternative in low-resource settings where 
GPU infrastructure is unavailable. Deep learning results are from a prior 
study on the same dataset available at 
https://github.com/Patro331/sickle-cell-detection

## Datasets

- Tushabe et al. (2024) Ugandan Sickle Cell Microscopy Dataset — 422 
  sickle cell images and 147 normal images collected from patients in 
  Soroti and Kumi districts, Uganda using mobile phone cameras on basic 
  microscopes. Available at: https://www.kaggle.com/datasets/florencetushabe/sickle-cell-disease-dataset

- BCCD Dataset (Shenggan et al.) — 364 normal blood smear images used 
  to supplement the negative class. MIT licence. Available at: 
  https://github.com/Shenggan/BCCD_Dataset

Combined raw total: 933 images. Split: 746 training / 187 test (80/20 stratified).

## Results

### Classical ML Models

| Model | Test Accuracy | AUC-ROC | Sensitivity | Specificity | Errors |
|---|---|---|---|---|---|
| SVM (Optimised) | 91.98% | 0.9651 | 92.94% | 91.18% | 15/187 |
| Random Forest (Optimised) | 87.17% | 0.9396 | 96.47% | 79.41% | 24/187 |

### Classical ML vs Deep Learning

| Model | Type | Accuracy | AUC-ROC | Sensitivity |
|---|---|---|---|---|
| SVM | Classical ML | 91.98% | 0.9651 | 92.94% |
| Random Forest | Classical ML | 87.17% | 0.9396 | 96.47% |
| Baseline CNN | Deep Learning* | 87.86% | 0.9588 | 95.24% |
| EfficientNet-B0 | Deep Learning* | 96.43% | 0.9951 | 95.24% |
| ResNet-50 | Deep Learning* | 98.57% | 0.9973 | 98.41% |

*Deep learning results from: Okidi Patrovas Gaabriel (2026). 
https://github.com/Patro331/sickle-cell-detection

Deep learning with transfer learning clearly outperforms classical ML. 
The best classical ML model, SVM, achieves 91.98% accuracy compared to 
ResNet-50 at 98.57%. Classical ML remains a viable option in settings 
where GPU infrastructure is unavailable.

## Project Structure

- notebooks/ — 7 Jupyter notebooks covering the full ML pipeline
- src/ — Python source modules
- figures/ — Plots and visualisations
- reports/ — Research paper and presentation slides
- data/ — Dataset placeholder (see datasets section for download links)

## Notebooks

- 01_data_understanding.ipynb — Dataset exploration and variable analysis
- 02_feature_extraction.ipynb — HOG feature extraction from images
- 03_preprocessing_pipeline.ipynb — StandardScaler pipeline, train-test split
- 04_svm_classifier.ipynb — SVM training, evaluation and optimisation
- 05_random_forest_classifier.ipynb — Random Forest training and optimisation
- 06_model_comparison.ipynb — Classical ML model comparison
- 07_ml_vs_dl_comparison.ipynb — Classical ML vs Deep Learning comparison

## Setup Instructions

Clone this repository using:
git clone https://github.com/Patro331/blood-smear-sickle-cell-classification.git

Install dependencies:
pip install -r requirements.txt

Download the datasets from the links above and place sickle cell images 
in data/raw/positive/ and normal images in data/raw/negative/. Run the 
notebooks in order from 01 through 07.

## Research Paper

Full research paper available in the reports/ folder.

## Key Limitations

Both classical ML models show overfitting with 100% training accuracy 
and lower test accuracy. The dataset of 933 images is small by machine 
learning standards. A domain gap exists between Tushabe mobile-phone 
images and BCCD laboratory images within the negative class. Deep 
learning significantly outperforms classical ML and should be the 
preferred approach where GPU infrastructure is available.
