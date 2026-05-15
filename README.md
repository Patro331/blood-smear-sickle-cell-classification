# Blood Smear Sickle Cell Classification

Classical machine learning pipeline for automated sickle cell disease 
detection from peripheral blood smear images, applied to Ugandan clinical 
microscopy data.

**Author:** Okidi Patrovas Gaabriel | 2025/HD07/26020U
**Institution:** Makerere University, Kampala, Uganda
**Course:** MSB7215: Machine Learning in Biomedicine

## Project Overview

Sickle cell disease affects 13.3% of the Ugandan population. This project 
applies classical machine learning — specifically SVM and Random Forest — 
to classify peripheral blood smear images as sickle cell positive or normal, 
using HOG feature extraction to convert images into a tabular features matrix.

## Datasets

- Tushabe et al. (2024) Ugandan Sickle Cell Microscopy Dataset
- BCCD Dataset (Shenggan et al.) — normal blood smear images

Combined raw total: 933 images

## Project Structure

- notebooks/ — Jupyter notebooks following the ML pipeline
- src/ — Python source modules
- figures/ — Plots and visualisations
- reports/ — Research paper
- data/ — Dataset placeholder (see datasets section for download links)

## Notebooks

- 01_data_understanding.ipynb — Dataset exploration and variable understanding
- 02_feature_extraction.ipynb — HOG feature extraction from images
- 03_preprocessing_pipeline.ipynb — Numeric pipeline with StandardScaler
- 04_svm_classifier.ipynb — SVM training and evaluation
- 05_random_forest_classifier.ipynb — Random Forest training and evaluation
- 06_optimisation_and_evaluation.ipynb — GridSearchCV and model comparison

## Results

To be updated upon project completion.
