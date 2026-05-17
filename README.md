# Blood Smear Sickle Cell Classification

Classical machine learning pipeline for automated sickle cell disease 
detection from peripheral blood smear images, applied to Ugandan clinical 
microscopy data.

**Author:** Okidi Patrovas Gaabriel | 2025/HD07/26020U
**Institution:** Makerere University, Kampala, Uganda
**Course:** MSB7215: Machine Learning in Biomedicine

## Research Gap

Sickle cell disease is one of the most prevalent genetic disorders in 
Uganda, yet it remains one of the most underdiagnosed. The sickle cell 
trait affects 13.3% of the national population and reaches 19.8% in 
Kampala alone. An estimated 15,000 to 20,000 children are born with the 
disease in Uganda each year. Many of these children never receive a formal 
diagnosis because current diagnostic methods depend entirely on trained 
laboratory scientists, haemoglobin electrophoresis equipment, and manual 
microscopic examination of blood smears — resources concentrated in urban 
hospitals and largely absent from rural district health facilities.

Without early diagnosis, children with sickle cell disease miss 
prophylactic treatment that prevents life-threatening crises, organ 
damage, and premature death. The diagnostic gap is not a medical problem 
— it is an access problem. The knowledge to diagnose sickle cell disease 
exists. The tools to bring that knowledge to rural Uganda do not.

Automated image analysis using machine learning offers a path to closing 
this gap. A system that can classify a blood smear photograph taken with 
a mobile phone on a basic microscope could enable community health workers 
with no laboratory training to identify patients who need urgent referral. 
However, for such a system to be deployable in rural Ugandan health 
facilities, it must run on standard hardware without GPU infrastructure.

Prior machine learning studies on sickle cell detection used datasets 
collected under controlled laboratory conditions in high-income countries. 
None had applied classical machine learning to the Tushabe et al. (2024) 
dataset — the only published dataset of sickle cell blood smear images 
collected from Ugandan patients using mobile phone microscopy under real 
clinical conditions. This project addresses that gap directly.

## Research Objective

Sickle cell disease remains largely undiagnosed in rural Uganda due to 
the absence of trained laboratory personnel and specialist diagnostic 
equipment. This project develops an automated sickle cell detection 
system from peripheral blood smear images captured using mobile phone 
cameras on basic microscopes, with the goal of enabling non-specialist 
health workers in rural Ugandan settings to screen patients for sickle 
cell disease without requiring laboratory expertise.

The specific objectives are:

1. Build an automated classifier that can distinguish sickle cell 
   positive blood smears from normal smears using images captured 
   under real Ugandan clinical conditions
2. Apply classical machine learning methods that can run without GPU 
   infrastructure, making the system deployable in low-resource 
   health facilities
3. Evaluate whether the system achieves clinically meaningful 
   sensitivity — ensuring missed sickle cell cases are minimised
4. Assess model reliability through overfitting analysis and 
   cross-validation to ensure results generalise beyond the 
   training data
5. Compare performance against deep learning approaches to 
   understand the trade-off between computational cost and 
   diagnostic accuracy

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
