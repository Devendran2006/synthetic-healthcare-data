# Privacy-Preserving Synthetic Healthcare Data Generation using CTGAN

## Project Overview

Healthcare organizations often face challenges in sharing patient data due to privacy regulations and security concerns. This project addresses the problem by detecting and removing Personally Identifiable Information (PII) and generating realistic synthetic healthcare records using CTGAN (Conditional Tabular Generative Adversarial Network).

The generated synthetic data preserves the statistical characteristics of the original dataset while protecting patient privacy.

---

## Objectives

* Detect and remove sensitive patient information.
* Perform healthcare data anonymization.
* Train a CTGAN model on anonymized healthcare data.
* Generate realistic synthetic healthcare records.
* Evaluate the quality of synthetic data.
* Validate utility using machine learning models.
* Build an automated pipeline for future healthcare datasets.

---

## Dataset

Healthcare Dataset

Records: 55,500+

Attributes:

* Age
* Gender
* Blood Type
* Medical Condition
* Insurance Provider
* Billing Amount
* Admission Type
* Medication
* Test Results
* Length of Stay

Sensitive fields such as patient names, doctors, hospitals, and identifiers are anonymized before model training.

---

## Project Architecture

Raw Healthcare Data

↓

PII Detection

↓

Data Anonymization

↓

Preprocessing

↓

CTGAN Training

↓

Synthetic Data Generation

↓

Evaluation

↓

Machine Learning Validation

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* SDV
* CTGAN
* Git
* GitHub

---

## Results

### Real Dataset Accuracy

29.58%

### Synthetic Dataset Accuracy

28.97%

### Utility Retention

Approximately 98%

The synthetic dataset preserves most of the predictive power of the original dataset while ensuring privacy protection.

---

## Project Structure

project/

├── data/

├── src/

│ ├── pii_detection.py

│ ├── preprocessing.py

│ ├── ctgan_train.py

│ ├── evaluation.py

│ ├── ml_validation_real.py

│ └── ml_synthetic.py

├── outputs/

├── requirements.txt

└── README.md

---

## Future Enhancements

* Automated CI/CD Pipeline
* Streamlit Dashboard
* Docker Deployment
* MIMIC-IV Integration
* ECG Synthetic Data Generation
* Medical Image Synthetic Generation
* Synthetic Digital Patient Framework

---

## Conclusion

This project demonstrates how synthetic healthcare data can be generated while preserving patient privacy. The solution enables healthcare institutions, researchers, and AI developers to use realistic healthcare datasets without exposing sensitive patient information.
