# project-folder
AIML ASSIGNMENT 02
# ML Assignment 2 – Multiple Classifiers + Streamlit App

## a) Problem Statement
Develop and compare multiple machine‑learning classification models on a single public dataset, evaluate them using standard performance metrics, and demonstrate their behavior through an interactive Streamlit web application. The application supports test data upload, model selection, metric visualization, and display of confusion matrix or classification report.

---

## b) Dataset Description
**Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset  
**Source:** UCI Machine Learning Repository

- Classification Type: Binary classification (Malignant vs Benign)
- Number of Instances: 569
- Number of Features: 30 numeric features
- Description: Features are computed from digitized images of fine needle aspirate (FNA) of a breast mass and describe characteristics of the cell nuclei.

**Target Encoding Used:**
- `1` → Malignant
- `0` → Benign

**Dataset References:**
- UCI Repository: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic  
- scikit-learn loader: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

---

## c) Models Used and Evaluation Metrics

### Implemented Classification Models
1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (kNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

### Evaluation Metrics Calculated
- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

### Model Comparison Table (Holdout Test Set)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| kNN | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Random Forest (Ensemble) | 0.9737 | 0.9944 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |
| XGBoost (Ensemble) | 0.9649 | 0.9924 | 1.0000 | 0.9048 | 0.9500 | 0.9258 |

---

## d) Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|--------------|-------------------------------------|
| Logistic Regression | Strong baseline model with excellent AUC and balanced precision–recall performance. |
| Decision Tree | Captures non-linear relationships but may overfit when used as a single model. |
| kNN | Sensitive to feature scaling and distance metrics; performs well when classes are clearly separated. |
| Naive Bayes | Very fast and simple baseline; independence assumption can limit performance with correlated features. |
| Random Forest (Ensemble) | Best overall performance with highest accuracy and MCC due to ensemble averaging and robustness. |
| XGBoost (Ensemble) | Boosted trees provide strong performance by sequentially correcting errors from previous models. |

---

## Project Structure
