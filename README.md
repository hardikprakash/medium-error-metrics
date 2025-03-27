# ML Error Metrics Analysis

A comprehensive toolkit for evaluating machine learning models through essential error metrics for both classification and regression tasks.

## 📊 Project Overview

This repository contains Jupyter notebooks demonstrating how to properly evaluate machine learning models by implementing and visualizing various error metrics. It provides practical examples for comparing model performance and making informed decisions.

## 📁 Repository Structure

- **notebooks/**
  - `classification.ipynb` - Evaluation metrics for classification models
  - `regression.ipynb` - Evaluation metrics for regression models

## 🔍 Classification Metrics

The classification notebook demonstrates:

- Accuracy Score
- Precision & Recall
- F1 Score
- Confusion Matrix
- ROC-AUC Curve
- Precision-Recall Curve

### Usage Example

```python
print_classification_metrics("Logistic Regression", y_test, y_pred_log)
print_classification_metrics("Random Forest Classifier", y_test, y_pred_rf)

📈 Regression Metrics
The regression notebook covers:

Mean Absolute Error (MAE)
Root Mean Square Error (RMSE)
R-squared (R²)
Residual plots
Usage Example

```python
# For Random Forest Regressor
mae_rf = mean_absolute_error(y_test, y_pred)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred))
r2_rf = r2_score(y_test, y_pred)

print(f"MAE: {mae_rf:.2f}")
print(f"RMSE: {rmse_rf:.2f}")
print(f"R²: {r2_rf:.2f}")
```

🛠️ Requirements
Python 3.x
scikit-learn
numpy
pandas
matplotlib
seaborn

🚀 Getting Started
Clone this repository
Install dependencies:
`pip install -r requirements.txt`
