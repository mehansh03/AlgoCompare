# AlgoCompare

**AlgoCompare** is a Flask-based web application that allows users to upload a dataset, select features, and compare predictions from multiple machine learning models in real-time. It's designed to provide a visual and intuitive way to understand how different algorithms behave on the same data.

---

##  Features

- Upload CSV datasets through a simple and clean UI
- Basic preprocessing and feature selection
- Simultaneous execution of multiple machine learning models
- Compare predictions from:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - XGBoost *(optional depending on installed packages)*

---

## Tech Stack

- **Backend**: Flask (Python)
- **ML Models**: Scikit-learn, XGBoost
- **Frontend**: HTML + Jinja2 Templates + (Optional) Bootstrap

---

## Example Workflow

1. **Upload** a `.csv` file containing your dataset.
2. **Select** input features and the target column from dropdowns.
3. Click **"Compare"** to run all ML models.
4. View predictions and outputs side-by-side from all supported models.

