# Diabetes Prediction Using AI & Machine Learning

A small end-to-end ML project that predicts diabetes (Outcome: 1/0) using the **Pima Indians Diabetes dataset**.  
Includes data loading, EDA visualizations, training/evaluating three models (KNN, Logistic Regression, Random Forest), saving models, and a simple Flask web UI for predictions.

## Dataset
- Source (UCI / Kaggle mirror): Pima Indians Diabetes Database  
- 768 female patients, Pima Indian heritage, age ≥ 21  
- Target: **Outcome** (1 = diabetic, 0 = not diabetic)

**Features**
Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age.

**Reference**
Smith et al. (1988). *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus.*

## What’s in this notebook
1. **Load Data**
   - Reads CSV from:
     `https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv`
2. **Exploratory Data Analysis**
   - Outcome distribution
   - Correlation heatmap
   - Pairplot (selected features)
   - Age vs Glucose scatter
3. **Preprocessing**
   - Train/Test split (80/20, stratified)
   - Standard scaling (important for KNN)
4. **Models**
   - **KNN** with best-k search (1..20)
   - **Logistic Regression**
   - **Random Forest**
5. **Evaluation**
   - Accuracy, Precision, Recall, F1, ROC-AUC
   - Classification report
   - Confusion matrix
6. **Model Saving**
   - `diabetes_knn_model.joblib`
   - `diabetes_logistic_regression_model.joblib`
   - `diabetes_rf_model.joblib`
7. **Prediction Demo**
   - Predicts Outcome for a sample input using all models
8. **Flask App**
   - Simple form UI to submit features and show predictions.

## Results (from your run)
- **KNN (best k = 12)**  
  Accuracy 0.773 | Precision 0.721 | Recall 0.574 | F1 0.639 | ROC-AUC 0.782
- **Logistic Regression**  
  Accuracy 0.714 | Precision 0.609 | Recall 0.519 | F1 0.560 | ROC-AUC 0.823
- **Random Forest**  
  Accuracy 0.760 | Precision 0.681 | Recall 0.593 | F1 0.634 | ROC-AUC 0.815

## How to Run (Colab / Notebook)
1. Open the notebook in Google Colab.
2. Run all cells top-to-bottom.
3. Models will be trained, evaluated, and saved as `.joblib`.

## How to Run the Flask App Locally
> The notebook includes Flask code. To run locally, copy the Flask section into a `app.py` and create a basic `templates/index.html`.

1. Install dependencies:
   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn joblib flask
   ```
2. Run the app:
   ```bash
   python app.py
   ```
3. Open in browser:
   ```bash
   http://127.0.0.1:5000/
   ```
   
## Notes / Limitations
- Dataset contains zero values in medical fields (e.g., Glucose, BMI) that may represent missing data; this notebook does not impute them.
- Performance is reasonable for a class project, but not clinical-grade.
- Flask route currently uses a hardcoded example input; update to use real form values for true interactive predictions.

## Authors
- Chetan Kapadia (801438508)
- Sanjyot Sathe (801426514)
- Mohan Krishna Otikunta (801418781)
- Yashodhan Rajesh Jaltare (801430080)
- Ravi Kumar (801304869)
