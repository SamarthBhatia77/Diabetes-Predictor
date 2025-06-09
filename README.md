# Diabetes Predictor Using Logistic Regression
A Machine Learning model that predicts whether an individual is diabetic or not, based on medical attributes such as Glucose, BMI, Age, and more.

##  Project Overview

This is a beginner-friendly machine learning classification project where we aim to predict whether an individual is diabetic or not based on their medical diagnostic measurements.

Using the **Pima Indians Diabetes Dataset**, we build a logistic regression model to classify individuals as diabetic (1) or non-diabetic (0). This project introduces core ML concepts such as data preprocessing, feature scaling, model building, evaluation, and prediction.

##  Aim

To develop a supervised machine learning model using Logistic Regression that can predict the likelihood of diabetes in a person using medical features like:

- Glucose level
- Blood Pressure
- BMI
- Insulin
- Age, etc.

##  Methodology

The project follows the typical end-to-end ML workflow:

1. **Data Loading**  
   The dataset is loaded using `pandas` from a CSV file.

2. **Data Cleaning & Preprocessing**
   - Handled zero values in critical columns like Glucose and BloodPressure.
   - Converted all missing/zero values into valid ranges or removed them as appropriate.
   - Scaled features using `StandardScaler` to ensure better model performance.

3. **Exploratory Data Analysis (EDA)**
   - Used `seaborn` and `matplotlib` for visualizations.
   - Checked feature correlation to identify influential predictors.

4. **Model Building**
   - Used `LogisticRegression` from `sklearn.linear_model`.
   - Split dataset into training and testing sets using `train_test_split`.

5. **Model Evaluation**
   - Evaluated performance using accuracy, confusion matrix, and classification report.
   - Achieved ~76% accuracy on the test dataset.

6. **Prediction Testing**
   - Demonstrated the model's ability to predict outcome for specific rows from the dataset.
   - Used `.iloc[[index]]` and `scaler.transform()` to predict on individual samples.

##  Results

- **Model Used**: Logistic Regression
- **Accuracy Achieved**: ~76%
- **Evaluation Metrics**: Accuracy Score, Confusion Matrix, Classification Report

The model performs reasonably well for a basic logistic regression implementation, considering no hyperparameter tuning or advanced algorithms were applied.

##  Insights

- Features like Glucose and BMI have strong correlation with the likelihood of diabetes.
- Zero or missing values in critical medical data can significantly affect model accuracy.
- Feature scaling greatly improves the logistic regression model's performance.

##  Limitations

- The dataset has some imbalance between diabetic and non-diabetic cases.
- Logistic Regression assumes linearity between features and the target — may not capture complex relationships.
- Zero values in medical fields like Glucose and Insulin are likely missing data and need careful handling.

---

##  Future Improvements

- Try more complex models like Random Forest, SVM, or Gradient Boosting for better performance.
- Use GridSearchCV for hyperparameter tuning.
- Build a basic web app using **Streamlit** or **Flask** to make real-time predictions with user input.
- Add cross-validation and ROC curve analysis for better evaluation.

##  Resources

- **Dataset**: [Pima Indians Diabetes Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Libraries Used**:
  - `pandas`, `numpy` for data manipulation
  - `matplotlib`, `seaborn` for EDA and visualization
  - `scikit-learn` for modeling and evaluation

##  Author

**Samarth Bhatia**  
BITS Pilani, Hyderabad Campus
