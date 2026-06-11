#  Heart Disease Prediction Model

> Predicting the presence of heart disease using Machine Learning — CodeAlpha Internship Task 2

---

##  Objective
Predict whether a person has **heart disease or not** based on medical data like age, blood pressure, cholesterol and other health parameters.

---

##  How It Works
- Heart disease dataset loaded and explored using **pandas**
- Missing values handled and data cleaned
- Data split into 80% training and 20% testing
- Three models trained and compared:
  - Random Forest Classifier
  - Logistic Regression
  - Decision Tree Classifier
- Best model selected automatically based on accuracy
- Predictions verified against actual dataset values

---

##  Tech Stack
- Python
- scikit-learn
- pandas
- matplotlib & seaborn

---

##  Project Structure
```
disease_prediction/
├── heart_disease_model.py     → Full ML pipeline
├── heart_disease.csv          → Dataset
├── confusion_matrix.png       → Model evaluation chart
├── feature_importance.png     → Top features chart



  # How to Run
```
pip install pandas scikit-learn matplotlib seaborn
python heart_disease_model.py
```

 📊 Results
| Model | Accuracy |
|-------|----------|
| Random Forest | 98.54% |
| Logistic Regression | 79.51% |
| Decision Tree | 98.54% |

✅ Best Model: Random Forest with 98.54% accuracy

---

 Sample Predictions
| Person | Details | Prediction |
|--------|---------|------------|
| Person 1 | 58yr Female | Heart Disease ❌ |
| Person 2 | 44yr Male | Heart Disease ❌ |
| Person 3 | 40yr Male | No Heart Disease ✅ |
| Person 4 | 55yr Male | Heart Disease ❌ |
