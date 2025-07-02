# 🌸 Iris Flower Classification — CodeAlpha Internship Project

This project is submitted as part of the **CodeAlpha Data Science Internship**. It involves classifying Iris flowers into one of three species using a machine learning model trained on petal and sepal measurements.

---

## 📌 Project Objective

To build a machine learning model that can classify Iris flowers into the following species:
- Setosa
- Versicolor
- Virginica

The classification is based on:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

---

## 🛠️ Tools & Libraries Used

- **Python** (v3.x)
- **Pandas** – Data loading and manipulation
- **Seaborn / Matplotlib** – Data visualization
- **Scikit-learn** – Machine learning model (Random Forest Classifier)
- **LabelEncoder** – For encoding string labels to numeric form

---

## 🧠 Workflow

1. **Load Dataset**
   - Loaded from: [Iris Dataset GitHub](https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv)

2. **Data Preprocessing**
   - Checked for null values
   - Encoded species labels using `LabelEncoder`

3. **Data Visualization**
   - Used `seaborn.pairplot()` to show feature distribution and class separation
   - Displayed feature importance using `RandomForestClassifier`

4. **Model Training**
   - Used `train_test_split` to divide dataset into training and test sets
   - Trained a **Random Forest Classifier**

5. **Evaluation**
   - Accuracy Score
   - Classification Report
   - Confusion Matrix

---

## 📊 Output Snapshots

### 🔹 Pairplot
Shows how the species are clustered based on combinations of features.

![Pairplot](https://github.com/yourusername/CodeAlpha_IrisClassification/blob/main/pairplot.png)

### 🔹 Confusion Matrix & Accuracy
Model accurately classifies most of the test data.

### 🔹 Feature Importance
Highlights which features are most important in classification.

---

## 📁 Files Included

- `iris_classification.py` — Main code file
- `README.md` — This explanation
- (Optional) `pairplot.png` — Save your plot image if submitting visuals

---

## ✅ How to Run

1. Clone or download this repository
2. Install required libraries:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn
