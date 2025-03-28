# Customer Churn Prediction Using Random Forest Classifier

![GitHub](https://img.shields.io/badge/Language-Python-blue)
![GitHub](https://img.shields.io/badge/Library-Scikit_Learn-orange)

A step-by-step machine learning tutorial to predict telecom customer churn with actionable business insights.

## 📁 Repository Structure

- LICENSE: MIT License file

- README.md: Project documentation

- WA_Fn-UseC_Telco-Customer-Churn.csv: Raw dataset from Kaggle

- codefile-1.ipynb: Jupyter Notebook with full implementation

- machinelearning-23113165.pdf: Detailed project report



## 🌟 Project Overview
This project demonstrates how machine learning (Random Forest Classifier) can predict customer churn in the telecom industry. It covers:
- Complete data preprocessing pipeline
- Exploratory data analysis (EDA)
- Model training and evaluation
- Feature importance analysis
- Actionable business recommendations

## 🚀 Key Features
✔️ **Data Preprocessing**: Handling missing values, encoding categorical variables, and feature scaling  
✔️ **EDA**: Visualizations showing churn patterns and relationships with key features  
✔️ **Model Building**: Random Forest Classifier with hyperparameter tuning  
✔️ **Evaluation Metrics**: Accuracy (79.99%), Precision-Recall analysis, and confusion matrix  
✔️ **Business Insights**: Identified top factors influencing churn with retention strategies  

## 📂 Dataset
**Telco Customer Churn Dataset** (from Kaggle) containing:
- 21 features including demographics, account details, services, and charges
- Binary target variable (`Churn: Yes/No`)
- No missing values (verified during preprocessing)

## 🔧 Workflow
1. **Data Preprocessing**  
   - Label Encoding for categorical variables
   - Train-test split (80:20)
   - Standardization using `StandardScaler`

2. **EDA Highlights**  
   - Class imbalance observed (more non-churners)
   - Higher monthly charges correlate with churn
   - Longer tenure reduces churn probability

3. **Model Development**  
   ```python
   from sklearn.ensemble import RandomForestClassifier
   model = RandomForestClassifier(n_estimators=100, random_state=42)
   model.fit(X_train, y_train)

```

   📊 Results
   Metric	Non-Churn (0)	Churn (1)
   Precision	83%	67%
   Recall	91%	49%
   F1-Score	87%	56%
   Top 5 Important Features:
   
   Monthly Charges
   
   Tenure
   
   Total Charges
   
   Contract Type
   
   Tech Support


💻 How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/supriya3129/23113165-Machine-Learning.git
   cd 23113165-Machine-Learning

   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

   ```

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook notebooks/codefile-1.ipynb

   ```

   

