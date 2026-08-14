# AI-Powered-Financial-Fraud-Detection
Machine learning project for detecting fraudulent financial transactions using Logistic Regression and Random Forest.
# AI-Powered Financial Fraud Detection System

## 📌 Project Overview

The *AI-Powered Financial Fraud Detection System* is a machine learning project designed to identify potentially fraudulent financial transactions.

The project uses two classification algorithms:

- Logistic Regression
- Random Forest

Because fraudulent transactions are highly imbalanced compared with normal transactions, *SMOTE (Synthetic Minority Oversampling Technique)* is used to balance the training data.

The models are evaluated using Precision, Recall, F1-Score, Accuracy, and Confusion Matrix.

---

## 🎯 Objectives

- Detect fraudulent financial transactions using machine learning.
- Handle highly imbalanced transaction data.
- Compare Logistic Regression and Random Forest models.
- Identify the most important features associated with fraud.
- Evaluate model performance using appropriate classification metrics.

---

## 📊 Dataset

The project uses a credit card transaction dataset containing:

- *284,807 transactions*
- *30 input features*
- *1 target variable (Class)*
- Class = 0 → Normal transaction
- Class = 1 → Fraudulent transaction

The dataset contains a very small percentage of fraudulent transactions, making class imbalance an important challenge.

---

## 🔧 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Imbalanced-learn
- SMOTE
- Logistic Regression
- Random Forest

---

## ⚙️ Project Workflow

The project follows these steps:

1. Load the financial transaction dataset.
2. Explore and analyze the data.
3. Visualize normal and fraudulent transactions.
4. Separate features and target variable.
5. Split the dataset into training and testing sets.
6. Apply SMOTE to balance the training data.
7. Standardize the features using StandardScaler.
8. Train a Logistic Regression model.
9. Train a Random Forest model.
10. Generate predictions on unseen test data.
11. Evaluate both models.
12. Compare model performance.
13. Analyze feature importance.

---

## 🤖 Models

### Logistic Regression

The Logistic Regression model was trained after SMOTE balancing and feature scaling.

Performance on the test dataset:

| Metric | Score |
|---|---:|
| Precision | 0.13 |
| Recall | 0.90 |
| F1-Score | 0.23 |
| Accuracy | 0.99 |

### Random Forest

The Random Forest model was trained using 100 decision trees with a maximum depth of 12.

Performance on the test dataset:

| Metric | Score |
|---|---:|
| Precision | 0.60 |
| Recall | 0.88 |
| F1-Score | 0.71 |
| Accuracy | 1.00 |

---

## 🏆 Model Comparison

| Model | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Logistic Regression | 0.13 | 0.90 | 0.23 |
| Random Forest | *0.60* | 0.88 | *0.71* |

Based on the results, *Random Forest performs better overall* because it provides a much higher precision and F1-score while maintaining a high recall.

This makes Random Forest the selected model for the final fraud detection system.

---

## 🔍 Important Features

The Random Forest model identified the following as the top features for fraud detection:

| Feature | Importance |
|---|---:|
| V14 | 0.228412 |
| V10 | 0.126571 |
| V4 | 0.116357 |
| V12 | 0.104634 |
| V17 | 0.081377 |
| V3 | 0.072368 |
| V11 | 0.048714 |
| V16 | 0.043230 |
| V2 | 0.039452 |
| V9 | 0.027727 |

*V14* was the most important feature according to the Random Forest model.

---

## 📈 Results

The Random Forest model achieved:

- *Precision:* 0.60
- *Recall:* 0.88
- *F1-Score:* 0.71
- *Accuracy:* approximately 1.00

The confusion matrix for Random Forest was:

```text
[[56806    58]
 [   12    86]]
The model correctly identified most normal and fraudulent transactions while keeping false positives relatively low.
🚀 Future Improvements
Future improvements could include:
Hyperparameter tuning.
Testing additional machine learning algorithms.
Exploring ensemble learning techniques.
Using deep learning methods.
Developing real-time fraud detection.
Deploying the model as a web application or API.
Adding explainable AI techniques.
📁 Project Structure
AI-Powered-Financial-Fraud-Detection/
├── AI_Powered_Financial_Fraud_Detection_System.ipynb
├── README.md
└── dataset/
The dataset is not included in this repository if it is too large or subject to redistribution restrictions.
▶️ How to Run
1.Open the .ipynb file using Google Colab or Jupyter Notebook.
2.Upload or provide the required dataset.
3.Run the notebook cells sequentially.
4.Review the visualizations and model evaluation results.
👨‍💻 Author
palakraj5
📌 Conclusion
This project demonstrates how machine learning can be used for financial fraud detection.
The results show that Random Forest provides a better overall balance between precision, recall and F1-score compared with Logistic Regression for this dataset.
The system can be further improved through advanced machine learning techniques, hyperparameter optimization and real-time deployment.
*Copy from # AI-Powered Financial Fraud Detection System down to the end.*  
Don't copy the first and last ``` lines.
