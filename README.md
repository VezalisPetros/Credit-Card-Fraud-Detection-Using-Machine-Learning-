
# 🕵️ Credit Card Fraud Detection using Machine Learning

## 📌 Overview

This project addresses the challenge of detecting fraudulent credit card transactions in highly imbalanced datasets. Using real-world anonymized data, I implemented a comprehensive machine learning pipeline — from data exploration and preprocessing to model selection, dimensionality reduction, and advanced hyperparameter optimization techniques.

> 📊 **Dataset**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
> 👨‍💻 **Author**: Petros Vezalis – Exemption Assignment for University of Macedonia  
> 🗓️ **Date**: February 2025

---

## 🔍 Problem Statement

Credit card fraud accounts for a small fraction of all transactions, making the dataset extremely imbalanced (~0.17% fraud). The goal of this project was to build robust machine learning models that:

- Accurately classify fraudulent transactions.
- Handle extreme class imbalance.
- Offer precision and recall suitable for real-world deployment.
- Evaluate trade-offs between performance and interpretability.

---

## 🧠 Techniques & Tools

| Stage | Methods/Tools |
|------|----------------|
| **Data Analysis & Cleaning** | Exploratory Data Analysis (EDA), Feature Standardization |
| **Imbalanced Data Handling** | Class weights, RandomOversampling, Stratified K-Fold CV |
| **Feature Engineering** | PCA (Principal Component Analysis), RFE (Recursive Feature Elimination), Feature Importance via Random Forests |
| **Models Trained** | Logistic Regression, Decision Trees, Random Forest, Support Vector Machines (SVM), Artificial Neural Networks (ANN) |
| **Evaluation Metrics** | Precision, Recall, F1-Score, Accuracy, AUC-ROC |
| **Hyperparameter Optimization** | Grid Search, Randomized Search, Bayesian Optimization (Optuna) |
| **Libraries Used** | Python, scikit-learn, matplotlib, seaborn, NumPy, pandas, Optuna |

---

## 📈 Key Results

| Model | F1-Score | Precision | Recall |
|-------|----------|-----------|--------|
| ✅ **Random Forest** (default) | **0.838** | 0.95 | 0.75 |
| SVM (Bayesian Opt) | 0.815 | 0.838 | **0.80** |
| ANN (Bayesian Opt) | 0.811 | 0.859 | 0.774 |
| Logistic Regression (Bayesian Opt) | 0.776 | 0.864 | 0.713 |
| Decision Tree (default) | 0.78 | 0.81 | 0.73 |

> 🎯 **Conclusion**: The Random Forest classifier performed best overall, especially in balancing high precision (to reduce false positives) with strong recall.

---

## 💡 Highlights

- Built a scalable pipeline that combines model interpretability with real-world performance needs.
- Performed **deep hyperparameter optimization** (Grid Search, Randomized Search, Bayesian Optimization).
- Applied **dimensionality reduction** while preserving model performance (PCA retained 95% variance; RFE improved recall).
- Provided comparative model analysis and practical deployment considerations.

---

## 🚀 Real-World Deployment Considerations

- **Interpretability**: Ensemble models like Random Forest offer feature importance; tools like SHAP can enhance transparency.
- **Efficiency**: RFE and PCA helped reduce computation time for deployment without sacrificing accuracy.
- **Monitoring & Updates**: Designed to allow retraining with new data to adapt to evolving fraud patterns.
- **Regulatory Readiness**: Maintains fairness, explainability, and transparency needed in the financial sector.

---

## 📂 Repository Structure

```bash
📁 Fraud-Detection
├── 📄 README.md
├── 📄 Fraud-Detection.pdf         # Full Report with Analysis
├── 📄 Machine_Learning.ipynb     # Annotated Jupyter Notebook with code
└── 📊 data/                       # Placeholder for dataset (not included)
```

> ⚠️ **Note**: Dataset not included due to size/license. Download it [here](https://www.kaggle.com/mlg-ulb/creditcardfraud).

---

## 🧠 Future Work

- Integrate SHAP/LIME for deeper interpretability.
- Test ensemble blending techniques (e.g., Voting, Stacking).
- Experiment with deep learning architectures (e.g., autoencoders for anomaly detection).
- Build a dashboard for real-time fraud monitoring using Streamlit or Dash.

---

## 🤝 Let's Connect

I'm actively looking for opportunities in AI/Data Science & Machine Learning.  
If you're hiring or collaborating on similar projects — feel free to reach out!

- 📧 **ics22106@uom.edu.gr**
- 💼 LinkedIn: *(add your profile)*
- 💡 Portfolio: Coming soon!
