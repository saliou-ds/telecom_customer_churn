# 🔍 Customer Churn Risk Prediction

## 📌 Project Overview

This project aims to develop a machine learning pipeline to **predict customer churn** in a business context. Churn prediction allows companies to identify clients at risk of leaving and take proactive retention actions, improving long-term revenue and customer satisfaction.

This end-to-end project includes:
- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Handling missing values and outliers
- Feature engineering and selection
- Addressing class imbalance (SMOTE, class weighting)
- Model training and selection (Logistic Regression, Random Forest, XGBoost)
- Performance evaluation and threshold tuning
- Model interpretability using SHAP
- Deployment of a dashboard in **Power BI** for business decision support

---

## 🧠 Business Objective

Customer churn significantly impacts subscription-based businesses. Being able to identify clients likely to leave enables companies to:
- Reduce churn rate
- Design personalized marketing campaigns
- Improve customer retention strategy
- Allocate resources more effectively

---

## 📂 Project Structure

<pre> ```bash telecom_customer_churn/ ├── data/ # Raw and processed data ├── notebooks/ # EDA and experiments │ └── churn_analysis.ipynb ├── src/ # Python scripts for modularity │ ├── data_preprocessing.py │ ├── feature_engineering.py │ ├── model_training.py │ ├── evaluation.py │ └── prediction_service.py ├── models/ # Saved models and metrics ├── dashboard/ # Power BI report (.pbix or images) ├── requirements.txt # Pip dependencies ├── requirements_conda.yml # Conda environment └── README.md # Project description ``` </pre>

---

## 📁 Dataset

The dataset used in this project is the **Telco Customer Churn** dataset, which contains customer-level information for a telecom provider, including contract details, services used, tenure, charges, and churn labels.

📦 **Source**: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
📄 License: Open Dataset for educational and non-commercial use

The dataset includes 7,043 records with 21 features, and a binary target variable indicating churn (`Yes` / `No`).

---

## ⚙️ Techniques and Tools Used

| Category               | Tools / Methods                              |
|------------------------|----------------------------------------------|
| Language               | Python, Power BI                             |
| Libraries              | Pandas, NumPy, Scikit-learn, Maplotlib, SHAP |
| Preprocessing          | Encoding, imputation, outlier treatment      |
| Feature Engineering    | Correlation analysis, tree-based selection   |
| Model Evaluation       | Recall, ROC-AUC, confusion matrix            |
| Class Imbalance        | SMOTE                                        |
| Visualization          | Matplotlib, Seaborn, SHAP, Power BI dashboard|

---

## 📈 Model Results

- Best-performing model: **Logistic Regression with class balancing (SMOTE) and threshold tuning (at 0.4)**
- Achieved good **recall on churn class**
- Interpretability via **SHAP** provided insight into key churn drivers (e.g., contract type, tenure, monthly charges and internet service)

---

## 📊 Power BI Dashboard

The final model results and churn insights are visualized in a **Power BI dashboard**, which enables:
- Filtering by customer characteristics
- Churn probability overview per customer segment
- Highlight of key features influencing churn predictions

*Note: A screenshot is available in the `dashboard/` folder.*

---

## 🚀 How to Run

1. Clone the repository
2. Install dependencies from `requirements.txt` with pip or use `environment.yml` to create an operational virtual environment with conda
3. Run notebook in `notebooks/` or modular scripts in the `src/` folder
4. Open the `.pbix` file in Power BI Desktop for the dashboard

---

## 📌 Status

✅ Project completed  
🛠️ Ready to be extended with time-series features or deployed via API

---

## 📧 Contact

For questions or collaborations, feel free to reach out:
**Saliou** – Data Scientist  
🌐 [[LinkedIn](https://www.linkedin.com/in/saliou-cisse-9b9935141/) / [GitHub](https://github.com/saliou-ds)]