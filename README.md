# 💳 QistRisk AI – Credit Scoring & Risk Assessment System

**QistRisk AI** is a machine learning-powered credit scoring application designed to evaluate customer default risk for installment models and financing plans. Built with **Python**, **Scikit-Learn**, and **Streamlit**, this tool automates credit evaluations by providing real-time risk scores and decision support for financial service providers.

---

## 🌟 Key Features

* **Instant Risk Categorization:** Evaluates applicant financial history and demographic details to classify risk into High, Medium, or Low buckets.
* **Predictive Credit Scoring:** Employs classification modeling (Logistic Regression) to calculate precision default probabilities.
* **Interactive Approval Dashboard:** Clean Streamlit interface allowing credit officers to input applicant profiles and immediately view decision insights.
* **Standardized Pipeline:** Preprocesses financial data, handles scaling, and transforms continuous/categorical variables seamlessly.

---

## 🛠️ Tech Stack & Tools

* **Language:** Python 3.x
* **Machine Learning:** Scikit-Learn (Logistic Regression)
* **Data Manipulation:** Pandas, NumPy
* **Deployment & UI:** Streamlit

---

## ⚙️ How It Works

1. **Applicant Data Input:** Credit parameters (income, existing debt, payment history, loan amount, duration) are collected via the UI.
2. **Preprocessing & Feature Scaling:** Numerical variables are normalized, and categorical fields are encoded.
3. **Model Inference:** The trained Logistic Regression model computes the probability of default and returns a credit score with actionable risk bands.

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone [https://github.com/your-username/qistrisk-credit-ai.git](https://github.com/your-username/qistrisk-credit-ai.git)
cd qistrisk-credit-ai
**2. Set up virtual environment**
               python -m venv venv
               # On Windows:
               venv\Scripts\activate
               # On Mac/Linux:
               source venv/bin/activate
**3. Install dependencies**
               streamlit run app.py
**4. Run the Streamlit App**
               streamlit run app.py
📌 Project Structure
               ├── data/                  # Credit risk datasets / samples
               ├── models/                # Trained Logistic Regression model artifacts (.joblib / .pkl)
               ├── app.py                 # Streamlit web interface
               ├── requirements.txt       # Dependencies list
               └── README.md              # Project documentation
👤 Author
               Portfolio / GitHub: [https://github.com/Sadaf144/qistrisk-credit-ai]
               
               LinkedIn: [https://www.linkedin.com/in/sadaf-m-dafii/?isSelfProfile=true]
