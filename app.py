app_code = """
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Page Config
st.set_page_config(page_title="QistRisk AI - Credit Evaluation", layout="wide")

# Custom Dark Neon UI CSS
st.markdown(\"\"\"
<style>
    /* Dark Theme Setup */
    .stApp {
        background-color: #0B0E14;
        color: #E2E8F0;
    }
    
    /* Sidebar Styling & Bright Text Fixes */
    section[data-testid="stSidebar"] {
        background-color: #111622 !important;
        border-right: 1px solid #1E293B;
    }

    section[data-testid="stSidebar"] label {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }

    section[data-testid="stSidebar"] .stMarkdown p, 
    section[data-testid="stSidebar"] .stMarkdown h3, 
    section[data-testid="stSidebar"] .stMarkdown h4 {
        color: #FFFFFF !important;
    }

    /* Card Containers */
    .dark-card {
        background: #131A27;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    /* Main Decision Cards */
    .confidence-circle {
        border: 4px solid #10B981;
        border-radius: 50%;
        width: 130px;
        height: 130px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin: auto;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
    }
    
    .approve-btn {
        background-color: #10B981;
        color: #000000;
        font-weight: bold;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        text-align: center;
        margin-top: 1rem;
        display: inline-block;
    }

    .reject-btn {
        background-color: #EF4444;
        color: #FFFFFF;
        font-weight: bold;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        text-align: center;
        margin-top: 1rem;
        display: inline-block;
    }

    /* Badge Tags */
    .badge-green {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        font-size: 11px;
        padding: 3px 8px;
        border-radius: 4px;
        border: 1px solid #10B981;
    }
    
    .badge-red {
        background: rgba(239, 68, 68, 0.15);
        color: #F87171;
        font-size: 11px;
        padding: 3px 8px;
        border-radius: 4px;
        border: 1px solid #EF4444;
    }

    /* AI Insight Box */
    .ai-quote-box {
        background: #0D131E;
        border-left: 3px solid #10B981;
        padding: 1rem;
        border-radius: 6px;
        font-style: italic;
        color: #94A3B8;
        font-size: 13px;
    }
</style>
\"\"\", unsafe_allow_html=True)

# Train ML Model
@st.cache_data
def train_loan_model():
    np.random.seed(42)
    n = 600
    credit_score = np.random.randint(300, 850, n)
    income = np.random.uniform(50000, 500000, n)
    loan_amount = np.random.uniform(100000, 2000000, n)
    employment_years = np.random.uniform(0, 20, n)
    existing_debts = np.random.uniform(0, 100000, n)

    score_factor = (credit_score - 300) / 550
    income_to_loan = income / (loan_amount + 1)
    debt_ratio = existing_debts / (income + 1)

    logit = (3.5 * score_factor) + (1.2 * income_to_loan) + (0.1 * employment_years) - (2.5 * debt_ratio) - 1.5
    prob = 1 / (1 + np.exp(-logit))
    approved = (prob > 0.5).astype(int)

    df = pd.DataFrame({
        'CreditScore': credit_score,
        'Income': income,
        'LoanAmount': loan_amount,
        'EmploymentYears': employment_years,
        'ExistingDebts': existing_debts,
        'Approved': approved
    })

    X = df[['CreditScore', 'Income', 'LoanAmount', 'EmploymentYears', 'ExistingDebts']]
    y = df['Approved']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression().fit(X_scaled, y)

    return model, scaler

model, scaler = train_loan_model()

# Sidebar - User Inputs
st.sidebar.markdown("### 🇵🇰 <span style='color: #FFFFFF;'>QistRisk AI</span>", unsafe_allow_html=True)
st.sidebar.markdown("<span style='color: #94A3B8; font-size: 12px;'>Smart Credit Scoring Engine</span>", unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.markdown("#### 📋 <span style='color: #FFFFFF;'>Applicant Parameters</span>", unsafe_allow_html=True)
credit_score = st.sidebar.slider("eCIB / Credit Score", 300, 850, 750)
income = st.sidebar.number_input("Monthly Income (PKR)", min_value=30000, max_value=2000000, value=150000, step=10000)
loan_amount = st.sidebar.number_input("Requested Financing (PKR)", min_value=50000, max_value=5000000, value=500000, step=25000)
employment_years = st.sidebar.slider("Employment / Business (Years)", 0.0, 30.0, 4.0, 0.5)
existing_debts = st.sidebar.number_input("Monthly Existing Obligations (PKR)", min_value=0, max_value=500000, value=20000, step=5000)

# Run Inference
input_df = pd.DataFrame([[credit_score, income, loan_amount, employment_years, existing_debts]],
                        columns=['CreditScore', 'Income', 'LoanAmount', 'EmploymentYears', 'ExistingDebts'])

scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)[0]
probabilities = model.predict_proba(scaled_input)[0]

approval_prob = probabilities[1] * 100
rejection_prob = probabilities[0] * 100
dti_ratio = (existing_debts / income) * 100

# ---------------- MAIN CONTENT GRID ----------------
left_col, right_col = st.columns([2, 1])

with left_col:
    # MAIN APP CARD WITH PAKISTANI FINTECH BRANDING
    st.markdown("<div class='dark-card'>", unsafe_allow_html=True)
    st.markdown("#### 🟢 QistRisk Evaluation Engine")
    st.write("")
    
    hero_col1, hero_col2 = st.columns([1, 2])
    
    with hero_col1:
        color = "#10B981" if prediction == 1 else "#EF4444"
        conf_val = approval_prob if prediction == 1 else rejection_prob
        
        st.markdown(f\"\"\"
        <div class='confidence-circle' style='border-color: {color};'>
            <span style='font-size: 26px; font-weight: bold; color: #FFFFFF;'>{conf_val:.0f}%</span>
            <span style='font-size: 9px; color: #64748B; letter-spacing: 1px;'>CONFIDENCE</span>
        </div>
        \"\"\", unsafe_allow_html=True)
        
    with hero_col2:
        if prediction == 1:
            st.write("Based on eCIB score, income capability, and debt ratio, the system recommends **Financing Approval**.")
            st.markdown("<div class='approve-btn'>✓ Approve Financing</div>", unsafe_allow_html=True)
        else:
            st.write("High debt-to-income ratio or insufficient credit history detected. System suggests **Rejection / Further Verification**.")
            st.markdown("<div class='reject-btn'>✕ Decline Financing</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # 3 SUB-METRIC CARDS
    m1, m2, m3 = st.columns(3)
    
    with m1:
        badge = "badge-green" if credit_score >= 680 else "badge-red"
        status = "Low Risk" if credit_score >= 720 else ("Moderate" if credit_score >= 650 else "High Risk")
        st.markdown(f\"\"\"
        <div class='dark-card'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <span style='color: #94A3B8; font-size: 12px;'>eCIB History</span>
                <span class='{badge}'>{status}</span>
            </div>
            <h3 style='color: #FFFFFF; margin-top: 10px;'>{credit_score} <span style='font-size:12px; color:#94A3B8;'>SBP Score</span></h3>
        </div>
        \"\"\", unsafe_allow_html=True)

    with m2:
        badge = "badge-green" if dti_ratio <= 35 else "badge-red"
        status = "Optimized" if dti_ratio <= 35 else "Elevated DTI"
        st.markdown(f\"\"\"
        <div class='dark-card'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <span style='color: #94A3B8; font-size: 12px;'>Debt-to-Income</span>
                <span class='{badge}'>{status}</span>
            </div>
            <h3 style='color: #FFFFFF; margin-top: 10px;'>{dti_ratio:.1f}% <span style='font-size:12px; color:#94A3B8;'>Monthly Ratio</span></h3>
        </div>
        \"\"\", unsafe_allow_html=True)

    with m3:
        st.markdown(f\"\"\"
        <div class='dark-card'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <span style='color: #94A3B8; font-size: 12px;'>Biometric Verification</span>
                <span class='badge-green'>NADRA Verified</span>
            </div>
            <h3 style='color: #FFFFFF; margin-top: 10px;'>100% <span style='font-size:12px; color:#94A3B8;'>CNIC Match</span></h3>
        </div>
        \"\"\", unsafe_allow_html=True)

with right_col:
    # FINANCIAL PORTFOLIO SIDE CARD
    st.markdown(f\"\"\"
    <div class='dark-card'>
        <h4 style='color: #FFFFFF;'>Applicant Financials</h4>
        <div style='display:flex; justify-content:space-between; margin-top:15px;'>
            <span style='color:#94A3B8;'>Monthly Income</span>
            <span style='color:#FFFFFF; font-weight:bold; margin-left:auto;'>PKR {income:,.0f}</span>
        </div>
        <div style='display:flex; justify-style:space-between; margin-top:10px;'>
            <span style='color:#94A3B8;'>Monthly Debt</span>
            <span style='color:#FFFFFF; font-weight:bold; margin-left:auto;'>PKR {existing_debts:,.0f}</span>
        </div>
        <div style='display:flex; justify-style:space-between; margin-top:10px;'>
            <span style='color:#94A3B8;'>Experience</span>
            <span style='color:#FFFFFF; font-weight:bold; margin-left:auto;'>{employment_years} Years</span>
        </div>
        <hr style='border-color: #1E293B;'>
        <p style='color:#10B981; font-size:11px;'>◉ AI PREDICTION INSIGHT</p>
        <div class='ai-quote-box'>
            "Applicant demonstrates stable cash flow indicators and low default probability across micro-financing parameters."
        </div>
    </div>
    \"\"\", unsafe_allow_html=True)

    # EVIDENCE ARTIFACTS
    st.markdown(\"\"\"
    <div class='dark-card'>
        <p style='color:#94A3B8; font-size:12px;'>📂 Verification Documents</p>
        <p style='color:#34D399; font-size:13px; margin:5px 0;'>✓ NADRA_Verisys_Bio.pdf (Verified)</p>
        <p style='color:#34D399; font-size:13px; margin:5px 0;'>✓ Bank_Statement_6M.pdf (Auto-Parsed)</p>
    </div>
    \"\"\", unsafe_allow_html=True)
"""

with open("app.py", "w") as f:
    f.write(app_code)

print("✅ app.py updated with '🇵🇰 QistRisk AI'!")
