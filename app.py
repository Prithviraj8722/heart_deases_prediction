"""
====================================================
  HEART DISEASE PREDICTION - Streamlit Web App
  Dataset: UCI Cleveland Heart Disease Dataset
  Models: Logistic Regression, Random Forest, SVM
====================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #E24B4A;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
        font-size: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 4px solid #E24B4A;
        margin: 0.5rem 0;
    }
    .risk-high {
        background: #E24B4A22;
        border: 2px solid #E24B4A;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .risk-low {
        background: #63992222;
        border: 2px solid #639922;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .risk-moderate {
        background: #FFA50022;
        border: 2px solid #FFA500;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    div[data-testid="stSidebar"] {
        background: #0f0f1a;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD & TRAIN MODELS (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_and_train():
    df = pd.read_csv("heart_disease_dataset.csv")
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM':                 SVC(kernel='rbf', probability=True, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        y_prob = model.predict_proba(X_test_sc)[:, 1]
        acc   = accuracy_score(y_test, y_pred)
        auc   = roc_auc_score(y_test, y_prob)
        cv    = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='accuracy')
        cm    = confusion_matrix(y_test, y_pred)
        results[name] = {
            'model': model, 'y_pred': y_pred, 'y_prob': y_prob,
            'accuracy': acc, 'auc': auc, 'cv_mean': cv.mean(),
            'cv_std': cv.std(), 'cm': cm
        }

    best_name = max(results, key=lambda k: results[k]['auc'])
    return df, scaler, results, best_name, X, y_test, X_test_sc

df, scaler, results, best_name, X, y_test, X_test_sc = load_and_train()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">❤️ Heart Disease Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">UCI Cleveland Dataset · Logistic Regression · Random Forest · SVM</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR — PATIENT INPUT
# ─────────────────────────────────────────────
st.sidebar.markdown("## 🩺 Patient Data Input")
st.sidebar.markdown("Fill in the patient's clinical features:")

with st.sidebar:
    age      = st.slider("Age", 20, 80, 55)
    sex      = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")
    cp       = st.selectbox("Chest Pain Type (cp)",
                            [0, 1, 2, 3],
                            format_func=lambda x: {0:"Typical Angina",1:"Atypical Angina",
                                                    2:"Non-Anginal",3:"Asymptomatic"}[x])
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 130)
    chol     = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 240)
    fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1],
                            format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
    restecg  = st.selectbox("Resting ECG",
                            [0, 1, 2],
                            format_func=lambda x: {0:"Normal",1:"ST-T Abnormality",2:"LV Hypertrophy"}[x])
    thalach  = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    exang    = st.selectbox("Exercise Induced Angina", [0, 1],
                            format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
    oldpeak  = st.slider("ST Depression (oldpeak)", 0.0, 6.5, 1.0, 0.1)
    slope    = st.selectbox("Slope of Peak ST Segment",
                            [0, 1, 2],
                            format_func=lambda x: {0:"Upsloping",1:"Flat",2:"Downsloping"}[x])
    ca       = st.selectbox("Major Vessels Colored (ca)", [0, 1, 2, 3])
    thal     = st.selectbox("Thalassemia (thal)",
                            [0, 1, 2, 3],
                            format_func=lambda x: {0:"Normal",1:"Fixed Defect",
                                                    2:"Reversible Defect",3:"Unknown"}[x])

    selected_model = st.selectbox("🤖 Model", list(results.keys()),
                                  index=list(results.keys()).index(best_name))
    predict_btn = st.button("🔍 Predict", use_container_width=True, type="primary")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📊 Model Analysis", "📋 Dataset Info"])

# ══════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════
with tab1:
    if predict_btn or True:  # Always show prediction area
        new_patient = pd.DataFrame([{
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
            'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
            'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
        }])

        new_scaled  = scaler.transform(new_patient)
        model_used  = results[selected_model]['model']
        pred        = model_used.predict(new_scaled)[0]
        prob        = model_used.predict_proba(new_scaled)[0][1]
        risk        = 'HIGH' if prob > 0.7 else 'MODERATE' if prob > 0.4 else 'LOW'
        risk_colors = {'HIGH': '#E24B4A', 'MODERATE': '#FFA500', 'LOW': '#639922'}
        risk_emoji  = {'HIGH': '🔴', 'MODERATE': '🟡', 'LOW': '🟢'}

        col1, col2, col3 = st.columns([1, 1.5, 1])

        with col2:
            st.markdown(f"""
            <div class="risk-{'high' if risk=='HIGH' else 'moderate' if risk=='MODERATE' else 'low'}">
                <div style="font-size:3rem">{risk_emoji[risk]}</div>
                <div style="font-size:1.8rem; font-weight:800; color:{risk_colors[risk]}">
                    {'❗ Disease DETECTED' if pred == 1 else '✅ No Disease Detected'}
                </div>
                <div style="font-size:1.1rem; margin-top:0.5rem; color:#ccc">
                    Risk Level: <b style="color:{risk_colors[risk]}">{risk}</b>
                    &nbsp;|&nbsp; Probability: <b>{prob*100:.1f}%</b>
                </div>
                <div style="color:#888; font-size:0.85rem; margin-top:0.4rem">
                    Model: {selected_model}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Probability gauge
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 📈 Disease Probability")
            fig, ax = plt.subplots(figsize=(6, 1.2))
            fig.patch.set_facecolor('#0f0f1a')
            ax.set_facecolor('#0f0f1a')
            bar_color = risk_colors[risk]
            ax.barh([0], [prob], color=bar_color, height=0.5, alpha=0.85)
            ax.barh([0], [1], color='#333', height=0.5, alpha=0.3)
            ax.axvline(0.5, color='orange', linestyle='--', alpha=0.5, lw=1.5)
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], color='white')
            ax.text(prob + 0.02, 0, f'{prob*100:.1f}%', va='center', color=bar_color,
                    fontweight='bold', fontsize=12)
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_visible(False)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_b:
            st.markdown("#### 🩺 Patient Summary")
            summary_data = {
                'Feature': ['Age', 'Sex', 'Chest Pain', 'Blood Pressure', 'Cholesterol',
                            'Max Heart Rate', 'ST Depression'],
                'Value':   [age,
                            'Male' if sex == 1 else 'Female',
                            {0:"Typical Angina",1:"Atypical Angina",2:"Non-Anginal",3:"Asymptomatic"}[cp],
                            f'{trestbps} mm Hg', f'{chol} mg/dl',
                            f'{thalach} bpm', f'{oldpeak}']
            }
            st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

        # All models comparison
        st.markdown("#### 🤖 All Models Comparison")
        cols = st.columns(3)
        for i, (name, res) in enumerate(results.items()):
            m    = res['model']
            p    = m.predict_proba(new_scaled)[0][1]
            pr   = m.predict(new_scaled)[0]
            r    = 'HIGH' if p > 0.7 else 'MODERATE' if p > 0.4 else 'LOW'
            with cols[i]:
                st.metric(
                    label=f"{'⭐ ' if name == best_name else ''}{name}",
                    value='Disease' if pr == 1 else 'No Disease',
                    delta=f"{p*100:.1f}% probability",
                    delta_color="inverse"
                )

# ══════════════════════════════════════════════
# TAB 2 — MODEL ANALYSIS (original plots)
# ══════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Model Performance Analysis")
    st.markdown("_Same analysis output as the original script — preserved exactly._")

    colors_model = ['#378ADD', '#639922', '#E24B4A']
    best         = results[best_name]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor('white')
    fig.suptitle('Heart Disease Prediction - ML Analysis', fontsize=16, fontweight='bold', y=1.01)

    # Plot 1: Accuracy
    ax = axes[0, 0]
    names = list(results.keys())
    accs  = [results[n]['accuracy']*100 for n in names]
    bars  = ax.bar(names, accs, color=colors_model, edgecolor='white', linewidth=0.5)
    ax.set_title('Model Accuracy Comparison', fontweight='bold')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 110)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(['LR', 'RF', 'SVM'])
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Plot 2: ROC
    ax = axes[0, 1]
    for (name, res), color in zip(results.items(), colors_model):
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name.split()[0]} (AUC={res['auc']:.2f})")
    ax.plot([0,1],[0,1],'--', color='gray', alpha=0.5)
    ax.set_title('ROC Curves', fontweight='bold')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Plot 3: Confusion Matrix
    ax = axes[0, 2]
    sns.heatmap(best['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'],
                linewidths=0.5, cbar=False)
    ax.set_title(f'Confusion Matrix\n({best_name})', fontweight='bold')
    ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')

    # Plot 4: Feature Importance
    ax = axes[1, 0]
    rf_model    = results['Random Forest']['model']
    importances = rf_model.feature_importances_
    feat_df     = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feat_df     = feat_df.sort_values('Importance', ascending=False)
    colors_feat = ['#378ADD' if i < 3 else '#B5D4F4' for i in range(len(feat_df))]
    ax.barh(feat_df['Feature'], feat_df['Importance'], color=colors_feat, edgecolor='white')
    ax.set_title('Feature Importance\n(Random Forest)', fontweight='bold')
    ax.set_xlabel('Importance Score')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Plot 5: Target Distribution
    ax = axes[1, 1]
    counts = df['target'].value_counts()
    wedge_props = dict(width=0.5, edgecolor='white', linewidth=2)
    ax.pie(counts, labels=['No Disease', 'Disease'], colors=['#639922', '#E24B4A'],
           autopct='%1.1f%%', startangle=90, wedgeprops=wedge_props,
           textprops={'fontsize': 11})
    ax.set_title('Target Distribution', fontweight='bold')

    # Plot 6: Age vs Max HR
    ax = axes[1, 2]
    for label, color, marker in [(0,'#639922','o'), (1,'#E24B4A','^')]:
        subset = df[df['target'] == label]
        ax.scatter(subset['age'], subset['thalach'], c=color, alpha=0.6,
                   marker=marker, s=50, label='No Disease' if label==0 else 'Disease')
    ax.set_title('Age vs Max Heart Rate', fontweight='bold')
    ax.set_xlabel('Age'); ax.set_ylabel('Max Heart Rate (thalach)')
    ax.legend(); ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Summary Table
    st.markdown("### 📋 Model Comparison Summary")
    summary = pd.DataFrame({
        'Model':    list(results.keys()),
        'Accuracy': [f"{results[n]['accuracy']*100:.2f}%" for n in results],
        'AUC':      [f"{results[n]['auc']:.4f}" for n in results],
        'CV Mean':  [f"{results[n]['cv_mean']*100:.2f}%" for n in results],
        'CV Std':   [f"±{results[n]['cv_std']*100:.2f}%" for n in results],
        'Best':     ['⭐' if n == best_name else '' for n in results],
    })
    st.dataframe(summary, hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — DATASET
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### 📋 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Samples", df.shape[0])
    col2.metric("Features", df.shape[1] - 1)
    col3.metric("Disease Cases", int(df['target'].sum()))
    col4.metric("Missing Values", int(df.isnull().sum().sum()))

    st.markdown("#### 📄 Raw Data")
    st.dataframe(df, use_container_width=True)

    st.markdown("#### 📈 Statistical Summary")
    st.dataframe(df.describe().round(2), use_container_width=True)

    st.markdown("#### 🔑 Feature Descriptions")
    feature_info = pd.DataFrame({
        'Feature':     ['age','sex','cp','trestbps','chol','fbs','restecg',
                        'thalach','exang','oldpeak','slope','ca','thal'],
        'Description': [
            'Age in years',
            'Sex (1=male, 0=female)',
            'Chest pain type (0-3)',
            'Resting blood pressure (mm Hg)',
            'Serum cholesterol (mg/dl)',
            'Fasting blood sugar > 120 mg/dl (1=true)',
            'Resting ECG results (0-2)',
            'Maximum heart rate achieved',
            'Exercise induced angina (1=yes)',
            'ST depression induced by exercise',
            'Slope of peak exercise ST segment',
            'Number of major vessels colored by flouroscopy',
            'Thalassemia type'
        ],
        'Type': ['Continuous','Categorical','Categorical','Continuous','Continuous',
                 'Binary','Categorical','Continuous','Binary','Continuous',
                 'Categorical','Categorical','Categorical']
    })
    st.dataframe(feature_info, hide_index=True, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#666; font-size:0.8rem'>"
    "Heart Disease Prediction · UCI Cleveland Dataset · "
    "For educational purposes only — not a medical diagnostic tool"
    "</div>",
    unsafe_allow_html=True
)
