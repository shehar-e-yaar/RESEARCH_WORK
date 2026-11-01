import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image

# Page Configuration
st.set_page_config(page_title="DDoS AI Hunter", page_icon="🛡️", layout="wide")

# Custom CSS for Dark Mode Optimization
st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: #2b2b36;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #4a4a5a;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Zero-Day DDoS Threat Detection")
st.markdown("Advanced AI-powered network traffic analyzer based on CICIDS2017 & CICDDoS2019 architecture.")

# Directories & Paths
MODEL_DIR = "models"
METRICS_DIR = "metrics"
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
scaler_cnn_path = os.path.join(MODEL_DIR, "scaler_cnn.pkl")
xgb_model_path = os.path.join(MODEL_DIR, "xgboost_model.pkl")
rf_model_path = os.path.join(MODEL_DIR, "random_forest_model.pkl")
cnn_model_path = os.path.join(MODEL_DIR, "cnn_model.h5")
columns_path = os.path.join(MODEL_DIR, "train_columns.pkl")

@st.cache_resource
def load_all_models():
    models = {}
    if os.path.exists(xgb_model_path): models['XGBoost'] = joblib.load(xgb_model_path)
    if os.path.exists(rf_model_path): models['Random Forest'] = joblib.load(rf_model_path)
    if os.path.exists(cnn_model_path): models['1D-CNN'] = load_model(cnn_model_path)
    return models

@st.cache_resource
def load_assets():
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    scaler_cnn = joblib.load(scaler_cnn_path) if os.path.exists(scaler_cnn_path) else None
    train_cols = joblib.load(columns_path) if os.path.exists(columns_path) else None
    return scaler, scaler_cnn, train_cols

models = load_all_models()
scaler, scaler_cnn, train_cols = load_assets()

st.sidebar.image("ddos.png", width=120)
st.sidebar.header("Scan Parameters")
selected_model_name = st.sidebar.selectbox("Select AI Engine", list(models.keys()))

st.sidebar.markdown("---")
st.sidebar.info("Upload a network traffic file (CSV) extracted from tools like Wireshark/CICFlowMeter.")

# --- TABS CREATION ---
tab_live, tab_archive = st.tabs(["🔴 Live Traffic Analysis", "📈 Model Performance Archive"])

# ==========================================
# TAB 1: LIVE ANALYSIS
# ==========================================
with tab_live:
    uploaded_file = st.file_uploader("Upload Network Traffic Data (.csv)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        df.replace([np.inf, -np.inf, 'Infinity', 'infinity', 'NaN', 'nan'], np.nan, inplace=True)
        
        st.markdown("### 📊 Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        st.caption(f"Total loaded instances: **{df.shape[0]:,} rows** | Features: **{df.shape[1]} columns**")
        
        st.markdown("---")
        
        if st.button("🚀 Execute Threat Analysis", use_container_width=True):
            with st.spinner(f'Initializing {selected_model_name} Engine and analyzing packets...'):
                
                # 1. Preprocessing & Alignment
                column_mapping = {
                    'Fwd Packets Length Total': 'Total Length of Fwd Packets',
                    'Bwd Packets Length Total': 'Total Length of Bwd Packets',
                    'Packet Length Min': 'Min Packet Length',
                    'Packet Length Max': 'Max Packet Length',
                    'Avg Packet Size': 'Average Packet Size',
                    'Init Fwd Win Bytes': 'Init_Win_bytes_forward',
                    'Init Bwd Win Bytes': 'Init_Win_bytes_backward',
                    'Fwd Act Data Packets': 'act_data_pkt_fwd',
                    'Fwd Seg Size Min': 'min_seg_size_forward'
                }
                
                label_col = 'Label' if 'Label' in df.columns else None
                X = df.drop(columns=[label_col]) if label_col else df.copy()
                X.rename(columns=column_mapping, inplace=True)
                X = X.reindex(columns=train_cols, fill_value=0)
                X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
                
                # 2. Prediction
                model = models[selected_model_name]
                
                if selected_model_name == '1D-CNN':
                    X_scaled = scaler_cnn.transform(X)
                    X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
                    probs = model.predict(X_reshaped).flatten()
                else:
                    X_scaled = scaler.transform(X)
                    probs = model.predict_proba(X_scaled)[:, 1]
                    
                preds = (probs >= 0.50).astype(int)
                
                # Calculations
                total = len(preds)
                attacks = sum(preds)
                normals = total - attacks
                
                # 3. Dynamic Alerting
                if attacks > 0:
                    st.error(f"🚨 ALERT: {attacks:,} malicious packets detected in the network stream!")
                else:
                    st.success("✅ ALL CLEAR: No malicious activity detected in this stream.")
                
                # 4. Metric Cards
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Packets Scanned", f"{total:,}")
                col2.metric("Detected Threats", f"{attacks:,}", delta="Critical" if attacks>0 else "Safe", delta_color="inverse")
                col3.metric("Normal Traffic", f"{normals:,}")
                
                st.markdown("---")
                
                # 5. Advanced Visualizations
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    st.subheader("Traffic Distribution")
                    res_df = pd.DataFrame({'Status': ['Normal', 'Attack'], 'Count': [normals, attacks]})
                    fig_pie = px.pie(res_df, values='Count', names='Status', 
                                     color='Status', 
                                     color_discrete_map={'Normal':'#2ecc71', 'Attack':'#e74c3c'},
                                     hole=0.4)
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                with viz_col2:
                    st.subheader("AI Confidence Distribution")
                    fig_hist = px.histogram(x=probs, nbins=20, 
                                            labels={'x':'Probability of being an Attack', 'y':'Packet Count'},
                                            color_discrete_sequence=['#3498db'])
                    fig_hist.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Threshold (0.50)")
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                # XAI: EXPLAINABLE AI
                if selected_model_name in ['XGBoost', 'Random Forest']:
                    st.markdown("---")
                    st.subheader("🧠 Explainable AI (XAI) - Decision Factors")
                    st.caption(f"Top 10 network features driving the **{selected_model_name}** model's predictions.")
                    
                    importances = model.feature_importances_
                    feat_df = pd.DataFrame({
                        'Feature': train_cols,
                        'Importance': importances
                    }).sort_values(by='Importance', ascending=False).head(10)
                    
                    fig_xai = px.bar(feat_df, x='Importance', y='Feature', orientation='h',
                                     color='Importance', color_continuous_scale='Reds' if attacks > 0 else 'Greens')
                    
                    fig_xai.update_layout(yaxis={'categoryorder':'total ascending'}, 
                                          margin=dict(l=0, r=0, t=30, b=0))
                    
                    st.plotly_chart(fig_xai, use_container_width=True)

                # 6. Ground Truth Comparison
                if label_col:
                    st.markdown("---")
                    st.subheader("🎯 Model Performance Analysis (Ground Truth vs Prediction)")
                    
                    true_labels = df[label_col].apply(lambda x: 0 if 'BENIGN' in str(x).upper() or str(x) == '0' else 1)
                    acc = accuracy_score(true_labels, preds)
                    st.info(f"**Accuracy Score:** {acc*100:.2f}%")
                    
                    report = classification_report(true_labels, preds, target_names=['Normal', 'Attack'], output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.format("{:.3f}").background_gradient(cmap='Blues'), use_container_width=True)

    else:
        st.info("👋 Welcome! Please upload a network traffic CSV file to begin live threat hunting.")


# ==========================================
# TAB 2: MODEL PERFORMANCE ARCHIVE
# ==========================================
with tab_archive:
    st.header(f"Historical Performance: {selected_model_name}")
    st.markdown("This section displays the static evaluation metrics generated during the model's training phase.")
    
    # Map selected model name to the prefix used in saved files
    prefix_map = {'XGBoost': 'xgb', 'Random Forest': 'rf', '1D-CNN': 'cnn'}
    pfx = prefix_map.get(selected_model_name, 'xgb')
    
    # File paths
    val_json_path = os.path.join(METRICS_DIR, f"{pfx}_val_metrics.json")
    val_png_path = os.path.join(METRICS_DIR, f"{pfx}_val_conf.png")
    test_json_path = os.path.join(METRICS_DIR, f"{pfx}_test2019_metrics.json")
    test_png_path = os.path.join(METRICS_DIR, f"{pfx}_test2019_conf.png")
    
    arch_col1, arch_col2 = st.columns(2)
    
    # --- Validation Set Column ---
    with arch_col1:
        st.subheader("Validation Set (CICIDS2017 + CIC-DDoS2019(33%))")
        if os.path.exists(val_json_path):
            with open(val_json_path, 'r') as f:
                val_metrics = json.load(f)
            
            # Display metrics nicely
            st.markdown(f"""
            * **Accuracy:** {val_metrics.get('accuracy', 0)*100:.2f}%
            * **Precision:** {val_metrics.get('precision', 0)*100:.2f}%
            * **Recall:** {val_metrics.get('recall', 0)*100:.2f}%
            * **F1-Score:** {val_metrics.get('f1_score', 0)*100:.2f}%
            * **ROC-AUC:** {val_metrics.get('roc_auc', 0)*100:.2f}%
            """)
        else:
            st.warning("Validation metrics JSON not found.")
            
        if os.path.exists(val_png_path):
            img_val = Image.open(val_png_path)
            st.image(img_val, caption=f"{selected_model_name} - Validation Confusion Matrix", use_container_width=True)
        else:
            st.warning("Validation Confusion Matrix PNG not found.")

    # --- Test Set Column ---
    with arch_col2:
        st.subheader("Zero-Day Test Set (CICDDoS2019(67%))")
        if os.path.exists(test_json_path):
            with open(test_json_path, 'r') as f:
                test_metrics = json.load(f)
            
            st.markdown(f"""
            * **Accuracy:** {test_metrics.get('accuracy', 0)*100:.2f}%
            * **Precision:** {test_metrics.get('precision', 0)*100:.2f}%
            * **Recall:** {test_metrics.get('recall', 0)*100:.2f}%
            * **F1-Score:** {test_metrics.get('f1_score', 0)*100:.2f}%
            * **ROC-AUC:** {test_metrics.get('roc_auc', 0)*100:.2f}%
            """)
        else:
            st.warning("Test metrics JSON not found.")
            
        if os.path.exists(test_png_path):
            img_test = Image.open(test_png_path)
            st.image(img_test, caption=f"{selected_model_name} - Test2019 Confusion Matrix", use_container_width=True)
        else:
            st.warning("Test Confusion Matrix PNG not found.")