import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from utils import load_data, preprocess_data
from model import get_models, train_evaluate_models

st.set_page_config(page_title="Multi-Model ML Dashboard", layout="wide")

st.title("🧠 Multi-Model Machine Learning Comparison Dashboard")
st.markdown("""
Welcome to the Multi-Model ML Comparison Dashboard! 
You can upload your own dataset or use a built-in one, preprocess the data, train multiple models, and compare their performance.
*Tip: You can switch to Dark/Light mode in the Streamlit settings (top right menu -> Settings -> Theme).*
""")

# --- Sidebar Controls ---
st.sidebar.header("1. Dataset Selection")
dataset_source = st.sidebar.selectbox("Choose Dataset Source", ["Built-in: Breast Cancer", "Built-in: Iris", "Upload CSV"])

uploaded_file = None
if dataset_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV dataset", type=["csv"])
    if uploaded_file is not None:
        df = load_data("Upload CSV", uploaded_file)
    else:
        df = None
elif dataset_source == "Built-in: Iris":
    df = load_data("Iris")
elif dataset_source == "Built-in: Breast Cancer":
    df = load_data("Breast Cancer")
else:
    df = None

if df is not None:
    st.header("📊 Dataset Overview")
    st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
    st.dataframe(df.head())
    
    st.subheader("Missing Values")
    st.write(df.isnull().sum()[df.isnull().sum() > 0])

    st.sidebar.header("2. Preprocessing Options")
    
    target_col = st.sidebar.selectbox("Select Target Column", df.columns, index=len(df.columns)-1)
    
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    imputation = st.sidebar.selectbox("Handle Missing Values", ["None", "Mean", "Median", "Most_Frequent"])
    scaling = st.sidebar.selectbox("Feature Scaling", ["None", "StandardScaler", "MinMaxScaler"])
    apply_smote = st.sidebar.checkbox("Apply SMOTE (Handle Imbalance)")

    st.sidebar.header("3. Model Selection")
    available_models = ["Logistic Regression", "SVM", "Decision Tree", "Random Forest", "KNN", "Neural Network (MLP)"]
    selected_models = st.sidebar.multiselect("Select Models to Train", available_models, default=["Logistic Regression", "Random Forest"])

    if st.sidebar.button("🚀 Train Models"):
        if not selected_models:
            st.error("Please select at least one model to train.")
        else:
            with st.spinner("Preprocessing Data..."):
                try:
                    X_train, X_test, y_train, y_test, feature_cols, target_classes, le_target, scaler = preprocess_data(
                        df, target_col, test_size, imputation, scaling, apply_smote
                    )
                    
                    st.header("⚙️ Preprocessing Results")
                    col1, col2 = st.columns(2)
                    col1.metric("Training Set Shape", f"{X_train.shape[0]} samples")
                    col2.metric("Testing Set Shape", f"{X_test.shape[0]} samples")
                    
                    is_multiclass = len(np.unique(y_train)) > 2
                    
                except Exception as e:
                    st.error(f"Error during preprocessing: {e}")
                    st.stop()

            with st.spinner("Training Models..."):
                try:
                    models_to_train = get_models(selected_models)
                    results = train_evaluate_models(models_to_train, X_train, X_test, y_train, y_test, is_multiclass)
                except Exception as e:
                    st.error(f"Error during model training: {e}")
                    st.stop()
                    
            st.header("🏆 Evaluation Results")
            
            # Create Results Table
            metrics_data = []
            for name, res in results.items():
                metrics_data.append({
                    "Model": name,
                    "Accuracy": res["Accuracy"],
                    "Precision": res["Precision"],
                    "Recall": res["Recall"],
                    "F1 Score": res["F1 Score"],
                    "AUC": res["AUC"] if res["AUC"] is not None else np.nan,
                    "Training Time (s)": res["Training Time (s)"]
                })
            
            results_df = pd.DataFrame(metrics_data).set_index("Model")
            
            # Highlight best model based on accuracy
            st.dataframe(results_df.style.highlight_max(subset=["Accuracy", "F1 Score"], color="lightgreen", axis=0))
            
            # Downloadable Report
            csv = results_df.to_csv()
            st.download_button(
                label="📥 Download Metrics Report (CSV)",
                data=csv,
                file_name="model_comparison_metrics.csv",
                mime="text/csv",
            )
            
            # Visualizations
            st.subheader("📈 Model Performance Comparisons")
            
            tabs = st.tabs(["Accuracy Chart", "ROC Curves", "Confusion Matrices", "Feature Importance"])
            
            with tabs[0]:
                fig_acc = px.bar(results_df.reset_index(), x='Model', y='Accuracy', color='Model', title='Model Accuracy Comparison')
                fig_acc.update_layout(yaxis=dict(range=[0, 1]))
                st.plotly_chart(fig_acc, use_container_width=True)
                
            with tabs[1]:
                if not is_multiclass:
                    fig_roc = go.Figure()
                    for name, res in results.items():
                        if res["FPR"] is not None and res["TPR"] is not None:
                            fig_roc.add_trace(go.Scatter(x=res["FPR"], y=res["TPR"], mode='lines', name=f'{name} (AUC = {res["AUC"]:.2f})'))
                    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random'))
                    fig_roc.update_layout(title='ROC Curves', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
                    st.plotly_chart(fig_roc, use_container_width=True)
                else:
                    st.info("ROC Curve visualization is currently simplified for binary classification. Multiclass AUC is shown in the table.")
                    
            with tabs[2]:
                selected_model_cm = st.selectbox("Select Model for Confusion Matrix", list(results.keys()))
                cm = results[selected_model_cm]["Confusion Matrix"]
                fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                                   labels=dict(x="Predicted", y="Actual"),
                                   x=target_classes, y=target_classes,
                                   title=f"Confusion Matrix: {selected_model_cm}")
                st.plotly_chart(fig_cm, use_container_width=True)
                
            with tabs[3]:
                tree_models = [name for name, res in results.items() if res["Feature Importances"] is not None]
                if tree_models:
                    selected_tree_model = st.selectbox("Select Model for Feature Importance", tree_models)
                    importances = results[selected_tree_model]["Feature Importances"]
                    feat_imp_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
                    feat_imp_df = feat_imp_df.sort_values(by="Importance", ascending=False).head(10)
                    
                    fig_feat = px.bar(feat_imp_df, x="Importance", y="Feature", orientation='h', title=f"Top 10 Feature Importances: {selected_tree_model}")
                    fig_feat.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_feat, use_container_width=True)
                else:
                    st.info("No Tree-based models selected for feature importance.")
            
            st.divider()
            
            # Real-time prediction section
            st.header("🔮 Real-Time Prediction")
            st.write("Use the best model (by Accuracy) to make a live prediction.")
            
            best_model_name = results_df["Accuracy"].idxmax()
            best_model = results[best_model_name]["Model"]
            st.write(f"**Best Model Selected:** {best_model_name}")
            
            st.write("Enter feature values:")
            
            # Create a form for inputs to avoid rerunning constantly
            with st.form("prediction_form"):
                input_data = {}
                cols = st.columns(3)
                for i, col in enumerate(feature_cols):
                    # We just use numeric inputs for simplicity
                    # In a real scenario, we might want to handle categorical UI differently, 
                    # but since we already applied get_dummies, feature_cols are all numeric now.
                    # Wait, if get_dummies was applied, the user shouldn't enter raw values but the encoded ones?
                    # For simplicity, we just ask for numerical input.
                    with cols[i % 3]:
                        input_data[col] = st.number_input(f"{col}", value=0.0)
                
                submitted = st.form_submit_button("Predict")
                
                if submitted:
                    input_df = pd.DataFrame([input_data])
                    # If scaler was used, scale input
                    if scaler is not None:
                        input_df[input_df.columns] = scaler.transform(input_df)
                        
                    prediction = best_model.predict(input_df)
                    predicted_class = le_target.inverse_transform(prediction)[0]
                    
                    st.success(f"The predicted class is: **{predicted_class}**")

else:
    st.info("Please select a dataset to proceed.")
