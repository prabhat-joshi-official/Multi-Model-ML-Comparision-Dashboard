import time
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

def get_models(selected_models):
    models = {}
    if "Logistic Regression" in selected_models:
        models["Logistic Regression"] = LogisticRegression(max_iter=1000, random_state=42)
    if "SVM" in selected_models:
        models["SVM"] = SVC(probability=True, random_state=42)
    if "Decision Tree" in selected_models:
        models["Decision Tree"] = DecisionTreeClassifier(random_state=42)
    if "Random Forest" in selected_models:
        models["Random Forest"] = RandomForestClassifier(random_state=42)
    if "KNN" in selected_models:
        models["KNN"] = KNeighborsClassifier()
    if "Neural Network (MLP)" in selected_models:
        models["Neural Network (MLP)"] = MLPClassifier(max_iter=1000, random_state=42)
    return models

def train_evaluate_models(models, X_train, X_test, y_train, y_test, is_multiclass):
    results = {}
    
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        y_pred = model.predict(X_test)
        
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
        else:
            y_proba = None
            
        acc = accuracy_score(y_test, y_pred)
        
        avg_method = 'macro' if is_multiclass else 'binary'
        
        prec = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
        rec = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        auc = None
        fpr = None
        tpr = None
        
        if y_proba is not None:
            try:
                if is_multiclass:
                    auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
                else:
                    auc = roc_auc_score(y_test, y_proba[:, 1])
                    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            except Exception as e:
                pass
                
        results[name] = {
            "Model": model,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "Training Time (s)": training_time,
            "Confusion Matrix": cm,
            "AUC": auc,
            "FPR": fpr,
            "TPR": tpr,
            "Feature Importances": getattr(model, 'feature_importances_', None)
        }
        
        # Save model
        model_filename = f"{name.replace(' ', '_').lower()}_model.pkl"
        joblib.dump(model, model_filename)
        
    return results
