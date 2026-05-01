import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

def load_data(dataset_source, uploaded_file=None):
    if dataset_source == "Upload CSV" and uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            return None
    elif dataset_source == "Iris":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        # map target to names for clarity
        df['target'] = df['target'].map({i: name for i, name in enumerate(data.target_names)})
        return df
    elif dataset_source == "Breast Cancer":
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df['target'] = df['target'].map({0: 'malignant', 1: 'benign'})
        return df
    return None

def preprocess_data(df, target_col, test_size, imputation, scaling, apply_smote):
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encoding Target if it's categorical
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)
    target_classes = le_target.classes_

    # Handling missing values
    num_cols = X.select_dtypes(include=np.number).columns
    cat_cols = X.select_dtypes(exclude=np.number).columns

    if imputation != "None":
        strategy = imputation.lower()
        if len(num_cols) > 0:
            num_imputer = SimpleImputer(strategy=strategy if strategy in ['mean', 'median', 'most_frequent'] else 'mean')
            X[num_cols] = pd.DataFrame(num_imputer.fit_transform(X[num_cols]), columns=num_cols, index=X.index)
        if len(cat_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X[cat_cols] = pd.DataFrame(cat_imputer.fit_transform(X[cat_cols]), columns=cat_cols, index=X.index)

    # Categorical Encoding for features (One-Hot Encoding)
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        # Ensure all columns are boolean/numeric after get_dummies, not object
        X = X.astype(float)

    # Feature Scaling
    scaler = None
    if scaling == "StandardScaler":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    elif scaling == "MinMaxScaler":
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    # SMOTE
    if apply_smote:
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)

    # Train Test Split
    stratify_col = y if len(np.unique(y)) > 1 else None
    
    # Check if stratify is possible (classes with >= 2 members)
    if stratify_col is not None:
        class_counts = pd.Series(y).value_counts()
        if any(class_counts < 2):
            stratify_col = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify_col
    )

    return X_train, X_test, y_train, y_test, X.columns, target_classes, le_target, scaler
