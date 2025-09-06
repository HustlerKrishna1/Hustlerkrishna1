"""
Credit Risk Model Training Script
This script trains a complete machine learning model for credit risk prediction.
Copy this code exactly - it handles everything from data loading to model saving.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os

def load_and_preprocess_data(filename='data/sample_data.csv'):
    """
    Load and preprocess the credit data
    This function handles all data cleaning and feature engineering
    """
    print("📂 Loading data...")
    
    # Check if data file exists
    if not os.path.exists(filename):
        print(f"❌ Data file not found: {filename}")
        print("Please run 'python utils/data_generator.py' first to create the data")
        return None
    
    # Load the data
    data = pd.read_csv(filename)
    print(f"✅ Loaded {len(data)} samples")
    
    # Handle categorical variables
    print("🔄 Preprocessing categorical variables...")
    
    # Create label encoders for categorical columns
    label_encoders = {}
    categorical_columns = ['loan_purpose', 'home_ownership']
    
    for col in categorical_columns:
        le = LabelEncoder()
        data[f'{col}_encoded'] = le.fit_transform(data[col])
        label_encoders[col] = le
    
    # Create feature engineering
    print("⚙️ Creating engineered features...")
    
    # Loan to income ratio
    data['loan_to_income_ratio'] = data['loan_amount'] / data['annual_income']
    
    # Income per credit line
    data['income_per_credit_line'] = data['annual_income'] / (data['num_credit_lines'] + 1)
    
    # Age groups
    data['age_group'] = pd.cut(data['age'], bins=[0, 25, 35, 50, 100], 
                              labels=[0, 1, 2, 3])
    
    # Credit score categories
    data['credit_score_category'] = pd.cut(data['credit_score'], 
                                         bins=[0, 580, 670, 740, 850],
                                         labels=[0, 1, 2, 3])
    
    # Select features for model training
    feature_columns = [
        'age', 'annual_income', 'employment_length', 'loan_amount',
        'credit_score', 'debt_to_income_ratio', 'num_credit_lines',
        'loan_purpose_encoded', 'home_ownership_encoded',
        'loan_to_income_ratio', 'income_per_credit_line',
        'age_group', 'credit_score_category'
    ]
    
    X = data[feature_columns]
    y = data['default']
    
    print(f"📊 Features selected: {len(feature_columns)}")
    print(f"🎯 Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, label_encoders, feature_columns

def train_model(X, y):
    """
    Train the Random Forest model
    Uses the best parameters for credit risk prediction
    """
    print("🤖 Training Random Forest model...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📈 Training set: {len(X_train)} samples")
    print(f"📊 Test set: {len(X_test)} samples")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model with optimized parameters
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    print("\n📈 Model Performance:")
    print("=" * 40)
    print(classification_report(y_test, y_pred))
    
    # ROC AUC Score
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"🎯 ROC AUC Score: {auc_score:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔍 Top 5 Most Important Features:")
    print(feature_importance.head().to_string(index=False))
    
    return model, scaler, feature_importance

def save_model(model, scaler, label_encoders, feature_columns, feature_importance):
    """
    Save the trained model and all preprocessing components
    """
    print("\n💾 Saving model and components...")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save all components in a single file
    model_components = {
        'model': model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_columns': feature_columns,
        'feature_importance': feature_importance
    }
    
    joblib.dump(model_components, 'data/model.pkl')
    print("✅ Model saved to 'data/model.pkl'")
    
    # Save feature importance as CSV for reference
    feature_importance.to_csv('data/feature_importance.csv', index=False)
    print("✅ Feature importance saved to 'data/feature_importance.csv'")

def main():
    """Main training pipeline"""
    print("🏦 Credit Risk Model Training")
    print("=" * 50)
    
    # Load and preprocess data
    result = load_and_preprocess_data()
    if result is None:
        return
    
    X, y, label_encoders, feature_columns = result
    
    # Train model
    model, scaler, feature_importance = train_model(X, y)
    
    # Save model
    save_model(model, scaler, label_encoders, feature_columns, feature_importance)
    
    print("\n🎉 Training completed successfully!")
    print("Next step: Run 'streamlit run app.py' to launch your application")

if __name__ == "__main__":
    main()