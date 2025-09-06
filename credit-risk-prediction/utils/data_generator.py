"""
Data Generator for Credit Risk Prediction System
This script generates realistic sample data for training the credit risk model.
Copy this code exactly - it creates a complete dataset with all necessary features.
"""

import pandas as pd
import numpy as np
import os

def generate_credit_data(n_samples=10000):
    """
    Generate realistic credit data for training
    Creates a dataset that mimics real bank loan data
    """
    print("🔄 Generating sample credit data...")
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Generate basic applicant information
    age = np.random.normal(35, 10, n_samples).astype(int)
    age = np.clip(age, 18, 80)  # Keep age realistic
    
    # Income - higher for older people generally
    base_income = 30000 + (age - 18) * 800 + np.random.normal(0, 15000, n_samples)
    annual_income = np.clip(base_income, 15000, 200000)
    
    # Employment length correlates with age
    employment_length = np.array([max(0, (a - 22) / 3) for a in age])
    employment_length = np.random.poisson(employment_length)
    employment_length = np.clip(employment_length, 0, 40)
    
    # Loan amount - people request loans based on their income
    loan_amount = annual_income * np.random.uniform(0.1, 3.0, n_samples)
    loan_amount = np.clip(loan_amount, 1000, 500000)
    
    # Credit score - influenced by age and income
    credit_score_base = 600 + (age - 18) * 2 + (annual_income / 1000)
    credit_score = credit_score_base + np.random.normal(0, 50, n_samples)
    credit_score = np.clip(credit_score, 300, 850).astype(int)
    
    # Debt to income ratio
    debt_to_income = np.random.uniform(0, 0.8, n_samples)
    
    # Number of credit lines
    num_credit_lines = np.random.poisson(3, n_samples)
    
    # Loan purpose (categorical)
    purposes = ['debt_consolidation', 'home_improvement', 'major_purchase', 
               'medical', 'vacation', 'car', 'business', 'other']
    loan_purpose = np.random.choice(purposes, n_samples)
    
    # Home ownership
    home_ownership = np.random.choice(['RENT', 'OWN', 'MORTGAGE'], n_samples, 
                                    p=[0.4, 0.3, 0.3])
    
    # Create target variable (default) based on realistic factors
    # Higher risk factors: low credit score, high debt-to-income, low income
    risk_score = (
        -0.01 * credit_score +  # Lower credit score = higher risk
        10 * debt_to_income +   # Higher debt ratio = higher risk
        -0.00005 * annual_income +  # Lower income = slight risk
        0.1 * (loan_amount / annual_income) +  # Higher loan vs income = risk
        np.random.normal(0, 2, n_samples)  # Random noise
    )
    
    # Convert risk score to probability and then to binary outcome
    default_probability = 1 / (1 + np.exp(-risk_score))  # Sigmoid function
    default = (np.random.random(n_samples) < default_probability).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'age': age,
        'annual_income': annual_income.round(2),
        'employment_length': employment_length,
        'loan_amount': loan_amount.round(2),
        'credit_score': credit_score,
        'debt_to_income_ratio': debt_to_income.round(3),
        'num_credit_lines': num_credit_lines,
        'loan_purpose': loan_purpose,
        'home_ownership': home_ownership,
        'default': default
    })
    
    print(f"✅ Generated {n_samples} samples with {default.sum()} defaults ({default.mean()*100:.1f}% default rate)")
    return data

def save_data(data, filename='data/sample_data.csv'):
    """Save the generated data to CSV file"""
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    data.to_csv(filename, index=False)
    print(f"💾 Data saved to {filename}")
    
    # Print data summary
    print("\n📊 Dataset Summary:")
    print(f"Total samples: {len(data)}")
    print(f"Features: {len(data.columns) - 1}")
    print(f"Default rate: {data['default'].mean()*100:.1f}%")
    print("\nFirst 5 rows:")
    print(data.head())

if __name__ == "__main__":
    print("🏦 Credit Risk Data Generator")
    print("=" * 50)
    
    # Generate data
    credit_data = generate_credit_data(10000)
    
    # Save data
    save_data(credit_data)
    
    print("\n🎉 Data generation complete!")
    print("Next step: Run 'python train_model.py' to train your model")