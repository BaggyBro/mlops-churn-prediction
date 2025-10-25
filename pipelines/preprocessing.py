import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

class DataPreprocessor:
    """ Handles all data preprocessing for bank churn dataset"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def preprocess(self, df):
        """ Clean and transform data """
        df = df.copy()

        # drop unneeded columns
        drop_columns = ['RowNumber', 'Surname', 'CustomerId']
        df = df.drop(columns=[col for col in drop_columns if col in df.columns])

        # handle missing values
        print(f"Missing values: \n {df.isnull().sum()}")
        df = df.dropna()

        # target variable is binary 0/1
        if 'Exited' in df.columns:
            df['Exited'] = df['Exited'].astype(int)

        # encode categorical variables
        categorical_columns = ['Geography', 'Gender']
        print("Categorical columns to encode: {categorical_cols}")

        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        # separate features and target
        if 'Exited' in df.columns:
            X = df.drop('Exited', axis=1)
            y = df['Exited']
        else:
            raise ValueError("Target column not found")
        
        # scale numerical features
        numerical_columns = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
        print(f"Numerical columns to scale: {numerical_columns}")

        X[numerical_columns] = self.scaler.fit_transform(x[numerical_columns])
        return X, y
    

    def split_data(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        print(f"Data split: \n")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Training churn rate: {y_train.mean():.2%}")
        print(f"Test churn rate: {y_test.mean():.2%}")

        return X_train, X_test, y_train, y_test
    
    def save(self, filepath='models/preprocessor.pkl'):
        joblib.dump({
            'scaler':self.scaler,
            'label_encoders': self.label_encoders
        },filepath)
        print(f"Preprocessor saved to {filepath}")

    def load(self, filepath='models/preprocessor.pkl'):
        data = joblib.load(filepath)
        self.scaler = data['scaler']
        self.label_encoders = data['label_encoders']
        print(f"Preprocessor loaded from {filepath}")