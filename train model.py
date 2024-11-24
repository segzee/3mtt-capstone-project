import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Load and Prepare Data ---
def load_data(filepath):
    """
    Load data from a CSV file and return a DataFrame.
    """
    return pd.read_csv(filepath)

def preprocess_data(df, numeric_cols, categorical_cols, target):
    """
    Handle missing values, outliers, and split the data into train and test sets.
    """
    # Handle missing values and outliers
    for col in numeric_cols:
        df[col].fillna(df[col].mean(), inplace=True)
        df[col] = np.where(df[col] > 1e10, df[col].mean(), df[col])

    # Features (X) and Target (y)
    X = df[numeric_cols + categorical_cols]
    y = df[target]

    # Split into Train and Test Sets
    return train_test_split(X, y, test_size=0.2, random_state=42)

# --- Define Preprocessing Pipelines ---
def create_preprocessor(numeric_cols, categorical_cols):
    """
    Create preprocessing pipelines for numeric and categorical data.
    """
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

# --- Define Model and Hyperparameter Search ---
def create_model_pipeline(preprocessor):
    """
    Create a model pipeline with preprocessing and regressor.
    """
    return Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', RandomForestRegressor(random_state=42))])

def perform_hyperparameter_search(pipeline, X_train, y_train):
    """
    Perform Randomized Search for hyperparameter optimization.
    """
    param_dist = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_features': ['sqrt', 'log2'],
        'regressor__max_depth': [10, 20, 30, None],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4],
        'regressor__bootstrap': [True, False]
    }

    random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=20, 
                                       cv=3, verbose=2, random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)
    return random_search

# --- Evaluate Model ---
def evaluate_model(y_true, y_pred, dataset="Test"):
    """
    Calculate and print evaluation metrics.
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{dataset} Metrics:")
    print(f" - Mean Squared Error (MSE): {mse:.4f}")
    print(f" - Mean Absolute Error (MAE): {mae:.4f}")
    print(f" - R^2 Score: {r2:.4f}\n")
    return mse, mae, r2

# --- Save Model ---
def save_model(model, output_dir, filename="random_forest_model.pkl"):
    """
    Save the trained model to the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, filename)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

# --- Main Script ---
if __name__ == "__main__":
    # Filepath to dataset (replace 'data.csv' with your actual file)
    filepath = "data.csv"

    # Define columns
    numeric_cols = ['Deaths', 'Recovered', 'Active', 'New cases', 'New deaths', 'New recovered',
                    'Confirmed last week', '1 week change', '1 week % increase']
    categorical_cols = ['WHO Region']
    target = 'Confirmed'

    # Load and preprocess data
    df = load_data(filepath)
    X_train, X_test, y_train, y_test = preprocess_data(df, numeric_cols, categorical_cols, target)

    # Create preprocessor and pipeline
    preprocessor = create_preprocessor(numeric_cols, categorical_cols)
    pipeline = create_model_pipeline(preprocessor)

    # Train model with hyperparameter tuning
    print("Starting training with hyperparameter tuning...")
    random_search = perform_hyperparameter_search(pipeline, X_train, y_train)
    print("Training completed!")
    print(f"Best Hyperparameters: {random_search.best_params_}")

    # Evaluate model
    best_model = random_search.best_estimator_
    evaluate_model(y_train, best_model.predict(X_train), dataset="Train")
    evaluate_model(y_test, best_model.predict(X_test), dataset="Test")

    # Save the trained model
    save_model(best_model, output_dir="models/")
