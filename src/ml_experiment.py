import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler # Added for potential feature scaling
import joblib
import os
from src.utils import generate_synthetic_ml_data, plot_confusion_matrix

def run_ml_experiment():
    print("\n--- Running ML Experiment (Binary Black Hole Candidate Classification) ---")
    data_filepath = 'data/ml_stellar_data.csv'
    if not os.path.exists('data'): os.makedirs('data')

    if not os.path.exists(data_filepath):
        df = generate_synthetic_ml_data(n_samples=2000) # Generate more samples for ML
        df.to_csv(data_filepath, index=False)
    else:
        df = pd.read_csv(data_filepath)

    if df.empty:
        print("Error: ML data is empty. Skipping experiment.")
        return
    if 'is_black_hole_candidate' not in df.columns:
        print("Error: Target column 'is_black_hole_candidate' not found in ML data. Skipping experiment.")
        return


    features = ['stellar_mass', 'luminosity', 'distance_kpc', 'x_ray_activity_index']
    target = 'is_black_hole_candidate'

    # Check if all features are present
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"Error: Missing features in ML data: {missing_features}. Skipping experiment.")
        return


    X = df[features]
    y = df[target]

    # Stratify split is good for imbalanced datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Optional: Scale features (RandomForest is less sensitive, but good practice for other models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training RandomForestClassifier...")
    # Add class_weight='balanced' for imbalanced datasets
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    model.fit(X_train_scaled, y_train) # Use scaled data

    y_pred = model.predict(X_test_scaled) # Use scaled data for prediction

    # Define class names for the binary classification
    class_names_ml = ['Not BH Candidate', 'BH Candidate']
    report = classification_report(y_test, y_pred, target_names=class_names_ml, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0,1]) # Explicitly define labels for CM

    print("\nClassification Report (ML Experiment):\n", report)
    print("\nConfusion Matrix (ML Experiment):\n", cm)

    if not os.path.exists('results'): os.makedirs('results')
    with open('results/ml_metrics.txt', 'w') as f:
        f.write("ML Experiment Results (Binary Classification):\n"); f.write(report); f.write("\nConfusion Matrix:\n"); f.write(str(cm))
    plot_confusion_matrix(cm, classes=class_names_ml, filename='results/ml_confusion_matrix.png')

    model_filepath = 'models/ml_model.pkl'
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(model, model_filepath) # Save the trained model
    joblib.dump(scaler, 'models/ml_scaler.pkl') # Save the scaler too
    print(f"ML Model saved to {model_filepath}")
    print(f"ML Scaler saved to models/ml_scaler.pkl")
    print("ML Experiment finished.")