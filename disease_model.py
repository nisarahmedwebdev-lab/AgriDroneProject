"""
disease_model.py - Train and save Random Forest classifier
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from config import FEATURE_NAMES, DISEASE_CLASSES

def generate_synthetic_data(n_samples=5000):
    """
    Generate synthetic training data for disease classification.
    Returns feature matrix and labels.
    """
    np.random.seed(42)
    
    X = []
    y = []
    
    for _ in range(n_samples):
        # Random NDVI value (0 to 1)
        ndvi = np.random.uniform(0, 1)
        
        # Correlated features based on NDVI
        if ndvi > 0.6:
            # Healthy: high NDVI, good color, low texture variance, good moisture
            red = np.random.normal(0.1, 0.05)
            green = np.random.normal(0.4, 0.05)
            texture = np.random.normal(0.1, 0.03)
            moisture = np.random.normal(0.7, 0.05)
            label = 0  # Healthy
        elif ndvi > 0.3:
            # Early Disease: medium NDVI, some discoloration
            red = np.random.normal(0.3, 0.08)
            green = np.random.normal(0.3, 0.08)
            texture = np.random.normal(0.3, 0.08)
            moisture = np.random.normal(0.5, 0.08)
            label = 1  # Early Disease
        else:
            # Severe Disease: low NDVI, severe discoloration
            red = np.random.normal(0.6, 0.1)
            green = np.random.normal(0.1, 0.05)
            texture = np.random.normal(0.7, 0.1)
            moisture = np.random.normal(0.2, 0.08)
            label = 2  # Severe Disease
        
        # Add some noise
        features = [
            ndvi + np.random.normal(0, 0.02),
            max(0, min(1, red + np.random.normal(0, 0.02))),
            max(0, min(1, green + np.random.normal(0, 0.02))),
            max(0, min(1, texture + np.random.normal(0, 0.02))),
            max(0, min(1, moisture + np.random.normal(0, 0.02)))
        ]
        
        X.append(features)
        y.append(label)
    
    return np.array(X), np.array(y)

def train_and_save_model():
    """
    Train Random Forest classifier and save to disk.
    """
    print("Generating synthetic training data...")
    X, y = generate_synthetic_data(5000)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest
    print("Training Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=list(DISEASE_CLASSES.values())))
    
    # Feature importance
    importance = clf.feature_importances_
    print("\nFeature Importance:")
    for name, imp in zip(FEATURE_NAMES, importance):
        print(f"  {name}: {imp:.4f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, 'models/disease_clf.pkl')
    print("\nModel saved to models/disease_clf.pkl")
    
    return clf

if __name__ == "__main__":
    train_and_save_model()