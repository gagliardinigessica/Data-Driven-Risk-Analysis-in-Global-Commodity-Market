"""
Machine learning models and training utilities.
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os


def train_model(X, y, test_size=0.2, random_state=42):
    """
    Train a Random Forest classifier.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (model, scaler, X_train, X_test, y_train, y_test)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(random_state=random_state, n_estimators=100)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X_train_scaled, X_test_scaled, y_train, y_test


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_train: Training features
        X_test: Test features
        y_train: Training targets
        y_test: Test targets
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    # Check for overfitting
    gap = train_acc - test_acc
    
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'overfitting_gap': gap,
        'classification_report': report,
        'confusion_matrix': cm
    }


def save_model(model, scaler, filepath):
    """
    Save model and scaler to disk.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        filepath (str): Path to save the model
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(project_root, filepath)
    
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    with open(full_path, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)

