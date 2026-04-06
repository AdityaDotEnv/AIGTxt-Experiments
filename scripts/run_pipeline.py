import pandas as pd
import warnings

# Suppress minor warnings for clean output
warnings.filterwarnings('ignore')

# Import previously generated modular scripts
from data_exploration import load_and_clean
from preprocessing import preprocess_and_split
from model_training import train_baseline, train_interpretable, template_distilbert_finetuning
from evaluation import evaluate_predictions, extract_feature_importance, perform_error_analysis

def main():
    print("="*60)
    print("      AIGTxt Classification Pipeline (Phases 1-4)     ")
    print("="*60)
    
    DATA_PATH = "../data/AIGTxt.xlsx"
    
    # ---------------------------------------------------------
    # PHASE 1: Data Exploration (Load data)
    # ---------------------------------------------------------
    df = load_and_clean(DATA_PATH)
    
    # ---------------------------------------------------------
    # PHASE 2: Preprocessing & Feature Engineering
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test, vectorizer, indices_test = preprocess_and_split(df)
    
    # ---------------------------------------------------------
    # PHASE 3: Model Selection & Training
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("Phase 3: Model Training")
    print("="*60)
    
    # 1. Baseline Model (Naive Bayes)
    nb_model = train_baseline(X_train, y_train)
    
    # 2. Interpretable Model (Random Forest)
    rf_model = train_interpretable(X_train, y_train)
    
    # 3. Advanced Model (Template)
    template_distilbert_finetuning(df)
    
    # ---------------------------------------------------------
    # PHASE 4: Evaluation & Interpretation
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("Phase 4: Evaluation (Using Random Forest)")
    print("="*60)
    
    # Get predictions from Interpretable model
    y_pred = rf_model.predict(X_test)
    labels = sorted(list(set(y_test))) # e.g., ['AI', 'Human', 'Mixed']
    
    # 1. Confusion Matrix & Report
    evaluate_predictions(y_test, y_pred, labels)
    
    # 2. Feature Importance
    extract_feature_importance(rf_model, vectorizer)
    
    # 3. Error Analysis
    perform_error_analysis(df, indices_test, y_test, y_pred, df_features=df, n_examples=5)

if __name__ == "__main__":
    main()
