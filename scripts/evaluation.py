import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_predictions(y_test, y_pred, labels):
    print("\n--- Phase 4: Evaluation & Interpretation ---")
    
    print("\n1. Classification Report:")
    report = classification_report(y_test, y_pred, output_dict=False)
    print(report)
    
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    
    print("\n2. 'Mixed' Class Analysis:")
    if 'Mixed' in report_dict:
        mixed_f1 = report_dict['Mixed']['f1-score']
        print(f"   F1-score for 'Mixed' class: {mixed_f1:.4f}")
        if 'AI' in report_dict and 'Human' in report_dict:
            print(f"   Compared to AI: {report_dict['AI']['f1-score']:.4f}, Human: {report_dict['Human']['f1-score']:.4f}")
            if mixed_f1 < min(report_dict['AI']['f1-score'], report_dict['Human']['f1-score']):
                print("   -> The model struggles most with 'Mixed' text, likely because its features smoothly interpolate between pure AI and pure Human styles without distinct defining markers.")
    else:
        print("   'Mixed' class not present in the test set or predictions.")
    
    print("\n3. Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig("confusion_matrix.png")
    print("   Saved Confusion Matrix as 'confusion_matrix.png'")

def extract_feature_importance(model, vectorizer, top_n=20):
    print("\n--- Feature Importance Analysis ---")
    
    # 1. Get feature names
    tfidf_feature_names = vectorizer.get_feature_names_out()
    custom_feature_names = ['avg_sent_len', 'vocab_diversity']
    all_feature_names = list(tfidf_feature_names) + custom_feature_names
    
    # 2. Get importances
    importances = model.feature_importances_
    
    # Check length mismatch
    if len(all_feature_names) != len(importances):
        print(f"Warning: Expected {len(importances)} feature names, got {len(all_feature_names)}.")
        return
        
    feat_imp = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)
    
    print(f"\nTop {top_n} Strongest Predictors of Classification:")
    print(feat_imp.head(top_n))
    
    # Check custom feature rankings
    for custom_feat in custom_feature_names:
        if custom_feat in feat_imp.index:
            rank = feat_imp.index.get_loc(custom_feat) + 1
            print(f"\nCustom Feature '{custom_feat}' ranked #{rank} out of {len(all_feature_names)} in importance.")

def perform_error_analysis(df, indices_test, y_test, y_pred, df_features, n_examples=5):
    print("\n--- Error Analysis ---")
    
    # Assemble test results
    test_df = df.loc[indices_test].copy()
    test_df['True_Label'] = y_test
    test_df['Predicted_Label'] = y_pred
    
    # Merge custom features from the df we computed them in (df_features)
    test_df['avg_sent_len'] = df_features.loc[indices_test, 'avg_sent_len']
    test_df['vocab_diversity'] = df_features.loc[indices_test, 'vocab_diversity']
    
    # Filter for Human misclassified as AI and vice versa
    # Assuming 'Human' and 'AI' are the exact string labels
    confusions = test_df[((test_df['True_Label'] == 'Human') & (test_df['Predicted_Label'] == 'AI')) | 
                         ((test_df['True_Label'] == 'AI') & (test_df['Predicted_Label'] == 'Human'))]
    
    if confusions.empty:
        print("No pure Human/AI confusions found!")
        return
        
    sampled_confusions = confusions.sample(min(n_examples, len(confusions)), random_state=42)
    
    print(f"Analyzing {len(sampled_confusions)} instances where the model confused Human and AI:\n")
    
    for i, (_, row) in enumerate(sampled_confusions.iterrows(), 1):
        print(f"--- Example {i}: True {row['True_Label']} -> Predicted {row['Predicted_Label']} ---")
        
        # Print a snippet of the text
        text_snippet = str(row['text'])[:300] + "..." if len(str(row['text'])) > 300 else str(row['text'])
        print(f"Text Snippet: {text_snippet}")
        
        print("\nDiagnostic Hypothesis:")
        # Try to guess why it was confused
        if row['True_Label'] == 'Human' and row['Predicted_Label'] == 'AI':
            print("Why it failed: The human text here may use unusually formal/repetitive language structure or dense technical jargon, masquerading as AI style.")
            print(f"Notice the sentence complexity: Sent length = {row['avg_sent_len']:.2f}, Vocab diversity = {row['vocab_diversity']:.2f}")
        elif row['True_Label'] == 'AI' and row['Predicted_Label'] == 'Human':
            print("Why it failed: The AI may have been specifically prompted to mimic an informal or highly varied human tone, breaking standard AI 'tells'.")
            print(f"Notice the sentence complexity: Sent length = {row['avg_sent_len']:.2f}, Vocab diversity = {row['vocab_diversity']:.2f}")
        print("\n")
