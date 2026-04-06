import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

def evaluate_with_cv(model, X, y, cv=5):
    """
    Evaluates a model using Stratified K-Fold cross validation.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    
    print(f"   CV Accuracies: {scores}")
    print(f"   Mean Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    return scores

def train_baseline(X_train, y_train):
    print("--- Training Baseline Model (Multinomial Naive Bayes) ---")
    nb = MultinomialNB()
    
    # We validate using CV first
    evaluate_with_cv(nb, X_train, y_train)
    
    # Then fit on whole train
    nb.fit(X_train, y_train)
    return nb

def train_interpretable(X_train, y_train):
    print("--- Training Interpretable Model (Random Forest) ---")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    
    evaluate_with_cv(rf, X_train, y_train)
    
    rf.fit(X_train, y_train)
    return rf

# ==========================================
# ADVANCED MODEL TEMPLATE: DistilBERT
# ==========================================
"""
To use this template for fine-tuning DistilBERT on your dataset:
1. Ensure you have installed transformers, datasets, and torch:
   pip install transformers datasets torch

2. You will pass the raw original text from `df['text']` and the encoded version of `df['label']` to it.
"""

def template_distilbert_finetuning(df, text_col='text', label_col='label'):
    print("--- [Template] DistilBERT Fine-Tuning ---")
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from datasets import Dataset
    import torch
    
    # 1. Map string labels to integers
    labels = df[label_col].unique()
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    
    df['encoded_labels'] = df[label_col].map(label2id)
    
    # 2. Convert pandas dataframe to HuggingFace Dataset
    # Wait, we should split first!
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['encoded_labels'], random_state=42)
    
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # 3. Load Tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples[text_col], padding="max_length", truncation=True, max_length=512)
        
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    test_tokenized = test_dataset.map(tokenize_function, batched=True)
    
    train_tokenized = train_tokenized.rename_column("encoded_labels", "labels")
    test_tokenized = test_tokenized.rename_column("encoded_labels", "labels")
    train_tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # 4. Load Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )
    
    # 5. Define Training Arguments
    training_args = TrainingArguments(
        output_dir="./distilbert_results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        tokenizer=tokenizer,
    )
    
    # 7. Train and Evaluate
    trainer.train()
    print("DistilBERT evaluation results:")
    print(trainer.evaluate())
    
    # return model, trainer
    """
    print("Template loaded! See comments to execute this for Advanced Model Phase.")
