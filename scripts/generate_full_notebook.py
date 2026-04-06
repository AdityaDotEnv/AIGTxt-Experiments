import nbformat
from nbclient import NotebookClient
import os

def create_notebook():
    nb = nbformat.v4.new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    }

    cells = []

    # --- CELL 1: Introduction ---
    cells.append(nbformat.v4.new_markdown_cell("""# AIGTxt Classification: AI vs. Human vs. Mixed
This notebook implements a complete NLP pipeline to classify scientific text into three categories: **AI-generated**, **Human-generated**, and **Mixed**.

### Pipeline Phases:
1. **Data Exploration & Reshaping**: Loading from Excel and converting to long-format.
2. **Preprocessing & Feature Engineering**: Removing LaTeX/citations and calculating custom 'Sentence Complexity' features.
3. **Model Selection & Training**: Comparing Multinomial Naive Bayes (Baseline) with Random Forest (Interpretable).
4. **Evaluation & Interpretation**: Analyzing the 'Mixed' class, extracting feature importance, and performing error analysis.
"""))

    # --- CELL 2: Setup & Imports ---
    cells.append(nbformat.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import hstack
import warnings

warnings.filterwarnings('ignore')

# Ensure NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
"""))

    # --- CELL 3: Phase 1 - Loading ---
    cells.append(nbformat.v4.new_markdown_cell("## Phase 1: Data Exploration & Loading"))
    cells.append(nbformat.v4.new_code_cell("""DATA_PATH = "../data/AIGTxt.xlsx"
df = pd.read_excel(DATA_PATH)

# Reshape from Wide to Long format
id_vars = ['Domain'] if 'Domain' in df.columns else []
value_vars = [c for c in ['Human-Generated', 'ChatGPT-Generated', 'Mixed Text'] if c in df.columns]

df_long = pd.melt(df, id_vars=id_vars, value_vars=value_vars, var_name='label', value_name='text')
label_map = {'Human-Generated': 'Human', 'ChatGPT-Generated': 'AI', 'Mixed Text': 'Mixed'}
df_long['label'] = df_long['label'].map(label_map)

# Initial Cleaning
df = df_long.dropna(subset=['text', 'label'])
df = df[df['text'].astype(str).str.strip().astype(bool)]

print(f"Loaded {len(df)} samples across {df['label'].nunique()} classes.")
df['label'].value_counts().plot(kind='bar', title='Class Distribution')
plt.show()
"""))

    # --- CELL 4: Phase 2 - Preprocessing ---
    cells.append(nbformat.v4.new_markdown_cell("## Phase 2: Preprocessing & Feature Engineering"))
    cells.append(nbformat.v4.new_code_cell("""def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'\$.*?\$', ' ', text) # LaTeX math
    text = re.sub(r'\\\\[a-zA-Z]+\\{.*?\\}', ' ', text) # LaTeX tags
    text = re.sub(r'\\[\\d+\\]|\\(\\d{4}\\)', ' ', text) # Citations
    return re.sub(r'\\s+', ' ', text).strip()

def calculate_complexity(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    alnum_words = [w.lower() for w in words if w.isalnum()]
    if not sentences or not alnum_words: return 0.0, 0.0
    return len(alnum_words)/len(sentences), len(set(alnum_words))/len(alnum_words)

df['clean_text'] = df['text'].apply(clean_text)
complexities = df['clean_text'].apply(calculate_complexity)
df['avg_sent_len'] = [c[0] for c in complexities]
df['vocab_diversity'] = [c[1] for c in complexities]

vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)
X_tfidf = vectorizer.fit_transform(df['clean_text'])
X_custom = df[['avg_sent_len', 'vocab_diversity']].values
X = hstack((X_tfidf, X_custom)).tocsr()
y = df['label'].values

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.2, stratify=y, random_state=42
)
print(f"Feature matrix shape: {X.shape}")
"""))

    # --- CELL 5: Phase 3 - Training ---
    cells.append(nbformat.v4.new_markdown_cell("## Phase 3: Model Selection & Training"))
    cells.append(nbformat.v4.new_code_cell("""skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("--- Baseline: Naive Bayes ---")
nb = MultinomialNB()
nb_scores = cross_val_score(nb, X_train, y_train, cv=skf)
print(f"CV Accuracy: {nb_scores.mean():.4f}")
nb.fit(X_train, y_train)

print("\\n--- Interpretable: Random Forest ---")
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_scores = cross_val_score(rf, X_train, y_train, cv=skf)
print(f"CV Accuracy: {rf_scores.mean():.4f}")
rf.fit(X_train, y_train)
"""))

    # --- CELL 6: Phase 4 - Evaluation ---
    cells.append(nbformat.v4.new_markdown_cell("## Phase 4: Evaluation & Interpretation"))
    cells.append(nbformat.v4.new_code_cell("""y_pred = rf.predict(X_test)
labels = sorted(list(set(y_test)))

print("Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.show()
"""))

    # --- CELL 7: Phase 4 - Insights ---
    cells.append(nbformat.v4.new_markdown_cell("### Feature Importance & Error Analysis"))
    cells.append(nbformat.v4.new_code_cell("""# Feature Importance
feat_names = list(vectorizer.get_feature_names_out()) + ['avg_sent_len', 'vocab_diversity']
imps = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False)
print("Top 15 Features:")
print(imps.head(15))

# Error Analysis
test_res = df.loc[idx_test].copy()
test_res['pred'] = y_pred
errors = test_res[test_res['label'] != test_res['pred']].sample(3, random_state=42)

print("\\n--- Error Samples ---")
for _, row in errors.iterrows():
    print(f"True: {row['label']} | Pred: {row['pred']}")
    print(f"Text Snippet: {str(row['text'])[:150]}...")
    print("-" * 30)
"""))

    nb.cells = cells
    
    # --- EXECUTION ---
    print("Executing notebook cells using 'python3' kernel...")
    client = NotebookClient(nb, timeout=600, kernel_name='python3')
    client.execute()
    
    # --- SAVE ---
    output_path = "../notebooks/AIGTxt_Classification_Pipeline.ipynb"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Notebook successfully created and executed at: {output_path}")

if __name__ == "__main__":
    create_notebook()
