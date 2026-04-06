import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import numpy as np

# Ensure necessary NLTK models are downloaded
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def clean_text(text):
    """
    Remove LaTeX artifacts and citations but keep punctuation.
    (Punctuation density can be an AI 'tell').
    """
    if not isinstance(text, str):
        return ""
        
    # Remove LaTeX math environments ($...$, $$...$$)
    text = re.sub(r'\$.*?\$', ' ', text)
    
    # Basic attempt to remove LaTeX tags like \begin{figure}
    text = re.sub(r'\\[a-zA-Z]+\{.*?\}', ' ', text)
    
    # Remove typical citations
    # Bracketed numbers like [1], [1, 2], [1-3]
    text = re.sub(r'\[\s*\d+(?:\s*,\s*\d+)*\s*(?:-\s*\d+)?\s*\]', ' ', text)
    # Year in parentheses (2020)
    text = re.sub(r'\(\d{4}[a-zA-Z]?\)', ' ', text)
    # Author(s), Year citations limits (Smith et al., 2020)
    text = re.sub(r'\([A-Za-z\s\.,&]+,\s*\d{4}[a-zA-Z]?\)', ' ', text)
    
    # Condense multiple spaces into one, trim edges
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_sentence_complexity(text):
    """
    Compute sentence complexity via two features:
      1. Average sentence length (in words).
      2. Vocabulary diversity (unique words / total words).
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0, 0.0
        
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    
    # We filter for actual alphanumeric words to compute metrics robustly
    alphanumeric_words = [w.lower() for w in words if w.isalnum()]
    
    if len(sentences) == 0 or len(alphanumeric_words) == 0:
        return 0.0, 0.0
        
    avg_sentence_len = len(alphanumeric_words) / len(sentences)
    unique_words = set(alphanumeric_words)
    vocab_diversity = len(unique_words) / len(alphanumeric_words)
    
    return avg_sentence_len, vocab_diversity

def preprocess_and_split(df, text_col='text', label_col='label'):
    """
    Runs the full Phase 2 preprocessing pipeline.
    """
    print("--- Phase 2: Preprocessing & Feature Engineering ---")
    
    print("1. Cleaning text (LaTeX and Citations)...")
    df['clean_text'] = df[text_col].apply(clean_text)
    
    print("2. Calculating 'Sentence Complexity'...")
    complexities = df['clean_text'].apply(calculate_sentence_complexity)
    df['avg_sent_len'] = [c[0] for c in complexities]
    df['vocab_diversity'] = [c[1] for c in complexities]
    
    print("3. TF-IDF Vectorization (ngram_range=(1,3), 10,000 features)...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(df['clean_text'])
    
    custom_features = df[['avg_sent_len', 'vocab_diversity']].values
    
    print("4. Combining features...")
    # Add our 2 custom features at the end of the TF-IDF matrix
    X = hstack((tfidf_matrix, custom_features)).tocsr()
    y = df[label_col].values
    
    print(f"   Final feature matrix shape: {X.shape}")
    
    # We also keep the original text index so we can retrieve original examples for error analysis later.
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X, y, df.index, test_size=0.2, stratify=y, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, vectorizer, indices_test
