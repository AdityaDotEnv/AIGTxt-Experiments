import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Ensure NLTK resources are available
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def load_and_clean(file_path):
    print(f"--- Loading data from {file_path} ---")
    df = pd.read_excel(file_path)
    
    id_vars = []
    if 'Domain' in df.columns:
        id_vars.append('Domain')
        
    value_vars = [c for c in ['Human-Generated', 'ChatGPT-Generated', 'Mixed Text'] if c in df.columns]
    df_long = pd.melt(df, id_vars=id_vars, value_vars=value_vars, 
                      var_name='label', value_name='text')
    
    label_map = {'Human-Generated': 'Human', 'ChatGPT-Generated': 'AI', 'Mixed Text': 'Mixed'}
    df_long['label'] = df_long['label'].map(label_map)
    
    # Check for missing values
    initial_count = len(df_long)
    df = df_long.dropna(subset=['text', 'label'])
    # Remove empty or whitespace-only strings
    df = df[df['text'].astype(str).str.strip().astype(bool)]
    
    dropped = initial_count - len(df)
    print(f"Dropped {dropped} rows due to missing values or empty strings.")
    return df

def analyze_distributions(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(x='label', data=df, palette='viridis')
    plt.title('Distribution of Text Labels (AI vs. Human vs. Mixed)')
    plt.show()
    print("\nLabel Counts:")
    print(df['label'].value_counts(normalize=True) * 100)

def analyze_lengths(df):
    # Calculate metrics
    df['char_count'] = df['text'].apply(len)
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.boxplot(ax=axes[0], x='label', y='char_count', data=df)
    axes[0].set_title('Character Count by Label')
    
    sns.boxplot(ax=axes[1], x='label', y='word_count', data=df)
    axes[1].set_title('Word Count by Label')
    
    plt.tight_layout()
    plt.show()
    
    print("\nSummary Statistics for Word Counts:")
    print(df.groupby('label')['word_count'].describe())

def get_top_n_words(texts, n=20):
    stop_words = set(stopwords.words('english'))
    words = []
    for text in texts:
        # Simple tokenization: lowercase and remove non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        words.extend([w for w in tokens if w not in stop_words])
    
    return Counter(words).most_common(n)

def identify_scientific_noise(df):
    print("\n--- Potential Scientific Noise Detection ---")
    # Patterns for: Figures, Tables, LaTeX symbols ($...$), and Citations ([1], (2020))
    patterns = {
        'Figure/Table refs': r'(Figure|Fig\.|Table)\s+\d+',
        'LaTeX symbols': r'\$.*?\$|\\begin\{.*?\}',
        'Citations': r'\[\d+\]|\(\d{4}\)',
        'Section Markers': r'Abstract:|Introduction:|Methodology:'
    }
    
    noise_results = {}
    for label in df['label'].unique():
        subset = df[df['label'] == label]['text']
        label_noise = {}
        for name, pattern in patterns.items():
            count = subset.str.contains(pattern, regex=True, case=False).sum()
            label_noise[name] = count
        noise_results[label] = label_noise

    noise_df = pd.DataFrame(noise_results)
    print(noise_df)
    print("\n*If one label has significantly higher noise counts, the model may bias toward it.*")

if __name__ == "__main__":
    DATA_PATH = "data/AIGTxt.xlsx"
    
    # Execute Pipeline
    data = load_and_clean(DATA_PATH)
    analyze_distributions(data)
    analyze_lengths(data)
    
    print("\n--- Top 20 Words per Category ---")
    for label in data['label'].unique():
        top_words = get_top_n_words(data[data['label'] == label]['text'])
        print(f"\nLabel: {label}")
        print(", ".join([f"{word} ({count})" for word, count in top_words]))
    
    identify_scientific_noise(data)