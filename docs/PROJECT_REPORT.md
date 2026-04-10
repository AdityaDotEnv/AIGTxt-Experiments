# Detection and Classification of AI-Generated Scientific Text Using Machine Learning

### A Project Report on the AIGTxt Classification Pipeline

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Problem Statement](#2-problem-statement)
3. [Existing Systems & Literature Review](#3-existing-systems--literature-review)
4. [Proposed System & Methodology](#4-proposed-system--methodology)
   - 4.1 [Dataset Description](#41-dataset-description)
   - 4.2 [Data Preprocessing & Reshaping](#42-data-preprocessing--reshaping)
   - 4.3 [Feature Engineering](#43-feature-engineering)
   - 4.4 [Classification Algorithms](#44-classification-algorithms)
   - 4.5 [Evaluation Framework](#45-evaluation-framework)
5. [Result Analysis](#5-result-analysis)
   - 5.1 [Cross-Validation Performance](#51-cross-validation-performance)
   - 5.2 [Test Set Evaluation](#52-test-set-evaluation)
   - 5.3 [Confusion Matrix Analysis](#53-confusion-matrix-analysis)
   - 5.4 [The "Mixed" Class Challenge](#54-the-mixed-class-challenge)
   - 5.5 [Feature Importance Analysis](#55-feature-importance-analysis)
   - 5.6 [Error Analysis & Diagnostic Hypotheses](#56-error-analysis--diagnostic-hypotheses)
6. [Conclusion](#6-conclusion)
7. [Future Scope](#7-future-scope)
8. [References](#8-references)

---

## 1. Abstract

The emergence of powerful Large Language Models (LLMs) such as OpenAI's ChatGPT, Google's Gemini, and Meta's LLaMA has fundamentally disrupted the landscape of scientific writing and academic publishing. These models are capable of producing text that is syntactically correct, semantically coherent, and often indistinguishable from human-authored prose—posing a significant threat to academic integrity, peer review reliability, and the trustworthiness of scientific literature.

This project presents a comprehensive machine learning pipeline for the **three-class classification** of scientific text into **AI-generated**, **Human-written**, and **Mixed** (human-AI collaborative) categories. The pipeline operates on the **AIGTxt** dataset, a curated corpus of scientific passages spanning multiple academic domains. Our approach combines **Term Frequency–Inverse Document Frequency (TF-IDF)** vectorization with bigram support, custom **linguistic feature engineering** (average sentence length and vocabulary diversity), and two interpretable classification algorithms—**Multinomial Naive Bayes** (as a probabilistic baseline) and **Random Forest** (as an ensemble learner).

The Random Forest model achieves a test accuracy of approximately **44%**, significantly outperforming the Naive Bayes baseline at **39%**. Detailed analysis reveals that while binary Human-vs-AI classification is moderately successful (F1 ≈ 0.52 for AI, F1 ≈ 0.67 for Human), the **Mixed class** presents a fundamental challenge for frequency-based approaches, achieving an F1-score of only **0.05**. Feature importance analysis identifies **vocabulary diversity** as the single most discriminative feature, followed by sentence-level structural metrics and domain-specific bigrams.

Through systematic error analysis and diagnostic hypothesis generation, we demonstrate the inherent limitations of bag-of-words models for this task and motivate the transition to contextual embedding approaches (e.g., Transformer-based models such as DistilBERT and SciBERT) for future work.

---

## 2. Problem Statement

### 2.1 Background & Motivation

The rapid advancement in generative artificial intelligence has created a paradigm shift in how text content is produced, consumed, and verified. Since the public release of ChatGPT in November 2022, LLMs have been widely adopted across industries—including scientific research and academia. Researchers increasingly use AI assistants for drafting manuscripts, generating literature reviews, paraphrasing content, and even producing entire sections of research papers.

While AI-assisted writing can improve productivity and accessibility, it raises critical concerns:

- **Academic Integrity**: Universities and conferences require original intellectual contributions. Unattributed AI-generated content undermines the foundational principles of scholarly work.
- **Peer Review Trustworthiness**: Reviewers depend on the assumption that submitted manuscripts reflect genuine human expertise and reasoning. AI-generated passages may introduce factual errors ("hallucinations"), biased reasoning, or superficially correct but scientifically meaningless statements that are difficult to detect through human review alone.
- **Scientific Reproducibility**: If critical methodology sections or result interpretations are AI-generated without proper verification, the reproducibility and reliability of published findings are compromised.
- **Intellectual Property & Authorship**: The question of whether an AI system qualifies as an "author" remains legally and ethically unresolved, with major publishers (e.g., Nature, Science, IEEE) adopting varying policies.

### 2.2 The Three-Class Problem

Existing research on AI text detection has predominantly focused on the **binary classification** problem: distinguishing purely AI-generated text from purely human-written text. However, the real-world scenario is more nuanced. In practice, much scientific writing exists on a **spectrum**:

- **Purely Human**: The author independently conceived, drafted, and revised the text.
- **Purely AI**: The text was generated entirely by an LLM with minimal or no human editing.
- **Mixed / Collaborative**: The author used an LLM as a co-writing tool—perhaps generating an initial draft with ChatGPT and then substantially editing, restructuring, or augmenting it with original content.

The **Mixed** class is the most practically relevant and simultaneously the most challenging to detect. Mixed text blends the stylistic signatures of both human and AI authorship, creating ambiguity in the feature space that binary detectors cannot resolve.

### 2.3 Problem Definition

**Given** a corpus of scientific text passages labeled as `AI`, `Human`, or `Mixed`:

- **Objective**: Build a supervised machine learning pipeline capable of classifying unseen scientific passages into one of the three categories.
- **Constraints**: The solution must be interpretable (i.e., practitioners should understand *why* a given text is classified as AI, Human, or Mixed) and reproducible from publicly available tools and libraries.
- **Evaluation**: Performance is measured using Precision, Recall, F1-Score (per class), and overall accuracy on a stratified held-out test set.

---

## 3. Existing Systems & Literature Review

The problem of AI-generated text detection has attracted substantial research attention since 2023. Below we review the major categories of existing approaches, their strengths, and their limitations.

### 3.1 Statistical & Stylometric Methods

The earliest approaches to authorship attribution—long predating LLMs—relied on **stylometry**: the statistical analysis of linguistic style. Key features include:

- **Lexical diversity metrics**: Type-Token Ratio (TTR), Hapax Legomena ratio (proportion of words appearing exactly once), vocabulary richness indices (Yule's K, Simpson's D).
- **Syntactic complexity**: Average sentence length, clause depth, use of subordinate conjunctions, passive voice frequency.
- **Function word distributions**: Frequencies of articles, prepositions, pronouns—features that are largely content-independent and reflect an author's unconscious stylistic habits.

**Strengths**: Domain-agnostic; computationally inexpensive; interpretable.
**Limitations**: Easily defeated by paraphrasing tools; limited discriminative power when AI models are specifically fine-tuned to mimic human stylistic distributions.

Notable works in this tradition include Mosteller & Wallace (1964) on the Federalist Papers and more recently, Uchendu et al. (2020) on distinguishing GPT-2 outputs.

### 3.2 Perplexity & Probability-Based Methods

These methods exploit the fact that LLMs assign high probability to text they could have generated. Key approaches include:

- **Perplexity scoring**: Computing the perplexity of a text under a reference language model (e.g., GPT-2). AI-generated text tends to have lower perplexity (i.e., the model "expects" the text more) compared to human-written text.
- **Log-likelihood ratio**: Comparing the probability of the text under a large model (e.g., GPT-3) versus a smaller baseline model.
- **DetectGPT** (Mitchell et al., 2023): Uses the curvature of the log probability function. AI-generated text occupies regions of negative curvature in the model's log-probability landscape, whereas human text does not.
- **Entropy-based methods**: Measuring the token-level entropy of a passage under a pretrained model. AI text tends to have more uniform (lower) entropy distributions.

**Strengths**: Theoretically principled; do not require labeled training data (zero-shot detection).
**Limitations**: Require access to the generating model's architecture or a close proxy; performance degrades significantly when the generating model differs from the detector's reference model (e.g., detecting Claude-generated text with a GPT-2-based detector); computationally expensive for large models.

### 3.3 Watermarking Approaches

Watermarking embeds a statistically detectable signal into AI-generated text at generation time:

- **Token-level watermarking** (Kirchenbauer et al., 2023): At each generation step, the vocabulary is partitioned into "green" and "red" lists based on a hash of the preceding token. The model is biased to preferentially select "green" tokens, creating a statistical signature that is invisible to human readers but detectable by anyone with knowledge of the hash function and random seed.
- **Semantic watermarking**: Embedding signals at the sentence or paragraph level by biasing toward specific syntactic structures.

**Strengths**: High detection accuracy when the watermark is intact; does not require a classifier.
**Limitations**: Requires control over the generation process (not applicable post-hoc); easily removed by paraphrasing, synonym substitution, or back-translation; raises concerns about censorship and free expression.

### 3.4 Supervised Deep Learning Methods

These methods train neural classifiers on labeled corpora of human and AI text:

- **Fine-tuned Transformers**: Models like RoBERTa, DistilBERT, or SciBERT are fine-tuned on binary (Human vs. AI) or multi-class datasets. These models leverage deep contextual embeddings that capture semantic nuances far beyond bag-of-words representations.
- **OpenAI's Text Classifier** (2023, discontinued): A fine-tuned GPT model for AI text detection; achieved approximately 26% true positive rate at 9% false positive rate—illustrating the difficulty of the problem even for state-of-the-art methods.
- **GPTZero**: A commercial tool that combines perplexity scoring, burstiness analysis, and proprietary classifiers for AI text detection.

**Strengths**: Highest accuracy among all approaches; captures deep semantic and discourse-level patterns.
**Limitations**: Require large labeled training sets; computationally expensive; act as "black boxes" without additional explainability techniques (SHAP, LIME); performance drops on domains not represented in the training data.

### 3.5 Classical Machine Learning Methods (Our Approach)

Our approach sits between pure stylometric methods and deep learning:

- **TF-IDF + Classical Classifiers**: We use TF-IDF vectorization (with bigrams) to convert text into high-dimensional sparse feature vectors, augmented with hand-crafted linguistic features. Classification is performed using Multinomial Naive Bayes and Random Forest.

**Strengths**: Computationally efficient; interpretable (feature importance is directly accessible); reproducible with standard tools (scikit-learn, NLTK); provides a strong baseline that establishes the performance floor against which deep learning methods should be compared.
**Limitations**: Cannot model long-range semantic dependencies; struggles with the Mixed class (as our results demonstrate); sensitive to vocabulary drift across domains.

### 3.6 Comparative Summary

| Approach              | Accuracy  | Labeled Data? | Interpretable? | Post-hoc? | Cost    |
|-----------------------|-----------|:------------:|:--------------:|:---------:|---------|
| Stylometric           | Low–Med   | Yes          | ✅ High        | ✅ Yes    | Low     |
| Perplexity (DetectGPT)| Medium    | No           | ⚠️ Moderate    | ✅ Yes    | High    |
| Watermarking          | High      | No           | ✅ High        | ❌ No     | Low     |
| Fine-tuned Transformer| High      | Yes          | ❌ Low         | ✅ Yes    | High    |
| **TF-IDF + RF (Ours)**| **Medium**| **Yes**      | **✅ High**    | **✅ Yes**| **Low** |

Our method occupies the niche of **high interpretability at low computational cost**, serving as both an educational tool and a performance baseline for the three-class problem.

---

## 4. Proposed System & Methodology

### 4.1 Dataset Description

The **AIGTxt** dataset is a structured Excel workbook (`AIGTxt.xlsx`) containing scientific text passages across multiple academic domains. Each row represents a single thematic topic, with three parallel text columns:

| Column               | Description                                                                     |
|----------------------|---------------------------------------------------------------------------------|
| `Human-Generated`    | A scientific passage written entirely by a human researcher on the given topic.  |
| `ChatGPT-Generated`  | A scientific passage generated entirely by ChatGPT on the same topic.            |
| `Mixed Text`         | A passage that blends human and ChatGPT authorship on the same topic.            |
| `Domain`             | The academic domain (e.g., Physics, Computer Science, Biology, Chemistry).       |

The dataset contains **3,607 rows**, each contributing three text samples (one per class), yielding approximately **10,821 samples** after reshaping and cleaning.

**Key properties:**
- **Balanced classes**: Each class has approximately the same number of samples (~3,607), eliminating class imbalance as a confounding factor.
- **Topical alignment**: For each topic row, the Human, AI, and Mixed passages address the same subject, ensuring that classification is driven by stylistic/authorial differences rather than topical content.
- **Multi-domain coverage**: Passages span diverse scientific disciplines, providing a test of cross-domain generalizability.

### 4.2 Data Preprocessing & Reshaping

#### Wide-to-Long Transformation

The raw dataset arrives in **wide format**, where each class is stored as a separate column. Machine learning algorithms require a **long format** with a single feature column and a single target column. We use the Pandas `melt()` operation:

```python
df_long = pd.melt(
    df,
    id_vars=['Domain'],
    value_vars=['Human-Generated', 'ChatGPT-Generated', 'Mixed Text'],
    var_name='label',
    value_name='text'
)
```

This transforms the data from shape `(3607, 7)` to `(10821, 3)`, where each row is an independent `(text, label, domain)` tuple.

#### Label Mapping

Technical column names are mapped to friendly class labels:
- `Human-Generated` → `Human`
- `ChatGPT-Generated` → `AI`
- `Mixed Text` → `Mixed`

#### Data Cleaning

Rows with `NaN` or empty string values in the `text` column are dropped, and the index is reset for clean sequential access.

### 4.3 Feature Engineering

Feature engineering is the process of transforming raw data into representations that machine learning models can effectively use. Our pipeline employs two complementary feature extraction strategies:

#### 4.3.1 TF-IDF Vectorization

**Term Frequency–Inverse Document Frequency (TF-IDF)** is a cornerstone of classical text representation. It assigns a weight to each term in each document based on two factors:

**Term Frequency (TF):**

$$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$$

Where $f_{t,d}$ is the raw count of term $t$ in document $d$, normalized by the total number of terms in $d$. This measures how prominent a term is within a specific document.

**Inverse Document Frequency (IDF):**

$$\text{IDF}(t) = \log\left(\frac{N}{|\{d \in D : t \in d\}|}\right) + 1$$

Where $N$ is the total number of documents and the denominator counts how many documents contain term $t$. Terms that appear in many documents (e.g., "the", "and") receive low IDF scores, while rare, discriminative terms receive high scores.

**Final TF-IDF Score:**

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

**Configuration in our pipeline:**

| Parameter        | Value       | Rationale                                                                       |
|------------------|-------------|---------------------------------------------------------------------------------|
| `max_features`   | 10,000      | Limits the vocabulary to the 10K most informative terms, reducing dimensionality and noise. |
| `ngram_range`    | (1, 2)      | Captures unigrams (single words) and bigrams (two-word phrases). Bigrams like "deep learning", "neural network", or "we propose" carry discriminative signal that isolated unigrams cannot capture. |
| `stop_words`     | 'english'   | Removes 318 common English stop words (the, is, at, which, etc.) that contribute no discriminative power. |
| `sublinear_tf`   | False       | Raw term frequency is used (no logarithmic dampening). This was chosen for simplicity. |

The resulting TF-IDF matrix has shape `(10821, 10000)` — a sparse, high-dimensional representation where each document is a 10,000-dimensional vector.

#### 4.3.2 N-Grams: Capturing Local Word Order

An **N-gram** is a contiguous sequence of $n$ items from a given text sample:

- **Unigram (n=1)**: Individual words. E.g., `["deep", "learning", "model"]`
- **Bigram (n=2)**: Two-word sequences. E.g., `["deep learning", "learning model"]`
- **Trigram (n=3)**: Three-word sequences. E.g., `["deep learning model"]`

By using `ngram_range=(1, 2)`, our vectorizer creates features for both unigrams and bigrams. This is critical because:

1. **Phrase-level semantics**: "machine learning" carries fundamentally different meaning than "machine" and "learning" independently.
2. **AI signature phrases**: LLMs exhibit detectable preferences for certain collocations (e.g., "it is important to note", "in this study we", "plays a crucial role") that are more visible at the bigram level.
3. **Domain terminology**: Scientific bigrams like "quantum computing", "gene expression", or "convolutional neural" are strong domain indicators that help the model distinguish writing contexts.

#### 4.3.3 Custom Linguistic Features

Beyond TF-IDF, we engineer two document-level features that capture **writing style** rather than word content:

**Average Sentence Length (`avg_sent_len`):**

$$\text{avg-sent-len}(d) = \frac{1}{|S_d|} \sum_{s \in S_d} |s|$$

Where $S_d$ is the set of sentences in document $d$ (obtained via NLTK's `sent_tokenize`), and $|s|$ is the number of words in sentence $s$ (via `word_tokenize`).

**Rationale**: AI-generated text tends to produce sentences of remarkably uniform length—neither too short nor too long—due to the LLM's training objective of maximizing likelihood over typical text. Human writers, by contrast, exhibit greater variance in sentence length, mixing short emphatic statements with long, complex constructions.

**Vocabulary Diversity (`vocab_diversity`):**

$$\text{vocab-diversity}(d) = \frac{|\text{unique-words}(d)|}{|\text{total-words}(d)|}$$

This is the **Type-Token Ratio (TTR)**, measuring the proportion of unique words to total words in a document.

**Rationale**: LLMs, due to their training on massive corpora and beam-search/sampling decoding strategies, often exhibit lower vocabulary diversity—repeating certain "safe" or high-probability words. Human writers, especially in scientific contexts, tend to use more varied vocabulary, employing synonyms, technical jargon, and stylistic variation.

These two features are computed for each document and appended as additional columns to the TF-IDF sparse matrix using `scipy.sparse.hstack`, yielding a final feature matrix of shape `(10821, 10002)`.

### 4.4 Classification Algorithms

#### 4.4.1 Multinomial Naive Bayes (Baseline)

**Algorithm Overview:**

Multinomial Naive Bayes is a probabilistic classifier derived from **Bayes' Theorem**:

$$P(C_k \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid C_k) \cdot P(C_k)}{P(\mathbf{x})}$$

Where:
- $P(C_k \mid \mathbf{x})$: Posterior probability of class $C_k$ given feature vector $\mathbf{x}$.
- $P(\mathbf{x} \mid C_k)$: Likelihood of observing $\mathbf{x}$ given class $C_k$.
- $P(C_k)$: Prior probability of class $C_k$ (estimated from training data).
- $P(\mathbf{x})$: Evidence (constant across classes; ignored for classification).

**The "Naive" Independence Assumption:**

The defining characteristic—and primary simplification—of Naive Bayes is the assumption that all features are **conditionally independent** given the class:

$$P(\mathbf{x} \mid C_k) = \prod_{i=1}^{n} P(x_i \mid C_k)$$

This reduces the exponential complexity of estimating the full joint distribution $P(x_1, x_2, \ldots, x_n \mid C_k)$ to a simple product of $n$ marginal distributions—each estimated from training data via maximum likelihood.

**Multinomial Variant:**

For text classification with TF-IDF features, the multinomial variant is appropriate. It models the distribution of term frequencies across classes:

$$P(x_i \mid C_k) = \frac{N_{ki} + \alpha}{\sum_{j=1}^{|V|} (N_{kj} + \alpha)}$$

Where:
- $N_{ki}$: Total count (or TF-IDF weight sum) of feature $i$ across all training documents of class $k$.
- $|V|$: Vocabulary size (10,000 in our case).
- $\alpha$: Laplace smoothing parameter (default $\alpha = 1$) to prevent zero-probability features.

**Why Naive Bayes as a Baseline?**

1. **Speed**: Training and prediction are $O(nk)$ where $n$ is vocabulary size and $k$ is the number of classes—extremely fast even for large corpora.
2. **High-dimensional robustness**: Performs well with sparse, high-dimensional data (TF-IDF vectors).
3. **Interpretability**: Per-class feature log-probabilities directly indicate which words are most associated with each class.
4. **Well-calibrated baseline**: Any model that fails to outperform Naive Bayes is not capturing meaningful feature interactions.

**Limitation for our task**: The independence assumption is pathologically violated in natural language. Words within sentences are highly correlated (e.g., "neural" co-occurs with "network"; "quantum" co-occurs with "mechanics"). This prevents NB from modeling the feature interactions that distinguish AI writing style from human style.

#### 4.4.2 Random Forest (Interpretable Ensemble)

**Algorithm Overview:**

Random Forest is an **ensemble learning** method that constructs a collection of **decision trees** during training and outputs the **majority vote** (mode) of their individual predictions.

**Core Mechanism — Bootstrap Aggregating (Bagging):**

1. For each of $B$ trees (we use $B = 200$):
   a. Draw a **bootstrap sample** (random sample with replacement) of size $n$ from the training data.
   b. Grow a decision tree on this sample. At each node split:
      - Select a random subset of $m = \lfloor\sqrt{p}\rfloor$ features from all $p$ features (default for classification).
      - Choose the best feature and split point on this subset to maximize **information gain** (equivalently, minimize **Gini impurity**).
   c. Grow the tree to full depth (no pruning by default).
2. For prediction, pass the test instance through all $B$ trees and take the majority vote.

**Gini Impurity:**

$$\text{Gini}(S) = 1 - \sum_{k=1}^{K} p_k^2$$

Where $p_k$ is the proportion of class $k$ instances in node $S$. A pure node (all one class) has Gini = 0. The split criterion selects the feature and threshold that maximize the weighted reduction in Gini impurity across the resulting child nodes.

**Feature Importance:**

Random Forest provides a natural measure of feature importance: the **Mean Decrease in Impurity (MDI)**. For each feature, its importance is computed as the total reduction in Gini impurity achieved by splits on that feature, averaged across all $B$ trees:

$$\text{Importance}(x_j) = \frac{1}{B} \sum_{b=1}^{B} \sum_{t \in T_b} \mathbb{1}[v(t) = x_j] \cdot \Delta\text{Gini}(t)$$

Where $T_b$ is the set of nodes in tree $b$, $v(t)$ is the feature used at node $t$, and $\Delta\text{Gini}(t)$ is the impurity decrease at that node. This produces a ranking of all 10,002 features by their contribution to classification—a key advantage for interpretability.

**Hyperparameters Used:**

| Parameter        | Value  | Rationale                                                            |
|------------------|--------|----------------------------------------------------------------------|
| `n_estimators`   | 200    | Sufficient ensemble size for stable performance; diminishing returns beyond ~200 trees. |
| `random_state`   | 42     | Fixed seed for reproducibility.                                      |
| `n_jobs`         | -1     | Utilizes all available CPU cores for parallel tree construction.      |
| `max_features`   | 'sqrt' | Default for classification; selects $\sqrt{10002} \approx 100$ features per split. |
| `max_depth`      | None   | Trees grown to full depth; overfitting controlled by bagging.        |
| `criterion`      | 'gini' | Gini impurity as the split quality measure.                          |

**Why Random Forest for This Task?**

1. **Feature interaction modeling**: Unlike Naive Bayes, decision trees naturally capture interactions between features (e.g., "if `vocab_diversity` is low AND the word 'crucial' appears frequently, then predict AI").
2. **Robustness**: Bagging reduces variance; the ensemble is less prone to overfitting than a single deep decision tree.
3. **Interpretability**: Feature importance scores provide directly actionable insights about which words and stylistic metrics drive classification.
4. **No feature scaling required**: Decision trees are invariant to monotonic feature transformations, so TF-IDF weights and raw linguistic features can be used without normalization.

### 4.5 Evaluation Framework

#### 4.5.1 Train-Test Split

The dataset is split into **70% training** and **30% test** sets using stratified random sampling (`stratify=y`), ensuring that both splits preserve the original class distribution. This is critical for a balanced three-class problem.

#### 4.5.2 Stratified K-Fold Cross-Validation

On the training set, we perform **5-Fold Stratified Cross-Validation** to estimate each model's generalization performance:

1. The training data is divided into 5 equal folds, each preserving class proportions.
2. The model is trained on 4 folds and evaluated on the 5th.
3. This is repeated 5 times (rotating the held-out fold).
4. The mean accuracy across all 5 folds is reported.

This protocol provides a more reliable performance estimate than a single train-test split, reducing the impact of any particular random split.

#### 4.5.3 Classification Metrics

For the final test set evaluation, we report per-class and aggregate metrics:

**Precision:**

$$\text{Precision}_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FP}_k}$$

"Of all samples the model predicted as class $k$, what fraction truly belongs to class $k$?"

**Recall (Sensitivity):**

$$\text{Recall}_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FN}_k}$$

"Of all samples that actually belong to class $k$, what fraction did the model correctly identify?"

**F1-Score:**

$$F_1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

The **harmonic mean** of Precision and Recall. It penalizes models that sacrifice one metric for the other, providing a balanced single-number summary. The harmonic mean is preferred over the arithmetic mean because it is more sensitive to low values—a model with Precision=1.0 and Recall=0.0 would have $F_1 = 0$, not 0.5.

**Confusion Matrix:**

A $K \times K$ matrix (where $K = 3$ for our classes) where entry $(i, j)$ records the number of samples with true label $i$ that were predicted as label $j$. The diagonal represents correct predictions; off-diagonal entries reveal systematic misclassification patterns.

---

## 5. Result Analysis

### 5.1 Cross-Validation Performance

| Model                    | 5-Fold CV Mean Accuracy |
|--------------------------|:-----------------------:|
| Multinomial Naive Bayes  | ~0.39 (39%)             |
| Random Forest (200 trees)| ~0.44 (44%)             |

**Interpretation**: The Random Forest model achieves approximately 5 percentage points higher cross-validation accuracy than the Naive Bayes baseline. In a three-class balanced problem, random guessing would yield ~33.3% accuracy. Both models exceed this baseline, confirming that the features carry genuine discriminative signal. However, the moderate absolute accuracy reflects the inherent difficulty of three-class classification when one class (Mixed) systematically overlaps with the other two.

### 5.2 Test Set Evaluation

#### Random Forest Classification Report:

| Class      | Precision | Recall | F1-Score | Support |
|:----------:|:---------:|:------:|:--------:|:-------:|
| **AI**     | 0.50      | 0.55   | 0.52     | 721     |
| **Human**  | 0.61      | 0.73   | 0.67     | 722     |
| **Mixed**  | 0.06      | 0.04   | 0.05     | 722     |
| **Accuracy**|          |        | **0.44** | 2165    |
| **Macro Avg**| 0.39    | 0.44   | 0.41     | 2165    |
| **Weighted Avg**| 0.39 | 0.44   | 0.41     | 2165    |

**Key Observations:**

1. **Human class performs best** (F1 = 0.67): Human-written text has the most distinctive stylistic signature—higher vocabulary diversity, more varied sentence structures, and naturally occurring idiosyncrasies that the model can detect.

2. **AI class is moderately detectable** (F1 = 0.52): AI-generated text exhibits detectable patterns (uniform sentence length, specific phrase preferences), but some AI passages successfully mimic human style, leading to false negatives.

3. **Mixed class nearly undetectable** (F1 = 0.05): The model essentially fails on the Mixed class—precision of 0.06 and recall of 0.04 indicate near-random performance. This is the most significant finding and is analyzed in depth below.

### 5.3 Confusion Matrix Analysis

<p align="center">
  <img src="images/confusion_matrix.png" alt="Confusion Matrix" width="550"/>
</p>
<p align="center"><em>Figure 1: Confusion matrix for the Random Forest classifier on the test set.</em></p>

The confusion matrix reveals the systematic error structure:

- **AI → Human misclassification**: A significant portion of AI samples are misclassified as Human, suggesting that some AI-generated scientific text has been crafted to closely replicate human stylistic patterns.
- **Human → AI misclassification**: Some human text—particularly highly technical, formulaic academic writing—triggers features (low vocab diversity, rigid structures) that the model associates with AI.
- **Mixed → AI/Human misclassification**: The overwhelming majority of Mixed samples leak into the AI and Human classes. The Mixed class acts as an "absorbing boundary" in feature space, straddling the AI-Human decision boundary.

### 5.4 The "Mixed" Class Challenge

The collapse of Mixed class performance is the central analytical finding of this project. We hypothesize three contributing factors:

**Factor 1 — Feature Space Overlap:**

Mixed text, by definition, contains both human-authored and AI-generated segments within the same passage. At the bag-of-words level (TF-IDF), these features are literally a mixture of the AI and Human feature distributions:

$$\mathbf{x}_{\text{Mixed}} \approx \alpha \cdot \mathbf{x}_{\text{Human}} + (1 - \alpha) \cdot \mathbf{x}_{\text{AI}}, \quad \alpha \in (0, 1)$$

This linear combination places Mixed samples in the interior of the AI-Human convex hull, where the decision boundary is weakest.

**Factor 2 — No Unique Anchor Features:**

Unlike AI text (which has signature phrases) and Human text (which has stylistic idiosyncrasies), Mixed text does not possess exclusive "anchor" features that appear only in Mixed passages and nowhere else. This denies the classifier a reliable signal for positive identification of the Mixed class.

**Factor 3 — Feature Averaging:**

Our document-level features (`avg_sent_len`, `vocab_diversity`) compute averages over the entire passage. For a Mixed document, these averages fall between the typical AI and Human values, further blurring the distinction:

$$\text{vocab-diversity}_{\text{Mixed}} \approx \frac{\text{vocab-diversity}_{\text{Human}} + \text{vocab-diversity}_{\text{AI}}}{2}$$

This averaging effect suppresses the very signal that distinguishes the three classes.

### 5.5 Feature Importance Analysis

<p align="center">
  <img src="images/feature_importance.png" alt="Feature Importance" width="650"/>
</p>
<p align="center"><em>Figure 2: Top 20 features ranked by Random Forest Mean Decrease in Impurity.</em></p>

**Top findings:**

| Rank | Feature | Type | Interpretation |
|:----:|---------|:----:|----------------|
| 1    | `vocab_diversity` | Engineered | **Most discriminative feature overall.** Confirms that lexical richness is the strongest stylistic separator between human and AI text. |
| 7    | `avg_sent_len` | Engineered | Sentence-level structure carries significant signal; validates the hypothesis that AI produces more uniform sentence lengths. |
| 2–6, 8–20 | Various TF-IDF terms | N-gram | Domain-specific terminology and common scientific phrases; some may reflect AI preferences for certain collocations. |

The dominance of `vocab_diversity` is a notable result. It suggests that while AI models produce grammatically correct and topically relevant text, they exhibit a measurable "vocabulary bottleneck"—relying on a narrower set of word choices than human authors. This finding is consistent with the decoding strategies used by LLMs (top-k sampling, nucleus sampling), which tend to favor high-probability tokens.

### 5.6 Error Analysis & Diagnostic Hypotheses

We selected representative misclassified samples from the test set and generated diagnostic hypotheses:

#### Case 1: AI Predicted as Human

> **Text snippet**: *"Titanium dioxide (TiO₂) is a highly desirable material for metallic oxide semiconductors due to its unique properties [1], [2], [3], [4]. Several solvents have been used to prepare TiO₂ nanoparticles,..."*

**Diagnosis**: This AI-generated text exhibits higher-than-typical vocabulary diversity and uses citation markers (a human stylistic convention). The AI model has successfully mimicked the surface-level formatting of human academic writing, deceiving the classifier.

#### Case 2: Human Predicted as AI

> **Text snippet**: *"The ability to form associations among distinct elements (relational binding) is a critical component of higher order cognitive functioning, including episodic remembering, future planning, language p..."*

**Diagnosis**: This human-written text uses highly technical, dense academic prose with uniformly long sentences and formal diction. The rigid structure and low stylistic variation resemble the patterns the model has learned to associate with AI authorship.

#### Case 3: Mixed Predicted as Human/AI

**Diagnosis**: Mixed text that is predominantly human-edited (high α in the mixing coefficient) will inherit enough human stylistic features to be classified as Human. Conversely, Mixed text that retains most of the original AI draft will be classified as AI. The model lacks the resolution to detect the *transition points* between human and AI segments within a single passage—a limitation fundamental to bag-of-words representations.

---

## 6. Conclusion

This project presents, implements, and analyzes a complete machine learning pipeline for the **three-class classification** of scientific text into AI-generated, Human-written, and Mixed categories. The key contributions and findings are:

1. **Pipeline Design**: We demonstrate an end-to-end workflow from data reshaping (wide-to-long format transformation) through feature engineering (TF-IDF + custom linguistic metrics) to model training and evaluation, implemented entirely in Python using scikit-learn and NLTK.

2. **Comparative Analysis**: Random Forest outperforms Multinomial Naive Bayes (44% vs. 39% accuracy), validating the importance of feature interaction modeling over the naive independence assumption for this task.

3. **The Mixed Class Problem**: We identify and rigorously analyze the fundamental challenge posed by mixed-authorship text. The near-zero F1-score (0.05) for the Mixed class demonstrates that frequency-based models operating on document-level features cannot resolve the blended stylistic signature inherent in collaboratively authored text.

4. **Feature Importance Discovery**: Vocabulary diversity emerges as the single most powerful feature for distinguishing AI from Human scientific text, ranking first among over 10,000 features. This finding provides actionable guidance for both detection system design and AI model development.

5. **Diagnostic Framework**: We establish a systematic error analysis methodology that generates testable hypotheses about misclassification causes, moving beyond aggregate metrics to instance-level understanding.

6. **Interpretability**: Unlike black-box deep learning detectors, our approach provides transparent, explainable classification decisions through feature importance scores and decision tree logic—a valuable property for academic integrity applications where practitioners must justify detection decisions.

The moderate overall accuracy achieved by our system reflects both the genuine difficulty of the three-class problem and the inherent limitations of bag-of-words feature representations. The boundary between human and AI text in scientific writing is not a discrete line but a continuous **gradient**, and models that rely solely on surface-level word statistics will struggle with texts near this boundary.

---

## 7. Future Scope

1. **Transformer-Based Detection**: Fine-tuning pretrained models such as **DistilBERT**, **SciBERT**, or **RoBERTa** on this dataset. These models operate on contextual embeddings that capture deep semantic relationships, word order dependencies, and discourse-level coherence patterns that bag-of-words methods fundamentally cannot access. Expected improvement of 15–30% in accuracy.

2. **Explainability Integration**: Implementing **SHAP** (SHapley Additive exPlanations) values to provide per-prediction, per-token attributions—enabling users to see exactly which words and phrases in a specific passage triggered an AI or Human classification.

3. **Segment-Level Detection for Mixed Text**: Instead of classifying entire documents, developing a **sentence-level or paragraph-level** detector that can identify the transition points within a Mixed document where the writing shifts from human to AI (or vice versa). This requires sub-document labeling and sequential modeling (e.g., BiLSTM or sliding-window Transformers).

4. **Data Augmentation**: Expanding the Mixed class with more diverse mixing strategies (e.g., varying the proportion of AI content, using different LLMs for generation, applying different levels of human editing) to improve the model's ability to distinguish Mixed text from pure classes.

5. **Cross-Domain and Cross-Model Generalization**: Testing whether a model trained on ChatGPT-generated text in one academic domain generalizes to text generated by a different LLM (e.g., Claude, Gemini) in a different domain. This is critical for real-world deployment where the generating model and subject area are unknown.

6. **Adversarial Robustness**: Evaluating the pipeline's resilience to adversarial manipulations such as paraphrasing, synonym substitution, sentence reordering, and back-translation—common evasion strategies that users may employ to bypass detection.

---

## 8. References

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems, 30.

2. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. Proceedings of NAACL-HLT 2019.

3. Mitchell, E., Lee, Y., Khazatsky, A., Manning, C. D., & Finn, C. (2023). *DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature*. Proceedings of ICML 2023.

4. Kirchenbauer, J., Geiping, J., Wen, Y., Katz, J., Miers, I., & Goldstein, T. (2023). *A Watermark for Large Language Models*. Proceedings of ICML 2023.

5. Uchendu, A., Le, T., Shu, K., & Lee, D. (2020). *Authorship Attribution for Neural Text Generation*. Proceedings of EMNLP 2020.

6. Guo, B., Zhang, X., Wang, Z., Jiang, M., Nie, J., Ding, Y., Yue, J., & Wu, Y. (2023). *How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection*. arXiv preprint arXiv:2301.07597.

7. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. arXiv preprint arXiv:1910.01108.

8. Beltagy, I., Lo, K., & Cohan, A. (2019). *SciBERT: A Pretrained Language Model for Scientific Text*. Proceedings of EMNLP 2019.

9. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.

10. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. (Chapters 6, 13 on TF-IDF and Naive Bayes.)

11. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.

12. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media. (NLTK reference.)

13. Mosteller, F., & Wallace, D. L. (1964). *Inference and Disputed Authorship: The Federalist*. Addison-Wesley.

14. Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. Advances in Neural Information Processing Systems, 30. (SHAP reference.)

15. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*. Proceedings of KDD 2016. (LIME reference.)

---

*This report was prepared as part of the AIGTxt Classification Pipeline project. The complete, executable analysis is available in the accompanying Jupyter Notebook: `notebooks/AIGTxt_Classification_Pipeline.ipynb`.*
