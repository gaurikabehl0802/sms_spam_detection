# SMS Spam Detection using Natural Language Processing (NLP)

An end-to-end Machine Learning and NLP mini-project that classifies incoming SMS messages as **Ham** (legitimate) or **Spam** using TF-IDF feature extraction and a Multinomial Naive Bayes classifier.

---

## 📌 Project Overview

Spam messages can lead to security vulnerabilities, phishing attacks, and degraded user experience. This project automates the detection of unsolicited, fraudulent, or promotional text messages using text processing and statistical learning techniques.

- **Objective:** Build an automated binary classification pipeline for SMS messages.
- **Model:** Multinomial Naive Bayes (`MultinomialNB`).
- **Feature Extraction:** Term Frequency-Inverse Document Frequency (`TfidfVectorizer`).
- **Performance:** Achieves ~97–98% overall accuracy.

---

## 🗂️ Dataset

- **Name:** SMS Spam Collection Dataset
- **Total Records:** 5,572 messages (4,825 Ham, 747 Spam)
- **Labels:** 
  - `ham`: Legitimate personal messages (0)
  - `spam`: Unsolicited/promotional/phishing messages (1)
- **Source:** Direct TSV fetch via GitHub/UCI Machine Learning Repository.

---

## ⚙️ Tech Stack & Dependencies

- **Language:** Python 3.8+
- **Environment:** Google Colab / Jupyter Notebook
- **Libraries:**
  - `nltk` (Stopwords, PorterStemmer)
  - `pandas` (Data loading and manipulation)
  - `scikit-learn` (Vectorization, Naive Bayes, Evaluation metrics)
  - `matplotlib` & `seaborn` (Confusion matrix visualization)

---

## 🚀 NLP & ML Workflow

1. **Text Preprocessing:**
   - Case folding to lower-case.
   - Punctuation and numeric character removal via regex.
   - Whitespace tokenization.
   - Stopword removal using NLTK English stopwords list.
   - Word root extraction using the **Porter Stemmer**.
2. **Feature Extraction:**
   - Vectorization with `TfidfVectorizer(max_features=3000)`.
3. **Data Splitting:**
   - Stratified 80/20 train-test split (`random_state=42`).
4. **Model Training:**
   - `MultinomialNB` fitted on sparse TF-IDF feature matrices.
5. **Evaluation:**
   - Accuracy score, precision, recall, F1-score, and Confusion Matrix.

---

## 📊 Results & Metrics

| Metric | Ham (Class 0) | Spam (Class 1) | Overall |
| :--- | :--- | :--- | :--- |
| **Precision** | ~0.98 | ~0.96 | — |
| **Recall** | ~0.99 | ~0.84 | — |
| **F1-Score** | ~0.99 | ~0.90 | — |
| **Accuracy** | — | — | **~97.5% - 98.2%** |

---

## 💻 Quick Start (Google Colab)

1. Open a new notebook in [Google Colab](https://colab.research.google.com/).
2. Paste the implementation code into a code cell.
3. Run the cell (`Ctrl + Enter` or `Shift + Enter`).
4. Colab will download the required NLTK resources, fetch the dataset, train the model, output the classification report, and render the confusion matrix.

---

## 🧪 Inference Example

```python
# Function to classify custom SMS text
predict_sms("Congratulations! You won a $1,000 Walmart gift card. Click here.")
# Output: SPAM (98.4% confidence)

predict_sms("Hey, are we still meeting for lunch at 1 PM?")
# Output: HAM (99.1% confidence)
