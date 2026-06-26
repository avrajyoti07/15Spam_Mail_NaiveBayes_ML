# 📧 Spam Detection — Naive Bayes Classifier

A machine learning project that classifies SMS messages as **spam or ham (not spam)** using a **Multinomial Naive Bayes** classifier built with Python and scikit-learn.

---

## 📌 Project Overview

This project applies the Naive Bayes algorithm to a real-world SMS spam dataset. It covers the complete NLP + ML pipeline — from text vectorization and label encoding to model training, prediction, and pipeline construction using scikit-learn.

The model is tested on real-world example messages to demonstrate practical spam detection.

---

## 📂 Files

| File | Description |
|------|-------------|
| `14Naive_Bayes_Classifier.ipynb` | Main Jupyter Notebook with the complete ML pipeline |
| `spam.csv` | SMS spam dataset (5,572 messages, 2 columns) |

---

## 📊 Dataset

The dataset contains **5,572 SMS messages** labeled as either `spam` or `ham`.

| Column | Description |
|--------|-------------|
| `Category` | Label — `spam` or `ham` |
| `Message` | Raw SMS text content |

**Class distribution:**

| Label | Count | Percentage |
|-------|-------|------------|
| Ham (not spam) | 4,825 | 86.59% |
| Spam | 747 | 13.41% |

**Sample spam messages:**
- *"Free entry in 2 a wkly comp to win FA Cup final tkts..."*
- *"WINNER!! As a valued network customer you have been selected to receive a £900 prize reward..."*

**Sample ham messages:**
- *"Go until jurong point, crazy.. Available only in bugis n great world la e buffet..."*
- *"Ok lar... Joking wif u oni..."*

---

## 🔧 Steps Performed

### 1. Data Loading & Exploration
Loaded the SMS dataset using `pandas` and explored class distribution with `groupby`.

### 2. Label Encoding
Converted the text `Category` column into a binary numeric target:
```python
df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
```

### 3. Train-Test Split
Split the data using a **75/25 ratio**:
- Training set: ~4,179 messages
- Testing set: ~1,393 messages

### 4. Text Vectorization — CountVectorizer
Transformed raw SMS text into a numeric **Bag of Words (BoW)** matrix using `CountVectorizer`:
- `fit_transform()` on training data — learns the vocabulary and encodes
- `transform()` on test data — encodes using the learned vocabulary (no data leakage)

### 5. Model Training — MultinomialNB
Trained a `MultinomialNB` model — the go-to Naive Bayes variant for **word count / frequency features**.

### 6. Real-World Prediction
Tested the model on two custom messages:
```python
emails = [
    'Hey Bro, can we get together to watch world cup matches today?',     # → ham
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'  # → spam
]
```

### 7. Model Evaluation
Scored the model on the held-out test set using `model.score()`.

### 8. Pipeline (Clean Method)
Refactored the full workflow into a compact scikit-learn `Pipeline`:
```python
from sklearn.pipeline import Pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
clf.fit(X_train, Y_train)
clf.score(X_test, Y_test)
clf.predict(emails)
```
The pipeline chains vectorization and classification into a single reusable object — no need to manually transform before each predict call.

---

## 🧠 Key Concepts

**Why Multinomial Naive Bayes?**
Multinomial NB is specifically designed for discrete count data — perfect for word frequencies in text. It calculates the probability of a message being spam based on the likelihood of each word appearing in spam vs. ham messages.

**Why CountVectorizer?**
CountVectorizer converts raw text into a matrix of token counts (Bag of Words). Each column represents a unique word in the vocabulary, and each row represents a message with word frequencies.

**Why `fit_transform()` on train but only `transform()` on test?**
`fit_transform()` builds the vocabulary from training data AND encodes it. Using it on test data would cause **data leakage** — the model would "see" test vocabulary during training. `transform()` only encodes using the already-learned vocabulary.

**Why Pipeline?**
The `Pipeline` bundles preprocessing and modeling steps, ensures correct `fit`/`transform` behaviour, prevents data leakage, and makes the code cleaner and production-ready.

---

## 🛠️ Tech Stack

- **Python 3**
- **pandas** — data loading and manipulation
- **scikit-learn** — `CountVectorizer`, `MultinomialNB`, `train_test_split`, `Pipeline`
- **Jupyter Notebook** — interactive development

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas scikit-learn jupyter
```

### Run the Notebook
```bash
git clone https://github.com/your-username/spam-detection-naive-bayes.git
cd spam-detection-naive-bayes
jupyter notebook 14Naive_Bayes_Classifier.ipynb
```

> Make sure `spam.csv` is in the same directory as the notebook.

---

## 📈 Results

The model achieves high accuracy on the held-out test set. Run the notebook to see the exact score. The Pipeline version produces the same result with cleaner, more production-ready code.

---

## 🔮 Possible Improvements

- Use **TF-IDF Vectorizer** instead of CountVectorizer for better weighting of rare/common words
- Add **confusion matrix** and **classification report** for precision/recall analysis
- Handle class imbalance (86% ham vs 13% spam) with oversampling or class weights
- Try **ComplementNB** which is known to perform better on imbalanced text datasets

---

## 🤝 Contributing

Pull requests are welcome! Feel free to fork the repo and experiment with different vectorizers, classifiers, or evaluation metrics.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
