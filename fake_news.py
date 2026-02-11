# ==========================================
# Fake News Detection System (Improved Version)
# ==========================================

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# -----------------------
# 1️⃣ Load Dataset
# -----------------------
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels: 0 = Fake, 1 = Real
fake['label'] = 0
true['label'] = 1

# Combine and shuffle
data = pd.concat([fake, true]).sample(frac=1).reset_index(drop=True)

# Keep only text and label, fill missing
data = data[['text', 'label']].fillna('')

# -----------------------
# 2️⃣ Preprocessing
# -----------------------
def clean_text(text):
    # Keep letters and numbers, remove other chars
    text = re.sub('[^a-zA-Z0-9 ]', ' ', text)
    # Lowercase
    text = text.lower()
    # Lemmatize each word
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

data['text'] = data['text'].apply(clean_text)

# -----------------------
# 3️⃣ Features & Labels
# -----------------------
X = data['text']
y = data['label']

# TF-IDF vectorizer with unigrams + bigrams
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, ngram_range=(1,2))
X = vectorizer.fit_transform(X)

# -----------------------
# 4️⃣ Train-Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------
# 5️⃣ Train Model
# -----------------------
model = LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced')
model.fit(X_train, y_train)

# -----------------------
# 6️⃣ Evaluate Model
# -----------------------
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# -----------------------
# 7️⃣ Manual Testing
# -----------------------
print("\n=== Fake News Detection ===")
while True:
    news = input("Enter news (or type 'exit' to quit): ")
    if news.lower() == "exit":
        break

    # Preprocess and vectorize
    news_clean = clean_text(news)
    news_vector = vectorizer.transform([news_clean])

    # Predict
    prediction = model.predict(news_vector)
    confidence = model.predict_proba(news_vector)[0]

    # Display result
    if prediction[0] == 1:
        print(f"Result: Real News (Confidence: {confidence[1]*100:.2f}%)\n")
    else:
        print(f"Result: Fake News (Confidence: {confidence[0]*100:.2f}%)\n")
