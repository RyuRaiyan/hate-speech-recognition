import pandas as pd
import re
import string
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


nltk.download("stopwords")
nltk.download("wordnet")


df = pd.read_csv("dataset/hate_speech.csv")
df = df.dropna()

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\\S+", "", text)
    text = re.sub(r"@\\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]
    return " ".join(words)

df["clean_text"] = df["text"].apply(clean_text)

text = " ".join(df["clean_text"])

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(text)

plt.figure(figsize=(12,6))
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Word Cloud")
plt.show()


vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("\\nAccuracy:", accuracy_score(y_test, y_pred))
print("\\nClassification Report\\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()


def predict(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(vector).max() * 100
        print(f"Input      : {text}")
        print(f"Prediction : {prediction}")
        print(f"Confidence : {confidence:.2f}%")
    else:
        print(f"Input      : {text}")
        print(f"Prediction : {prediction}")


samples = [
    "I love everyone.",
    "You are an idiot.",
    "Go kill yourself.",
    "Have a wonderful day.",
    "You are amazing."
]

for sample in samples:
    print("-" * 50)
    predict(sample)
