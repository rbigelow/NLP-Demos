import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Load a small SMS spam dataset
# ---------------------------------------------------------
# Dataset source: SMS Spam Collection (public domain)
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep="\t", header=None, names=["label", "text"])

print("Dataset loaded:")
print(df.head())

# ---------------------------------------------------------
# 2. Train/test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 3. Vectorize text using Bag‑of‑Words
# ---------------------------------------------------------
vectorizer = CountVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------------------------------------------------
# 4. Train Naive Bayes classifier
# ---------------------------------------------------------
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# ---------------------------------------------------------
# 5. Evaluate model
# ---------------------------------------------------------
y_pred = clf.predict(X_test_vec)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------
# 6. Confusion Matrix
# ---------------------------------------------------------
cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["ham", "spam"],
            yticklabels=["ham", "spam"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ---------------------------------------------------------
# 7. Show misclassified examples
# ---------------------------------------------------------
print("\n=== Misclassified Messages ===")
misclassified = X_test[(y_test != y_pred)]

for text, true_label, pred_label in zip(
    misclassified, y_test[y_test != y_pred], y_pred[y_test != y_pred]
):
    print(f"\nMessage: {text}")
    print(f"Actual: {true_label} | Predicted: {pred_label}")