import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("Training started...")

# Load dataset
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true])

data = data.sample(frac=1).reset_index(drop=True)

data["content"] = data["title"]

X = data["content"]
y = data["label"]

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7, ngram_range=(1,2))

X_vector = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vector, y, test_size=0.2, random_state=42
)

model = LogisticRegression(class_weight="balanced", max_iter=2000)
model.fit(X_train, y_train)
# Predict
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

# Save model
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("Model saved successfully")
