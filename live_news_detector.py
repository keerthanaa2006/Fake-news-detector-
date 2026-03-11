import requests
import pickle

# load model
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

API_KEY = "f8b436b068df40a6bc1959ed114812ff"

url = f"https://newsapi.org/v2/top-headlines?language=en&pageSize=10&apiKey={API_KEY}"

response = requests.get(url)
data = response.json()

if data["status"] != "ok":
    print("Error fetching news:", data)
else:
    for article in data["articles"]:

        headline = article.get("title")

        if not headline:
            continue

        vector = vectorizer.transform([headline])
        prediction = model.predict(vector)

        if prediction[0] == 1:
            result = "Real News"
        else:
            result = "Fake News"

        print("Headline:", headline)
        print("Prediction:", result)
        print("-----------------------------")