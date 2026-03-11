import pickle

model = pickle.load(open("model/model.pkl","rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl","rb"))

while True:

    text = input("Enter news text: ")

    vector = vectorizer.transform([text])

    prediction = model.predict(vector)

    if prediction[0] == 1:
        print("Real News")
    else:
        print("Fake News")