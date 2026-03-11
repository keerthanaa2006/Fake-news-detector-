import streamlit as st
import pickle

# load model
model = pickle.load(open("model/model.pkl","rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl","rb"))

st.title("Fake News Detector")

news = st.text_area("Paste news headline or article")

if st.button("Check News"):

    vector = vectorizer.transform([news])

    prediction = model.predict(vector)

    if prediction[0] == 1:
        st.success("Real News")
    else:
        st.error("Fake News")