import streamlit as st
import pickle

import time
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
print(1)
ps = PorterStemmer()

print(2)
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)
print(3)
tfidf = pickle.load(open('vectorizer.pkl','rb'))
print(4)
model = pickle.load(open('model.pkl','rb'))
print(5)

st.title("Email/SMS Spam Classifier")
print(6)

input_sms = st.text_area("Enter the message")
print(7)
if st.button('Predict'):

    # 1. preprocess
    print(8)
    transformed_sms = transform_text(input_sms)
    print(transformed_sms)
    print(9)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    print('vector_input')
    print(vector_input)
    # vector_input=vector_input.toarray()
    # 3. predict
    print(vector_input)
    print('result 123')
    result = model.predict(vector_input)[0]
    

    print(result)
    print('result - ')
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
