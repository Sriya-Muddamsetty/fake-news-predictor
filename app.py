import streamlit as st
import pickle
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import sklearn
import re
nltk.download('stopwords')

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

port_stem = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ' ,content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

prediction = ' '
st.title("Fake News Predictor")
input = st.text_input("Enter the News below")
if st.button('Classify'):
    transformed_news = stemming(input)
    vector_input = tfidf.transform([transformed_news])
    result = model.predict(vector_input)[0]
    if result == 1:
        prediction = 'Fake News'
    else:
        prediction = 'Real News'
    st.success(prediction)







