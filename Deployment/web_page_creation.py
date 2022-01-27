# -*- coding: utf-8 -*-
import numpy as np
import pickle
import nltk
import re
import streamlit as st
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

vec_file=pickle.load(open('C:/Users/vinis/Sentiment-Analysis/Deployment/vectorizer.sav','rb'))
model_2=pickle.load(open('C:/Users/vinis/Sentiment-Analysis/Deployment/Example2.sav','rb'))

def predict_sentiment(sample_review):
    sample_review = re.sub(pattern='[^a-zA-Z]',repl=' ', string = sample_review)
    sample_review = sample_review.lower()
    sample_review_words = sample_review.split()
    sample_review_words = [word for word in sample_review_words if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    final_review = [ps.stem(word) for word in sample_review_words]
    final_review = ' '.join(final_review)
    temp = vec_file.transform([final_review]).toarray()
    return model_2.predict(temp)

def model(review):
    x=predict_sentiment(review)
    x1=np.array([1])
    if np.array_equal(x,x1):
        return "This is a POSITIVE review."
    else:
        return "This is a NEGATIVE review!"
        
def main():
    st.title("Sentiment Analysis for Product Samsung Galaxy M21")
    review=st.text_input("Please give review")
    
    results=[]
    if st.button("Click Here"):
        results=model(review)
      
    st.success(results)
  

    
if __name__=='__main__':
    main()
    