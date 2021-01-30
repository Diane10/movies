import pandas as pd
import numpy as np
from textblob import Word
from textblob import TextBlob
import nltk
import nltk
nltk.download('punkt')
import streamlit as st

nltk.download("wordnet")
nltk.download("brown")
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

st.title("Natural Language Processing with Streamlit")
def sumy_summarize(docx):
    parser= PlaintextParser.from_string(docx,Tokenizer("english"))
    lex_summarize=LexRankSummarizer(parser.document,3)
    summary_list=[str(sentence) for sentence in summary]
    result=' '.join(summary_list)
    return result
message =st.text_area("Enter text")
blob = TextBlob(message)
if st.sidebar.checkbox("NLP"):
    if st.checkbox('Noun phrases'):
        if st.button("Analyse",key="1"):
#              text1 =st.text_area("Enter text")
             blob = TextBlob(message)
             st.write(blob.noun_phrases)
    if st.checkbox("show sentiment analysis"):
#         st.subheader("analyse your text")
#         message=st.text_area("Enter your text")  
        if st.button("Analyse",key="2"):
            blob = TextBlob(message)
            result_sentiment= blob.sentiment
            st.success(result_sentiment)
            polarity = blob.polarity
            subjectivity = blob.subjectivity
            st.write(polarity, subjectivity)
    if st.checkbox("show words"): 
        if st.button("Analyse",key="3"):
            st.write (message.words)
    if st.checkbox("show sentence"): 
        if st.button("Analyse"):
            st.write(message.sentences)
    if st.checkbox("lemmatizer"):
        if st.button("Analyse",key="4"):
            st.write(Word(message).lemmatize("v"))
            st.write(message.lemmatize("v"))
    if st.checkbox("show text summarization"):
        if st.button("Analyse",key="5"):
            st.subheader("summarize your text")
            message = st.text_area("Enter text ","Type here...")
            st.text("using summy summarizer")
            summary_result= sumy_summarize(message)
            st.success(summary_result)
        
    if st.checkbox("splelling checker"):
        if st.button("Analyse",key="6"):
            blob = TextBlob(message)
            st.write(blob.correct())
    if st.checkbox("Translate to German from English"):
        if st.button("Analyse"):
            blob = TextBlob(message)
            blob.translate(to="de")



