import pandas as pd
import numpy as np
from textblob import Word
from textblob import TextBlob
import nltk
import nltk
nltk.download('punkt')
import streamlit as st
from nltk.stem import 	WordNetLemmatizer


nltk.download("wordnet")
nltk.download("brown")
nltk.download('averaged_perceptron_tagger') 
from nltk.corpus import wordnet 
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from nltk.stem.porter import PorterStemmer

st.title("Natural Language Processing with Streamlit")
def sumy_summarize(docx):
    parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result
    
def pos_tagger(nltk_tag): 
    if nltk_tag.startswith('J'): 
        return wordnet.ADJ 
    elif nltk_tag.startswith('V'): 
        return wordnet.VERB 
    elif nltk_tag.startswith('N'): 
        return wordnet.NOUN 
    elif nltk_tag.startswith('R'): 
        return wordnet.ADV 
    else:           
        return None    
    
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
            blob = TextBlob(message)
            st.write (blob.words)
    if st.checkbox("show sentence"):
        if st.button("Analyse"):
            blob = TextBlob(message)
            st.write(blob.sentences)
    if st.checkbox("Tokenize sentence"): 
        if st.button("Analyse"):
            list2 = nltk.word_tokenize(message) 
            st.write(list2) 
    if st.checkbox("POS tag "): 
        if st.button("Analyse"):
            pos_tagged = nltk.pos_tag(nltk.word_tokenize(message))   
            st.write(pos_tagged) 
    if st.checkbox("lemmatizer"):
        selection = st.selectbox("Select Analysis:", ("Lemmatizer", "PorterStemmer"))
        if st.button("Analyse",key="4"):
            if selection == "Lemmatize":
                wordnet_lemmatizer = WordNetLemmatizer()
	            tokenization = nltk.word_tokenize(message)
                for w in tokenization:
                    st.write("Lemma for {} is {}".format(w, wordnet_lemmatizer.lemmatize(w))) 
	  
             elif selection == "PorterStemmer":
                porter_stemmer  = PorterStemmer()
	            tokenization = nltk.word_tokenize(message)
                for w in tokenization:
                    st.write("Stemming for {} is {}".format(w,porter_stemmer.stem(w)))   
                
              
    if st.checkbox("show text summarization"):
        if st.button("Analyse",key="5"):
            st.subheader("summarize your text")
            summary_result= sumy_summarize(message)
            st.success(summary_result)
        
    if st.checkbox("splelling checker"):
        if st.button("Analyse",key="6"):
            blob = TextBlob(message)
            st.write(blob.correct())
    if st.checkbox("Translate to German from English"):
        if st.button("Analyse"):
            blob = TextBlob(message)
            translated=blob.translate(to="de")
            st.write(translated)



