

import pandas as pd
import numpy as np
from textblob import Word
from textblob import TextBlob
import nltk
import nltk
nltk.download('punkt')
import streamlit as st
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


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

if st.sidebar.checkbox("Market Basket Anlysis"):
    dataset = pd.read_csv('GroceryStoreDataSet.csv')
    dataset = list(dataset["Transaction"].apply(lambda x:x.split(',')))
    st.write(dataset.head())
    te = TransactionEncoder()
    te_data = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_data,columns=te.columns_)
    df = df.replace(True,1) 
    df = df.astype(int)
    st.write(df)
    
    freq_items = apriori(df,min_support=0.10,use_colnames=True) # Support Values
    
    st.write(freq_items.sort_values(by = "support" , ascending = False))
    
    rules = association_rules(freq_items, metric="lift", min_threshold=1)
    st.write(rules.head())
    
    df1 = association_rules(freq_items, metric = "confidence", min_threshold = 0.02)
    
    st.write(df1.head())
    
    st.write(df1[(df1.confidence > 0.8) & (df1.lift > 1)].sort_values(by="lift", ascending=False))

if st.sidebar.checkbox("NLP"):
    st.text_area("Enter text")
    blob = TextBlob(text1)
    if st.checkbox('Noun phrases'):
        st.write(blob.noun_phrases)
    if st.checkbox("show sentiment analysis"):
        st.subheader("analyse your text")
        message=st.text_area("Enter your text")  
        if st.button("Analyse"):
            blob = TextBlob(text1)
            result_sentiment= blob.sentiment
            st.success(result_sentiment)
            polarity = blob.polarity
            subjectivity = blob.subjectivity
            st.write(polarity, subjectivity)
    if st.checkbox("show words"):         
        st.write (blob.words)
    if st.checkbox("show sentence"):    
        blob.sentences
    if st.checkbox("lemmatizer"):
        word1= st.text_area('Enter number:')
        st.write(Word(word1).lemmatize("v"))
        st.write(word1.lemmatize("v"))
    if st.checkbox("show text summarization"):
        st.subheader("summarize your text")
        message = st.text_area("Enter text ","Type here...")
        st.text("using summy summarizer")
        summary_result= sumy_summarize(message)
        st.success(summary_result)
        
    if st.checkbox("splelling checker"):
        word1= st.text_area('Enter number:')
        blob = TextBlob(word1)
        st.write(blob.correct())
    if st.checkbox("Translate to German from English"):
        word1= st.text_area('Enter number:')
        blob = TextBlob(word1)
        blob.translate(to="de")



