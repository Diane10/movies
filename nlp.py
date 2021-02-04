import pandas as pd
import numpy as np
from textblob import Word
from textblob import TextBlob
import nltk
import nltk
nltk.download('punkt')
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import 	WordNetLemmatizer

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
nltk.download('averaged_perceptron_tagger') 
from nltk.corpus import wordnet 
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from nltk.stem.porter import PorterStemmer
#Nlp
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

if st.sidebar.checkbox("Market Basket Anlysis"):
    dataset = pd.read_csv('https://raw.githubusercontent.com/Diane10/movies/main/GroceryStoreDataSet.csv')
    dataset = list(dataset["Transaction"].apply(lambda x:x.split(',')))
    st.write(dataset)
    te = TransactionEncoder()
    te_data = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_data,columns=te.columns_)
    df = df.replace(True,1) 
    df = df.astype(int)
    st.write(df)
    
    freq_items = apriori(df,min_support=0.10,use_colnames=True) # Support Values
    
    st.write(freq_items.sort_values(by = "support" , ascending = False))
    
    rules = association_rules(freq_items, metric="lift", min_threshold=1)
    st.write(rules)
    
    df1 = association_rules(freq_items, metric = "confidence", min_threshold = 0.02)
    
    st.write(df1)
    
    st.write(df1[(df1.confidence > 0.8) & (df1.lift > 1)].sort_values(by="lift", ascending=False))    
    
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
            
    if st.checkbox("Removing stopword"): 

        if st.button("Analyse",key='13'):
            stop_words=set(stopwords.words("english"))
            word_tokens=word_tokenize(message)
            filtered_sentence = [w for w in word_tokens if not w in stop_words]  
            filtered_sentence = []  
            for w in word_tokens:  
                if w not in stop_words:  
                    filtered_sentence.append(w)  
              
            st.write(word_tokens)  
            st.write(filtered_sentence)
            
            
    if st.checkbox("lemmatizer"):
        selection = st.selectbox("Select type:", ("Lemmatizer", "PorterStemmer"))
        if st.button("Analyse",key="4"):
            if selection == "Lemmatizer":
                wordnet_lemmatizer=WordNetLemmatizer()
                tokenization=nltk.word_tokenize(message)
                for w in tokenization:
                    st.write("Lemma for {} is {}".format(w,wordnet_lemmatizer.lemmatize(w))) 
                    wordnet_lemmatizer=WordNetLemmatizer()
	                     
	  
            elif selection == "PorterStemmer":
                porter_stemmer=PorterStemmer()
                tokenization=nltk.word_tokenize(message)
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
    if st.checkbox("language detector"):
        if st.button("Analyse",key="15"):
            blob = TextBlob(message)
            st.write(blob.detect_language())

    if st.checkbox("Translate sentences"):
        if st.button("Analyse"):
            selection = st.selectbox("Select language:", ("French", "Spanish","Chinese"))
            if selection == "French":
                blob = TextBlob(message)
                translated=blob.translate(to="fr")
                st.write(translated)
                
            if selection == "Spanish":
                blob = TextBlob(message)
                translated=blob.translate(to='es')
                st.write(translated)
                
            if selection == "Chinese":
                blob = TextBlob(message)
                translated=blob.translate(to="zh")
                st.write(translated)    
