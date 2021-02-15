# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 16:26:01 2021

@author: HP
"""

import streamlit as st
#import streamlit.components.v1 as stc

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel

def load_data(data):
    df= pd.read_csv(data)
    return df

def vectorize_text_to_cosine_max(data):
    count_vec= CountVectorizer()
    cv_mat= count_vec.fit_transform(data)
    cosine_sim=cosine_similarity(cv_mat)
    return cosine_sim
def get_recommendation(title,cosine_sim_mat,df,num_of_rec=5):
    course_indices=pd.Series(df.index,index=df['title']).drop_duplicates()
    idx=course_indices[title]
    sim_scores=list(enumerate(cosine_sim_mat[idx]))
    sim_scores= sorted(sim_score,key=lambda x:x[1],reverse=True)
    return sim_scores[1:]

st.title("Book Recommendation APP")
menu= ["Home","Recommender","About"]
choice=st.sidebar.selectbox("Menu",menu)
df=load_data("https://raw.githubusercontent.com/sahilpocker/Book-Recommender-System/master/Dataset/books.csv")
if choice=='Home':
    st.subheader("Home")
    st.dataframe(df.head(10))
    
elif choice =="Recommender":
    st.subheader("Recommend Book")
    cosine_sim_mat=vectorize_text_to_cosine_max(df['title'])
    search_term=st.text_input("Search Boook")
    num_of_rec= st.sidebar.number_input("Number",4,30,7)
    if st.button("Recommend"):
        if search_term is not None:
            result= get_recommendation(search_term,cosine_sim_mat,df,num_of_rec)
            st.write(result)
else:
    st.subheader("About US")
    
    
    
    
    
   