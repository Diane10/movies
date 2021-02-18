# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 16:26:01 2021

@author: HP
"""

import streamlit as st
import streamlit.components.v1 as stc

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
    sim_scores= sorted(sim_scores,key=lambda x:x[1],reverse=True)
    selected_course_indices=[i[0] for i in sim_scores[1:]]
    selected_course_score=[i[0] for i in sim_scores[1:]]
    result_df= df.loc[selected_course_indices]   
    result_df['similarity score']=selected_course_score
    final_recommeded= result_df[['title','authors','similarity score','image_url']]
    return final_recommeded.head(num_of_rec)

@st.cache
def search_term_if_not_found(term,df):
    result_df= df[df['title'].str.contains(term)]
    return result_df

#css style

RESULT_TEMP=""" 

<div class="card" style="box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
  transition: 0.3s;">
  <img src={} alt="Avatar" style="width:100%">
  <div class="container" style="padding: 2px 16px;">
    <h4><b>{}</b></h4>
    <p>{}</p>
  </div>
</div>

"""
st.title("Book Recommendation APP")
menu= ["Home","Recommender","About"]
choice=st.sidebar.selectbox("Menu",menu)
df=load_data("https://raw.githubusercontent.com/sahilpocker/Book-Recommender-System/master/Dataset/books.csv")
df=df[:200]
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
            try:
                result= get_recommendation(search_term,cosine_sim_mat,df,num_of_rec)
                for row in result.iterrows():
                    rec_title= row[1][0]
                    rec_author= row[1][1]
                    rec_score= row[1][2]
                    rec_image= row[1][3]
#                     c1,c2,c3 = st.beta_columns([1,2,1])
#                     with c1:
#                         with st.beta_expander("Title"):
#                             st.success(rec_title)
                            
#                     with c2:
#                         with st.beta_expander("image"):
#                             st.image(rec_image,use_column_with=True) 
                            
                            
#                     with c1:
#                         with st.beta_expander("author"):
#                             st.success(rec_author)        

                    st.write("Title",rec_title,"author",rec_author)
                    stc.html(RESULT_TEMP.format(rec_image,rec_title,rec_author)
                 
    
                    
                    
            except: 
                st.warning('Book Not Found')
                st.info("suggested Option include")
                result_df=search_term_if_not_found(search_term,df)
                st.dataframe(result_df)
#                 result= get_recommendation(search_term,cosine_sim_mat,result_df,num_of_rec)
#                 for row in result.iterrows():
#                     rec_title= row[1][0]
#                     rec_author= row[1][1]
#                     rec_score= row[1][2]
#                     rec_image= row[1][3]
#                 st.write("Title",rec_title)
                
                    
            
            
                
else:
    st.subheader("About US")
    
    
    
    
    
   
