#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
import copy
import warnings
warnings.filterwarnings("ignore")
#plotly.offline.init_notebook_mode (connected = True)


# In[2]:


data=pd.read_csv('genres_v2.csv')


# In[3]:





# In[4]:





# In[5]:




# In[6]:


data.isnull()
# In[13]:


data.drop('Unnamed: 0',axis=1,inplace=True)


# ##  Let's Make A Recommendation System....
# ###   Over here we are gonna use different methods to find the closest songs to the one that user have liked :)
# 
# ##  Remove all the rows with no song name

# In[15]:


data=data.dropna(subset=['song_name'])


# ##  Need to Preprocess the Data
# ###   Question arises why do we need to standardize the data ??
# Answer : You see that the data has columns like duration ms whose distance difference can be high causing lot of change in the answer we want every field to contribute the same to the distance (euclidean) hence we have to standardize the data .

# In[16]:


# Creating a new dataframe with required features
df=data[data.columns[:11]]
df['genre']=data['genre']
df['time_signature']=data['time_signature']
df['duration_ms']=data['duration_ms']
df['song_name']=data['song_name']


# In[17]:


x=df[df.drop(columns=['song_name','genre']).columns].values
scaler = StandardScaler().fit(x)
X_scaled = scaler.transform(x)
df[df.drop(columns=['song_name','genre']).columns]=X_scaled


# ##  Recommendation System Using Euclidean Distance

# In[18]:


# This is a function to find the closest song name from the list
def find_word(word,words):
    t=[]
    count=0
    if word[-1]==' ':
        word=word[:-1]
    for i in words:
        if word.lower() in i.lower():
            t.append([len(word)/len(i),count])
        else:
            t.append([0,count])
        count+=1
    t.sort(reverse=True)
    return words[t[0][1]]


# In[19]:


# Making a weight matrix using euclidean distance
def make_matrix(data,song,number):
    df=pd.DataFrame()
    data.drop_duplicates(inplace=True)
    songs=data['song_name'].values
#    best = difflib.get_close_matches(song,songs,1)[0]
    best=find_word(song,songs)
    print('The song closest to your search is :',best)
    genre=data[data['song_name']==best]['genre'].values[0]
    df=data[data['genre']==genre]
    x=df[df['song_name']==best].drop(columns=['genre','song_name']).values
    if len(x)>1:
        x=x[1]
    song_names=df['song_name'].values
    df.drop(columns=['genre','song_name'],inplace=True)
    df=df.fillna(df.mean())
    p=[]
    count=0
    for i in df.values:
        p.append([distance.euclidean(x,i),count])
        count+=1
    p.sort()
    ans=[]
    for i in range(1,number+1):
        ans.append(song_names[p[i][1]])
    return ans
    
    


# In[ ]:


st.title('Song Recommendation System')
st.sidebar.header('User Input Parameter')
a=st.sidebar.text_input("Insert Song Name")
b=st.sidebar.slider("Pick a Number to Recommend Song",50,100)
st.sidebar.header('User Input Parameter')
data={
        'song_name':a,
        'number':b
    }
st.write(pd.DataFrame(data,index=[0]))
if st.sidebar.button("Submit:)"):
    c = make_matrix(df,a,b)
    st.subheader('Rcommended Songs')
    for i in range(len(c)):
        st.write("-->",c[i])

    # In[20]:


#a=input('Please enter The name of the song :')
#b=int(input('Please enter the number of recommendations you want: ')



# In[ ]:




