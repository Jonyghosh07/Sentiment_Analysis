#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re

import pandas as pd


# In[2]:


#Extract Date Time
def date_time(s):
  pattern = "^([0-9]+)(\/)([0-9]+)(\/)([0-9]+), ([0-9]+):([0-9]+)[ ]?(AM|PM|am|pm)? - "
  result = re.match(pattern, s)
  if result:
    return True
  else:
    return False

#Extract Contacts
def find_contact(s):
  s=s.split(":")
  if len(s)==2:
    return True
  else :
    return False

#Extract Message
def getMessage(line):
  splitline = line.split(" - ")
  datetime = splitline[0]
  date, time = datetime.split(", ")
  message = " ".join(splitline[1:])

  if find_contact(message):
    splitmessage = message.split(": ")
    author = splitmessage[0]
    message = " ".join(splitline[1:])

  else :
    author = None
  return date, time, author, message


# In[3]:


import io
from io import StringIO
import streamlit as st
data=[]
st.sidebar.title("Sentiment Analyzer")
dp = st.sidebar.file_uploader("Choose a File")
if dp is not None:
    fp = StringIO(dp.getvalue().decode("utf-8"))
    #fp = dp.decode("utf-8")
    fp.readline()
    messageBuffer = []
    date, time, author = None, None, None
    while True:
        line= fp.readline()
        if not line:
            break
        line=line.strip()
        if date_time(line):
            if len(messageBuffer)>0:
                data.append([date, time, author, ''.join(messageBuffer)])
        messageBuffer.clear()
        date, time, author, message = getMessage(line)
        messageBuffer.append(message)
    else:
        messageBuffer.append(line)


# In[4]:


import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('vader_lexicon')


# In[5]:


import pandas as pd
df = pd.DataFrame(data, columns=['Date', 'Time', 'Contact', 'Message'])
df['Date']= pd.to_datetime(df['Date'])
data = df.dropna()
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentiments = SentimentIntensityAnalyzer()
data['positive']=[sentiments.polarity_scores(i)['pos'] for i in data['Message']]
data['negative']=[sentiments.polarity_scores(i)['neg'] for i in data['Message']]
data['neutral']=[sentiments.polarity_scores(i)['neu'] for i in data['Message']]
st.write(data.head(100))


# In[7]:


st.sidebar.subheader("Deep Analyser")
select = st.sidebar.selectbox('Sentiment Type', ['Positive', 'Negative', 'Neutral'],key=1)
if select == 'Positive':
  st.write("The total positive sentiment is", ((sum(data['positive'][:20]))/20)*100, '%')
if select == 'Negative':
  st.write("The total negative sentiment is", ((sum(data['negative'][:20]))/20)*100, '%')
if select == 'Neutral':
  st.write("The total neutral sentiment is", ((sum(data['neutral'][:20]))/20)*100, '%')

# In[ ]:




