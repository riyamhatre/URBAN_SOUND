#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import librosa
import librosa.display
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import IPython.display as ipd

data = pd.read_csv("UrbanSound8K.csv")


# In[5]:


data['class'].value_counts()


# In[6]:


data.groupby('fold').count().get('start')


# In[7]:


data.head()


# In[8]:


filename = 'audio/fold1/7061-6-0-0.wav'


# In[6]:


plt.figure(figsize=(14,5))
data,sample_rate=librosa.load(filename)
librosa.display.waveplot(data, sr=sample_rate)
ipd.Audio(filename)


# In[7]:


sample_rate


# In[8]:


from scipy.io import wavfile as wav
wave_sample_rate, wave_audio = wav.read(filename)


# In[9]:


wave_sample_rate


# In[ ]:





# In[1]:



import librosa
audio_file_path='UrbanSound8K/100263-2-0-3.wav'
librosa_audio_data,librosa_sample_rate=librosa.load(audio_file_path)


# In[ ]:




