#!/usr/bin/env python
# coding: utf-8

# In[148]:


import tensorflow as tf


# In[2]:


import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd


# In[168]:


import librosa
import librosa.display
from tqdm import tqdm
import glob


# # Ouverture de l'audio et affichage temporel

# In[905]:


from IPython import display as ipd
ipd.Audio('cat&dog/cats_dogs/dog_barking_4.wav')


# In[907]:


audio, sr = librosa.load('cat&dog/cats_dogs/dog_barking_4.wav',22050, duration =5 )
pad_len = 110250 - len(audio)
print(len(audio))
audio = np.pad(audio, (0, pad_len))
plt.plot(audio)


# In[383]:


mfcc = librosa.feature.mfcc(audio,n_mfcc=5)
#mfcc.shape
plt.imshow(mfcc,aspect='auto')


# # Spectrogram

# In[625]:


def calcul_median(spectrogram):
    median=[]
    line, colone=spectrogram.shape
    for i in range (0,line):
        new_line=[]
        for j in range(0,colone):
            if spectrogram[i][j]>0:
                new_line.append(spectrogram[i][j])
        med=np.median(new_line)
        if np.isnan(med):
            median.append(0)
        else:
            median.append(med)
    return  median


# In[902]:


file_path='cat&dog/cats_dogs/dog_barking_4.wav'
x, sr = librosa.load(file_path)
X = librosa.stft(x)

Xdb = librosa.amplitude_to_db(abs(X))
print(Xdb.shape)
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr = sr, x_axis = 'time', y_axis = 'hz')
#plt.ylabel('Log')
plt.colorbar()


# In[903]:


median=calcul_median(X)
  
plt.plot(median)


# In[898]:


file_path1='cat&dog/cats_dogs/cat_2.wav'
x, sr = librosa.load(file_path1)
X = librosa.stft(x)

Xdb = librosa.amplitude_to_db(abs(X))
print(Xdb.shape)
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr = sr, x_axis = 'time', y_axis = 'hz')
#plt.ylabel('Log')
plt.colorbar()


# In[899]:


median=calcul_median(X)
  
plt.plot(median)


# In[ ]:





#  
# 

# In[ ]:





# In[ ]:





# In[529]:





# In[430]:



    
    
    
    
    


# In[ ]:





# In[ ]:





# In[ ]:





# # Preprocessing

# In[795]:



def audio_processing(file):
    audio_list=[]
    name_file=[]
    for name in tqdm(glob.glob(file+'/*/*')):
        x, sr = librosa.load(name, sr=44100)
        X = librosa.stft(x)
        #Xdb = librosa.amplitude_to_db(abs(X))
        Y=calcul_median(abs(X))
        #Y=np.transpose(Y)
        #Y=tf.constant(Y)
        audio_list.append(Y)
        name_file.append(name)
    return audio_list, name_file

    
    


# In[908]:


audio_list,name_file=audio_processing('cat&dog/cats_dogs/train')


# Mise en forme dataframe 

# In[909]:


d={'name_file':name_file,'median spectrogram':audio_list}
df=pd.DataFrame(data=d)
df['cat_or_dog']=0 
df.cat_or_dog[df['name_file'].str.contains('barking')]=1
df


# In[910]:


label=np.array(df['cat_or_dog'])


# In[911]:


audio_list1=np.array(audio_list)


# In[912]:


audio_list2,name_file2=audio_processing('cat&dog/cats_dogs/test')


# In[913]:


d1={'name_file':name_file2,'median spectrogram':audio_list2}
df1=pd.DataFrame(data=d1)
df1['cat_or_dog']=0 
df1.cat_or_dog[df1['name_file'].str.contains('barking')]=1
df1


# In[914]:


label2=np.array(df1['cat_or_dog'])
audio_list2=np.array(audio_list2)


# In[915]:


input_shape=audio_list1[1].shape
input_shape


# In[916]:


import numpy as np
import pandas as pd
import os
import pathlib
import matplotlib.pyplot as plt
import seaborn

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

from sklearn.metrics import classification_report, log_loss, accuracy_score


# In[917]:


tf.config.run_functions_eagerly(True)


# In[918]:


input_tensor = layers.Input(shape = input_shape)
x=layers.Dense(20,activation='relu')(input_tensor)
x=layers.Dense(20,activation='relu')(x)
output_tensor = layers.Dense(1, activation = 'sigmoid')(x)
model = tf.keras.Model(input_tensor, output_tensor)
model.summary()


# In[919]:


model.compile(optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryFocalCrossentropy(from_logits=False),
    metrics=['accuracy'],
)


# In[920]:


his=model.fit(audio_list1, label,epochs = 50,validation_data =(audio_list2,label2),shuffle = True,)


# In[921]:


get_acc = his.history['accuracy']
val_accu=his.history['val_accuracy']
epochs = range(len(get_acc))
plt.plot(epochs, get_acc, val_accu, label='Accuracy of Training data')
plt.title('Training accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()


# In[922]:


get_loss = his.history['loss']
val_loss=his.history['val_loss']
epochs = range(len(get_loss))
plt.plot(epochs, get_loss, 'r', label='Loss of Training data')
plt.plot(epochs,val_loss,'b',label='Loss of Validation data')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.legend(loc=0)
plt.figure()
plt.show()


# In[694]:


#model.predict(audio_list)


# In[875]:


y_pred0=model.predict(audio_list1)
#y_pred=np.argmax(y_pred0,axis=1)
#y_true = np.array(label1)
y_pred0


# In[ ]:





# In[888]:


input_tensor = layers.Input(shape = input_shape)
x=layers.Dense(20,activation='relu')(input_tensor)
x=layers.Dense(20,activation='relu')(x)
output_tensor = layers.Dense(1, activation = 'sigmoid')(x)
model = tf.keras.Model(input_tensor, output_tensor)
model.summary()


# In[889]:


model.compile(optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryFocalCrossentropy(from_logits=False),
    metrics=['accuracy'],
)


# In[890]:


his=model.fit(audio_list1, label,epochs = 15,validation_data =(audio_list2,label2),shuffle = True)


# In[891]:


get_loss = his.history['loss']
val_loss=his.history['val_loss']
epochs = range(len(get_loss))
plt.plot(epochs, get_loss, 'r', label='Loss of Training data')
plt.plot(epochs,val_loss,'b',label='Loss of Validation data')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.legend(loc=0)
plt.figure()
plt.show()


# In[854]:


audio_list3,name_file3=audio_processing('cat&dog/cats_dogs/val_final')


# In[855]:


d2={'name_file':name_file3,'median spectrogram':audio_list3}
df2=pd.DataFrame(data=d2)
df2['cat_or_dog']=0 
df2.cat_or_dog[df2['name_file'].str.contains('barking')]=1
df2


# In[883]:


label3=np.array(df2['cat_or_dog'])
label3


# In[884]:


audio_list3=np.array(audio_list3)


# In[892]:


#tf.data.experimental.enable_debug_mode()
y_pred0=model.predict(audio_list3)
#y_pred=np.argmax(y_pred0,axis=1)
#y_true = np.array(label3)
y_pred0


# In[ ]:





# In[894]:


score = model.evaluate(audio_list3,label3)
print(score)


# In[ ]:





# In[ ]:




