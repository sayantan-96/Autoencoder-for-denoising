#!/usr/bin/env python
# coding: utf-8

# In[1]:


#imports
import numpy as np
import pandas as pd
import os
import tensorflow.keras.models
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, mean_squared_error,r2_score, mean_squared_error, mean_absolute_error


# In[2]:


#Data load
datapath=os.getcwd()+"\\time-series_Dataset\\time-series_data\\"
xtrain=pd.read_csv(datapath+"X_train.csv",header=None)
xval=pd.read_csv(datapath+"X_val.csv",header=None)
xtest=pd.read_csv(datapath+"X_test.csv",header=None)
ytest=pd.read_csv(datapath+"Y_test.csv",header=None)
ytrain=pd.read_csv(datapath+"Y_train.csv",header=None)
yval=pd.read_csv(datapath+"Y_val.csv",header=None)


# In[3]:


#My test data is from 2461-2480, also concatinate train and vaidation data
xtest=xtest.iloc[2461:2481,:].copy()
ytest=ytest.iloc[2461:2481,:].copy()
xtrain=pd.concat([xtrain,xval],axis=0)
ytrain=pd.concat([ytrain,yval],axis=0)


# In[4]:


#convert to numpy arrays
xt=xtrain.to_numpy()
yt=ytrain.to_numpy()
xtst=xtest.to_numpy()
ytst=ytest.to_numpy()


# In[5]:


#shape_checks
train_shape=xt.shape
test_shape=xtst.shape
if((train_shape[1]==test_shape[1]) and (xt.shape==yt.shape) and(test_shape==ytst.shape)):
    print("compatible shapes of data present")
else:
    print("data dimension error")


# In[6]:


#Deep network for the autoencoder
def deep_network(dimension):
    model=Sequential()
    model.add(Dense(200,input_dim=dimension,activation='relu',kernel_regularizer=regularizers.l2(1e-3)))
    model.add(Dropout(0.5))
    model.add(Dense(100,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(50,activation='tanh'))
    model.add(Dense(100,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(200,activation='relu',kernel_regularizer=regularizers.l2(1e-3)))
    model.add(Dropout(0.5))
    model.add(Dense(dimension,activation="sigmoid"))
    opti = keras.optimizers.Adam(lr = 0.003)
    model.compile(loss="mse",optimizer="adam",metrics = ["mae", "mse"])
    return model
    


# In[7]:


model=deep_network(train_shape[1])
checkpointer = [EarlyStopping(monitor='val_loss', patience=7)]
batches = 2048
epoch_num = 40


# In[8]:


history = model.fit(xt,yt, batch_size=batches, epochs = epoch_num, validation_data=(xtst, ytst),callbacks= checkpointer)


# In[9]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# ###Performance Evaluation

# In[10]:


#Train R squared and Mean squared Errors
yhat_train=model.predict(xt,batch_size=1024)
mae_train = mean_absolute_error(yt,yhat_train)
rmse_train = np.sqrt(mean_squared_error(yt, yhat_train))
print('Train MAE      :',mae_train)
print('Train Mean     :',yt.mean())
print('Train MAPE     :',mae_train/yt.mean())
print('Train RMSE     :',rmse_train)


# In[11]:


#test R squared and Mean squared Errors
yhat_test=model.predict(xtst,batch_size=20)
mae_test = mean_absolute_error(ytst,yhat_test)
rmse_test = np.sqrt(mean_squared_error(ytst, yhat_test))
print('Test MAE      :',mae_test)
print('Test Mean     :',ytst.mean())
print('Test MAPE     :',mae_test/ytst.mean())
print('Test RMSE     :',rmse_test)


# In[12]:


#Visualization of the data
#tsne plot
from sklearn.manifold import TSNE
y_pred = TSNE().fit_transform(yhat_test)


# In[13]:


y_tru=TSNE().fit_transform(ytst)


# In[14]:


plt.scatter(y_tru[:,0], y_tru[:,1],label='True noiseless data')
plt.scatter(y_pred[:,0], y_pred[:,1],label='Denoised data')
plt.legend()

