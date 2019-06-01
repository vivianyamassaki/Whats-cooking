#!/usr/bin/env python
# coding: utf-8

# # Modelo A - Regressão Logística

# Este notebook contém a modelagem e a avaliação do modelo A, que utiliza uma regressão logística.

# ### Imports

# #### Importação das bibliotecas utilizadas

# In[1]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
import pickle


# #### Importação do conjunto de treinamento e teste

# O conjuntos de treinamento e de teste estão em formato JSON, portando é necessário usar a função __read_json__ do pandas para realizar a leitura dos arquivos:

# In[2]:


train = pd.read_csv('../dados/train/processed_train_scaled.csv', encoding='utf-8')
train_target = pd.read_csv('../dados/train/train_target.csv', encoding='utf-8')
test = pd.read_csv('../dados/test/processed_test_scaled.csv', encoding='utf-8')


# In[3]:


# visualização das cinco primeira receitas presentes no conjunto de treino
train.head(2)


# ### Modelagem
# 
# Os seguintes tratamentos serão realizados nos dois conjuntos de dados:

# In[4]:


X_train = train.copy()
X_train = X_train.drop(['cuisine','id','ingredients','ingredients_text','ingredients_qtt'], axis=1)


# In[5]:


X_train.head()


# In[6]:


logit = LogisticRegression(random_state=42, solver='lbfgs') #, multi_class='ovr'


# In[7]:


model = OneVsRestClassifier(logit, n_jobs=1)
model.fit(X_train, train_target)


# In[8]:


with open('../models/logit.pkl', 'wb') as local_model_file:
    pickle.dump(model, local_model_file)


# In[9]:


test.head(2)


# In[10]:


X_test = test.copy()
X_test = X_test.drop(['id','ingredients','ingredients_text','ingredients_qtt'], axis=1)


# In[11]:


y_pred_test = model.predict(X_test)


# ### Predições para o conjunto de teste

# Como as classes foram encodificadas para serem fornecidas para o modelo. É necessário fazer a decodificação:

# In[12]:


encoder = LabelEncoder()
classes = encoder.fit_transform(train.cuisine)


# In[15]:


list(zip(encoder.classes_, encoder.transform(encoder.classes_)))


# In[13]:


classes


# In[14]:


y_pred_test


# In[16]:


y_test_decoded = encoder.inverse_transform(y_pred_test)


# In[17]:


y_test_decoded


# In[24]:


submission = pd.concat([test['id'],pd.DataFrame(y_test_decoded, columns=[ 'cuisine'])], axis=1)


# In[27]:


submission.to_csv("../dados/test/submission_logit.csv", index=False)

