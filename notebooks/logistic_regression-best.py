#!/usr/bin/env python
# coding: utf-8

# # Melhor modelo - Regressão Logística

# Este notebook contém a modelagem e a avaliação do modelo A, que utiliza uma regressão logística.

# ### Imports

# #### Importação das bibliotecas utilizadas

# In[64]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report, accuracy_score
import pickle
import numpy as np
import matplotlib.pyplot as plt
import itertools 


# #### Importação do conjunto de treinamento e teste

# O conjuntos de treinamento e de teste estão em formato JSON, portando é necessário usar a função __read_json__ do pandas para realizar a leitura dos arquivos:

# In[2]:


train = pd.read_csv('../dados/train/processed_train_scaled.csv', encoding='utf-8')
train_target = pd.read_csv('../dados/train/train_target.csv', encoding='utf-8')


# In[3]:


# visualização das cinco primeira receitas presentes no conjunto de treino
train.head(2)


# ### Modelagem
# 
# Os seguintes tratamentos serão realizados nos dois conjuntos de dados:

# In[17]:


X_train, X_test, y_train, y_test = train_test_split(train, train_target, train_size=0.7, random_state=42)


# In[5]:


X_train.head()


# In[6]:


X_train = X_train.drop(['cuisine','id','ingredients','ingredients_text','ingredients_qtt'], axis=1)


# In[7]:


logit = LogisticRegression(random_state=42) 


# In[8]:


model = OneVsRestClassifier(logit)

parameters = [
  {'estimator__C': [0.1,1,10],
             'estimator__penalty':['l2'],     
             'estimator__solver':['lbfgs']},
  {'estimator__C': [0.1,1,10],
             'estimator__penalty':['l2','l1'],     
             'estimator__solver':['saga']}
 ]

grid_obj = GridSearchCV(model,parameters, scoring = 'f1_weighted', verbose=3)
grid_fit = grid_obj.fit(X_train, y_train)

melhor_modelo = grid_fit.best_estimator_
melhor_modelo


# In[9]:


with open('../models/logit_best.pkl', 'wb') as local_model_file:
    pickle.dump(melhor_modelo, local_model_file)


# In[21]:


test = X_test.copy()
X_test = X_test.drop(['id','ingredients','ingredients_text','ingredients_qtt', 'cuisine'], axis=1)


# In[22]:


y_pred_test = melhor_modelo.predict(X_test)


# ### Predições para o conjunto de teste

# Como as classes foram encodificadas para serem fornecidas para o modelo. É necessário fazer a decodificação:

# In[12]:


encoder = LabelEncoder()
classes = encoder.fit_transform(train.cuisine)


# In[13]:


list(zip(encoder.classes_, encoder.transform(encoder.classes_)))


# In[14]:


y_test_decoded = encoder.inverse_transform(y_pred_test)


# In[15]:


y_test_decoded


# In[31]:


test = test.reset_index()

true_pred_y_test = pd.concat([test['cuisine'],pd.DataFrame(y_test_decoded, columns=[ 'cuisine_pred'])], axis=1)


# In[47]:


test['cuisine'].unique()


# In[44]:


true_pred_y_test.head()


# In[51]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[56]:


cnf_matrix = confusion_matrix(test['cuisine'], y_test_decoded, labels=test['cuisine'].unique())
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(20,10))
plot_confusion_matrix(cnf_matrix, classes=test['cuisine'].unique(),title='Matriz de confusão')


# In[62]:


print(classification_report(test['cuisine'], y_test_decoded))


# In[65]:


accuracy_score(test['cuisine'], y_test_decoded)

