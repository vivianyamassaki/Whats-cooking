#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering

# Este notebook contém a feature engineering dos conjuntos de dados de treino e teste, que serão realizados com base na EDA realizada anteriormente.
# 
# Ao final desse notebook, serão gerados dois conjuntos de dados de treino e teste processados e que serão fornecidos para o modelo classificador.

# ## Imports

# #### Importação das bibliotecas utilizadas

# In[1]:


import pandas as pd
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize as TK

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


# #### Importação do conjunto de treinamento e teste

# O conjuntos de treinamento e de teste estão em formato JSON, portando é necessário usar a função __read_json__ do pandas para realizar a leitura dos arquivos:

# In[3]:


train = pd.read_json('../dados/train/train.json', encoding='utf-8')
test = pd.read_json('../dados/test/test.json', encoding='utf-8')


# In[4]:


print(f"O conjunto de treinamento possui {train.shape[0]} exemplos, cada qual representando uma receita, e {train.shape[1]} colunas, que são as seguintes:") 


# * **cuisine**: tipo de culinária da qual a receita pertence. Essa é a variável resposta (a classe que o modelo a ser criado deve prever)
# * **id**: número identificador único da receita
# * **ingredients**: lista de ingredientes que compõem a receita

# In[5]:


# visualização das cinco primeira receitas presentes no conjunto de treino
train.head()


# In[6]:


print(f"O conjunto de teste possui {test.shape[0]} exemplos, cada qual representando uma receita, e {test.shape[1]} colunas. Diferentemente do que ocorreu com o conjunto de treino, a classe não foi fornecida para o conjunto de teste.") 


# In[7]:


# visualização das cinco primeira receitas presentes no conjunto de teste
test.head()


# ## Tratamento dos dados
# 
# Os seguintes tratamentos serão realizados nos dois conjuntos de dados:
#   * Conversão de todas as palavras para letras minúsculas
#   * Remoção de caracteres numéricos e especiais
#   * Remoção de palavras indicando unidades de medida
#   * Remoção de stopwords
#   * Lematização das palavras

# In[8]:


# função para contabilizar a quantidade de ingredientes únicos
def unique(data):
    return list(dict.fromkeys(data))


# In[9]:


print(f"Antes do tratamento dos dados, o conjunto de treino possui {len(unique([j for i in train['ingredients'] for j in i]))} ingredientes únicos.")


# In[10]:


print(f"Antes do tratamento dos dados, o conjunto de teste possui {len(unique([j for i in test['ingredients'] for j in i]))} ingredientes únicos.")


# #### Conversão para letras minúsculas

# In[11]:


train['ingredients'] = train['ingredients'].apply(lambda x: list(map(lambda x:x.lower(),x)))
test['ingredients'] = test['ingredients'].apply(lambda x: list(map(lambda x:x.lower(),x)))


# #### Remoção de unidades medidas

# As unidades de medida abaixo e que foram encontradas durante a EDA serão removidas dos dois conjuntos de dados:
# 
# * [oz.](https://en.wikipedia.org/wiki/Ounce)
# * [ounc](https://en.wikipedia.org/wiki/Ounce) (Ounce escrita de maneira incorreta)
# * [lb.](https://en.wikipedia.org/wiki/Pound_(mass))
# * [pound](https://en.wikipedia.org/wiki/Pound_(mass))
# * [inch](https://en.wikipedia.org/wiki/Inch)

# In[12]:


train['ingredients'] = train['ingredients'].apply(lambda x: 
                                        list(map(lambda x:x.replace('oz.', '')
                                                 .replace('ounc', '')
                                                 .replace('lb.', '')
                                                 .replace('pound', '')
                                                 .replace(' inch', '')
                                                 ,x)))
test['ingredients'] = test['ingredients'].apply(lambda x: 
                                        list(map(lambda x:x.replace('oz.', '')
                                                 .replace('ounc', '')
                                                 .replace('lb.', '')
                                                 .replace('pound', '')
                                                 .replace(' inch', '')
                                                 ,x)))


# #### Remoção números e caracteres especiais

# In[13]:


# retira os caracteres numéricos 
train['ingredients'] = train['ingredients'].apply(lambda x: list(map(lambda x:re.sub('[0-9]', '',x),x)))
test['ingredients'] = test['ingredients'].apply(lambda x: list(map(lambda x:re.sub('[0-9]', '',x),x)))

# retira os caracteres especiais
train['ingredients'] = train['ingredients'].apply(lambda x: list(map(lambda x:re.sub('[!%&/.,®€™()]', '',x),x)))
test['ingredients'] = test['ingredients'].apply(lambda x: list(map(lambda x:re.sub('[!%&/.,®€™()]', '',x),x)))

# no caso do hífen, ápice e apóstrofo, será dado um espaço para que as palavras separadas não sejam agrupadas em uma só
train['ingredients'] = train['ingredients'].apply(lambda x: list(map(lambda x:re.sub('[-’\']', ' ',x),x)))
test['ingredients'] = test['ingredients'].apply(lambda x: list(map(lambda x:re.sub('[-’\']', ' ',x),x)))


# É necessário retirar os acentos de algumas palavras, visto que há casos do mesmo ingrediente escrito de maneiras diferentes no conjunto de dados (por exemplo, o ingrediente `açaí` está escrito como `açai` e `acai`):

# In[14]:


train['ingredients'] = train['ingredients'].apply(lambda x: 
                                        list(map(lambda x:x.replace('â', "a")
                                                 .replace('ç', 'c')
                                                 .replace('è', 'e')
                                                 .replace('é', 'e')
                                                 .replace('í', 'i')
                                                 .replace('î', 'i')
                                                 .replace('ú', 'u')
                                                 ,x)))
test['ingredients'] = test['ingredients'].apply(lambda x: 
                                        list(map(lambda x:x.replace('â', "a")
                                                 .replace('ç', 'c')
                                                 .replace('è', 'e')
                                                 .replace('é', 'e')
                                                 .replace('í', 'i')
                                                 .replace('î', 'i')
                                                 .replace('ú', 'u')
                                                 ,x)))


# #### Remoção stopwords

# In[15]:


# lista de stopwords em inglês
stop_words = set(stopwords.words('english')) 

# função para remover as stopwords da lista de ingredientes das receitas
def remove_stopwords(df, column):
    for i, row in df.iterrows():
        ingredients_without_stopwords = []
        for sentence in df.at[i, column]:
            temp_list=[]
            for word in sentence.split():
                if word.lower() not in stop_words:
                    temp_list.append(word)
            ingredients_without_stopwords.append(' '.join(temp_list))
        df.at[i, column] = ingredients_without_stopwords
    return df[column]


# In[16]:


train['ingredients'] = remove_stopwords(train,'ingredients')
test['ingredients'] = remove_stopwords(test,'ingredients')


# #### Lematização dos ingredientes

# In[17]:


# converte as palavras do plural para o singular
lemmatizer = WordNetLemmatizer()

def lemmatize(ingredients):
    for i in ingredients.index.values:
        for j in range(len(ingredients[i])):
            ingredients[i][j] = ingredients[i][j].strip()
            token = TK(ingredients[i][j])
            for k in range(len(token)):
                token[k] = lemmatizer.lemmatize(token[k])
            token = ' '.join(token)
            ingredients[i][j] = token
    return ingredients


# In[18]:


train['ingredients'] = lemmatize(train.ingredients)
test['ingredients'] = lemmatize(test.ingredients)


# #### Remoção dos espaços em branco em excesso

# Por conta dos tratamentos realizadas, alguns espaços a mais ficaram sobrando no nome dos ingredientes (por exemplo, o ingrediente `7 Up` não existe mais porque seu nome era formado apenas por um número e uma stopword, que foram removidos). O código abaixo resolve esse problema:

# In[19]:


def remove_empty_ingredients(df, column):
    for i, row in df.iterrows():
        df.at[i, column] = [x.strip() for x in df.at[i, column] if x.strip()]
    return df[column]


# In[20]:


train['ingredients'] = remove_empty_ingredients(train,'ingredients')
test['ingredients'] = remove_empty_ingredients(test,'ingredients')


# In[21]:


print(f"Após o tratamento dos dados, restaram {len(unique([j for i in train['ingredients'] for j in i]))} ingredientes únicos no conjunto de treino.")


# In[22]:


print(f"Após o tratamento dos dados, restaram {len(unique([j for i in test['ingredients'] for j in i]))} ingredientes únicos no conjunto de teste.")


# ## Limpeza dos dados

# Com base na EDA realizada anteriormente, foram identificadas as seguintes tarefas de limpeza de dados:
# 
# * Exclusão das receitas repetidas por tipo de cozinha
# * Exclusão de outliers (receitas com apenas 1 ingrediente ou com mais de 40)
# * Exclusão de ingredientes repetidos na lista de ingredientes de algumas receitas
# 
# As duas primeiras tarefas serão realizadas somente no conjunto de treinamento, visto que o conjunto de teste não não possui a classe, que permitiria identificar as receitas repetidas, e porque não podem ser removidos os outliers, visto que o arquivo a ser fornecido para o Kaggle deve conter as predições para todos os exemplos existentes no conjunto de teste.

# #### Exclusão das receitas repetidas por tipo de cozinha

# In[23]:


train["ingredients_text"] = train["ingredients"].apply(lambda x: ", ".join(x))
train.drop_duplicates(subset=['cuisine','ingredients_text'], keep='first', inplace=True)


# #### Exclusão de outliers

# In[24]:


train['ingredients_qtt'] = train['ingredients'].apply(lambda x: len(x))
train = train[(train['ingredients_qtt'] <= 40) | (train['ingredients_qtt'] > 1)]


# #### Exclusão de ingredientes repetidos na lista de ingredientes

# In[25]:


# remoção dos ingredientes duplicados
train['ingredients'] = train['ingredients'].apply(lambda x: list(set(x)))
test['ingredients'] = test['ingredients'].apply(lambda x: list(set(x)))


# ### Criação de novas features
# 
# Duas novas features serão adicionadas aos conjuntos de dados:
# * `ingredients_qtt`: indica a quantidade de ingredientes presentes na receita
# * `ingredients_text`: lista de ingredientes presentes na receita. Ao invés de ser uma lista de strings, é somente uma string na qual os ingredientes estão separados por vírgula

# #### ingredients_qtt

# In[26]:


# cria uma nova coluna contendo a quantidade de ingredientes presentes em cada receita
train['ingredients_qtt'] = train['ingredients'].apply(lambda x: len(x))
test['ingredients_qtt'] = test['ingredients'].apply(lambda x: len(x))


# #### ingredients_text

# In[27]:


# cria uma nova coluna contendo a tranformação da lista de ingredientes em uma única string, com os ingredientes 
# separados por vírgula
train["ingredients_text"] = train["ingredients"].apply(lambda x: " ".join(x))
test["ingredients_text"] = test["ingredients"].apply(lambda x: " ".join(x))


# #### Conjuntos de dados finais

# In[28]:


print(f"O conjunto de treinamento processado possui {train.shape[0]} exemplos e {train.shape[1]} colunas:") 


# In[29]:


# visualização das cinco primeira receitas presentes no conjunto de treino
train.head()


# In[30]:


print(f"O conjunto de teste processado possui {test.shape[0]} exemplos e {test.shape[1]} colunas:") 


# In[31]:


# visualização das cinco primeira receitas presentes no conjunto de teste
test.head()


# ## Aplicação do TF-IDF e exportação dos conjuntos de dados processados

# Após o pré-processamento dos dados, é necessário salvá-los para que sejam fornecidos aos modelos.
# 
# ### Conjunto de treino
# #### Salva as classes do conjunto de treino
# 
# Como as classes presentes no conjunto de dados são categóricas, é necessário convertê-las para variáveis numéricas. Essa conversão é realizada abaixo:

# In[32]:


encoder = LabelEncoder()
y_train = encoder.fit_transform(train.cuisine)


# In[33]:


# lista o nome da cozinha e o número que orá representá-la
list(zip(encoder.classes_, encoder.transform(encoder.classes_)))


# In[34]:


# salva as classes após a aplicação do encoder
pd.DataFrame(y_train, columns=['target']).to_csv("../dados/train/train_target.csv", index=False)


# #### Salva o conjunto de treino vetorizado

# Como os modelos não trabalham com dados textuais, será necessário realizar a vetorização da lista de ingredientes utilizando o [TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).

# In[35]:


train = train.reset_index(drop=True)


# In[36]:


vectorizer = make_pipeline(
        TfidfVectorizer(binary=True),
        FunctionTransformer(lambda x: x.astype('float'), validate=False)
    )


# In[37]:


tfidf = vectorizer.fit_transform(train['ingredients_text'])
tfidf = pd.DataFrame(tfidf.toarray())


# In[38]:


print (tfidf.shape)


# In[39]:


X_train = pd.concat([train, tfidf],axis=1)


# In[40]:


X_train.to_csv("../dados/train/processed_train.csv", index=False)


# #### Salva o conjunto de treino vetorizado e padronizado
# 
# Como um dos modelos a ser utilizado é uma regressão logística, é necessário que os dados estejam na mesma escala. Como a escala da quantidade de ingredientes é diferente do array gerado pelo TF-IDF, é necessário gerar outro conjunto de dados padronizado:

# In[41]:


scaler = StandardScaler()
scaler.fit(pd.concat([train['ingredients_qtt'], tfidf],axis=1))
X_train_scaled = scaler.transform(pd.concat([train['ingredients_qtt'], tfidf],axis=1))


# In[42]:


X_train_scaled = pd.concat([train, pd.DataFrame(X_train_scaled)],axis=1)


# In[43]:


X_train_scaled.to_csv("../dados/train/processed_train_scaled.csv", index=False)


# ### Conjunto de teste
# 
# Para o conjunto de teste, também será salvo o dataset vetorizado e também a versão vetorizada e padronizada:
# 
# #### Salva as classes do conjunto de teste vetorizado

# In[44]:


tfidf = vectorizer.transform(test['ingredients_text'])
tfidf = pd.DataFrame(tfidf.toarray())


# In[45]:


print (tfidf.shape)


# In[46]:


X_test = pd.concat([test, tfidf],axis=1)


# In[47]:


X_test.to_csv("../dados/test/processed_test.csv", index=False)


# #### Salva o conjunto de teste vetorizado e padronizado

# In[48]:


scaler.fit(pd.concat([test['ingredients_qtt'], tfidf],axis=1))
X_test_scaled = scaler.transform(pd.concat([test['ingredients_qtt'], tfidf],axis=1))


# In[49]:


X_test_scaled = pd.concat([test, pd.DataFrame(X_test_scaled)],axis=1)


# In[50]:


X_test_scaled.to_csv("../dados/test/processed_test_scaled.csv", index=False)

