#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis (EDA) e visualização exploratória

# Este notebook tem como objetivo apresentar a análise exploratória realizada no conjunto de dados de treinamento brevemente descrito na [proposta do capstone](https://github.com/vivianyamassaki/Whats-cooking/blob/master/README.md), assim como a visualização exploratória para o problema denominado [What's cooking?](https://www.kaggle.com/c/whats-cooking) .

# #### Versão do Python utilizada

# In[1]:


from platform import python_version

print('Versão do Python: ' + python_version())


# In[2]:


get_ipython().system('pip freeze')


# In[3]:


import wordcloud


# #### Importação das bibliotecas utilizadas

# In[4]:


import pandas as pd
import json

# para visualização dos dados
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


# In[5]:


# altera o tamanho das imagens
plt.rcParams["figure.figsize"] = (12, 9)


# #### Importação do conjunto de treinamento

# O conjunto de treinamento está em formato JSON, portando é necessário usar a função __read_json__ do pandas para realizar a leitura do arquivo.

# In[6]:


train = pd.read_json('../dados/train.json')


# ### Descrição geral do conjunto de dados

# In[7]:


print(f"O conjunto de treinamento possui {train.shape[0]} exemplos, cada qual representando uma receita, e {train.shape[1]} colunas, que são as seguintes:") 


# * **cuisine**: tipo de culinária da qual a receita pertence. Essa é a variável resposta (a classe que o modelo a ser criado deve prever)
# * **id**: número identificador único da receita
# * **ingredients**: lista de ingredientes que compõem a receita

# Com base na coluna contendo a lista de ingredientes, serão criadas duas novas colunas para auxiliar a análise dos dados:

# In[8]:


# cria uma nova coluna contendo a tranformação da lista de ingredientes em uma única string, com os ingredientes 
# separados por vírgula
train["ingredients_text"] = train["ingredients"].apply(lambda x: ", ".join(x))


# In[9]:


# cria uma nova coluna contendo a quantidade de ingredientes presentes em cada receita
train['ingredients_qtt'] = train['ingredients'].apply(lambda x: len(x))


# In[10]:


# visualização das cinco primeira receitas presentes no conjunto de dados
train.head()


# In[11]:


# print de algumas informações sobre o conjunto de treinamento
train.info()


# Como é possível observar acima, não há nenhuma coluna no conjunto de treinamento com valores faltantes, visto que não há nenhum exemplo com dados nulos.
# 
# Além disso, também é possível identificar os tipos de informações presentes no conjunto de dados:
# 
# |coluna|tipo|
# |------|----|
# |`cuisine`| string|
# |`id`| int|
# |`ingredients`| lista de strings|
# |`ingredients_text`| string|
# |`ingredients_qtt`| int|

# In[12]:


print(f"Com relação à variável resposta, temos {len(train.cuisine.unique())} tipos de cozinha diferentes presentes no conjunto de treinamento. Isso significa que o problema a ser resolvido é multiclasse, visto que há mais de duas classes presentes no conjunto de dados.")


# Esses 20 tipos diferentes de cozinha estão descritos na tabela abaixo. A tabela contém uma coluna contando o número da cozinha (apenas para facilitar a contagem das 20 cozinhas), a bandeira do país de origem do tipo de cozinha, o nome da cozinha presente no conjunto de dados e esse nome de cozinha traduzido para o português:

# |#|bandeira|nome no dataset| nome traduzido|
# |--|--|---|--|
# |1|<img src="../figuras/flags/brazil.png" width="40"/>|brazilian|brasileira|
# |2|<img src="../figuras/flags/united-kingdom.png" width="40"/>|british|britânica|
# |3|<img src="../figuras/flags/united-states.png" width="40"/>|cajun_creole| cajun/crioula<sup>1</sup>|
# |4|<img src="../figuras/flags/china.png" width="40"/>|chinese| chinesa|
# |5|<img src="../figuras/flags/philippines.png" width="40"/>|filipino| filipina|
# |6|<img src="../figuras/flags/france.png" width="40"/>|french| francesa|
# |7|<img src="../figuras/flags/greece.png" width="40"/>|greek| grega|
# |8|<img src="../figuras/flags/india.png" width="40"/>|indian| indiana|
# |9|<img src="../figuras/flags/ireland.png" width="40"/>|irish| irlandesa|
# |10|<img src="../figuras/flags/italy.png" width="40"/>|italian| italiana|
# |11|<img src="../figuras/flags/jamaica.png" width="40"/>|jamaican| jamaicana|
# |12|<img src="../figuras/flags/japan.png" width="40"/>|japanese| japonesa|
# |13|<img src="../figuras/flags/south-korea.png" width="40"/>|korean| coreana|
# |14|<img src="../figuras/flags/mexico.png" width="40"/>|mexican| mexicana|
# |15|<img src="../figuras/flags/morocco.png" width="40"/>|moroccan| marroquina|
# |16|<img src="../figuras/flags/russia.png" width="40"/>|russian| russa|
# |17|<img src="../figuras/flags/united-states.png" width="40"/>|southern_us|sulista (do sul dos Estados Unidos)|
# |18|<img src="../figuras/flags/spain.png" width="40"/>|spanish| espanhola|
# |19|<img src="../figuras/flags/thailand.png" width="40"/>|thai| tailandesa|
# |20|<img src="../figuras/flags/vietnam.png" width="40"/>|vietnamese| vietnamita|
# 
# <sup>1</sup> As culinárias cajun/crioula são originárias do estado da Luisiana, localizado na região sul dos Estados Unidos. Esse [link](https://www.louisianatravel.com/articles/cajun-vs-creole-food-what-difference) explica a diferença entre esses dois tipos de cozinha.
# 
# <sup>* Os ícones das bandeiras foram criados pelo [Freepik](https://www.flaticon.com/packs/international-flags-2), licenciado pela [Creative Commons BY 3.0](http://creativecommons.org/licenses/by/3.0/).</sup>

# ### Quantidade de receitas por tipo de cozinha

# In[13]:


# descreve a quantidade de receitas por tipo de cozinha e a proporção que ela representa no conjunto de dados
pd.concat([train['cuisine'].value_counts(),train['cuisine'].value_counts(normalize=True)*100], axis=1, keys=['qtd', '%'])


# In[14]:


# plota a quantidade de ingredientes por tipo de cozinha
sns.countplot(y='cuisine', data=train, order = train['cuisine'].value_counts().index)


# Por meio das informações e do gráfico acima, é possível notar que a cozinha com mais receitas presentes no conjunto de dados é a italiana, enquanto que a cozinha com menos receitas é a brasileira, que representam, respectivamente, 19.7% e 1.17% do conjunto de dados. 
# 
# Somente as três cozinhas com mais receitas (italiana, mexicana e sulista) contemplam quase metade de todas as receitas do conjunto de treinamento (46.75%).

# ### Quantidade de ingredientes

# In[15]:


train['ingredients_qtt'].describe()


# In[16]:


sns.boxplot(train['ingredients_qtt'])


# A mediana das receitas presentes no conjunto de treinamento é de  10 ingredientes. E, segundo o boxplot acima, receitas com mais de 20 ingredientes são consideradas _outliers_.
# 
# O número mínimo de ingredientes que uma receita presente no conjunto de treinamento contém é 1, enquanto o número máximo é 65. Seguem as receitas contendo o número mínimo de ingredientes:

# In[33]:


# lista as receitas com apenas 1 ingrediente
train[train['ingredients_qtt'] == 1]


# Ao todo, são 22 receitas existentes no conjunto de dados contendo somente 1 ingrediente. Observando essa lista de ingredientes, parece mais que eles podem ser usados em conjunto com outros ingredientes do que sendo usados sozinhos para criar um prato. Por exemplo, a receita da cozinha japonesa de id _12805_ tem como ingrediente apenas água e é difícil imaginar um prato criado somente com ela; já a outra receita japonesa de id _16116_ que tem como único ingrediente o udon, que é um tipo de macarrão japonês, mas só tem essa massa como ingrediente e não constam os demais ingredientes, como molho de soja e dashi, para preparação do caldo que é servido junto com o macarrão. Por esse motivo, todos esses 22 exemplos serão excluídos na etapa de pré-processamento.
# 
# Além disso, essa lista permitiu verificar que há receitas repetidas no conjunto de dados: as receitas da cozinha indiana de ids _19772_, _14335_, e _27192_ contém o mesmo único ingrediente (_unsalted butter_, ou manteiga sem sal, em português). Portanto, será necessário remover esses exemplos duplicados na etapa de pré-processamento dos dados antes de treinar o modelo (uma análise mais detalhada desses casos será realizado mais abaixo).
# 
# Já com relação ao nṹmero máximo de ingredientes, temos o seguinte caso:

# In[18]:


# lista as receitas com 65 ingredientes
train[train['ingredients_qtt'] == 65]


# In[19]:


train[train['ingredients_qtt'] == 65]['ingredients_text'].values


# A receita acima, cuja classe é de cozinha italiana, parece ser um dado escrito erroneamente. Isso porque contém diversos tipos de massa (fettucine, spaghetti, penne, orzo e até mesmo soba, que é um tipo de massa de origem japonesa, não italiana), de molhos, queijos (incluindo até um queijo francês, o Neufchâtel) e pimentas. Alguns ingredientes parecem ser os mesmos, mas escritos de maneiras diferentes (por exemplo, _freshly ground pepper_ e _ground pepper_). Com essa diversidade e quantidade enorme de ingredientes, pode-se tratar de diversas receitas que, por algum erro, acabaram agrupadas em uma única receita. 
# 
# Além desse caso extremo de 65 ingredientes, o boxplot acima indica que há outliers no conjunto de dados. Por meio dele, é possível observar que receitas a partir de 20 ingredientes são considerados outliers; entretanto, essas receitas estão mais próximas umas das outras e podem indicar receitas mais complexas que requerem mais ingredientes para serem preparadas. Por esse motivo, serão verificados somente os casos de receitas com mais de 40 ingredientes, visto que, a partir dessa quantidade de ingredientes, as receitas começam a ficar mais espaçadas no boxplot:

# Vendo as 6 receitas acima que possuem mais de 40 ingredientes, podemos notar que há algo estranho nelas. É possível observar que há receitas que misturam ingredientes doces com salgados (considerando apenas "doce" e "salgado" para simplificar). Seguem alguns exemplos com alguns dos ingredientes listados:
# 
# |Cozinha| Ingredientes "doces"| Ingredientes "salgados"|
# |-------|-------------------|----------------------|
# |indiana|nutella, açúcar de coco e açúcar|polenta, mostarda inglesa, berinjela e macarrão para chow mein|
# |mexicana|sorvete de baunilha, mel, cookies de chocolate e mistura para brownie|cebola roxa, feijão preto, aspargos, peito de frango e tortilha de milho|
# |brasileira|marshmallows, leite condensado, extrato de baunilha, abacate, banana e mel| queijo cheddar, mostarda dijon, peito de frango e dashi |
# |italiana|extrato de baunilha e açúcar|chouriço, massa para conchiglione, pão e batata |
# 
# Observando o caso brasileiro, 
# 
# marshmallows, fresh corn, cheddar cheese, shredded coconut, water, honey, baking soda, dijon mustard, sweet potatoes, chicken breasts, vegetable oil, salt, condensed milk, candy, canola oil, eggs, brown sugar, glutinous rice, white onion, dashi, whole wheat flour, oat flour, flour, boneless skinless chicken breasts, fresh thyme leaves, sprinkles, grated lemon zest, ham, white sugar, avocado, chili flakes, coconut oil, skim milk, pepper, kale, parmesan cheese, unsalted butter, tapioca starch, baking powder, parsley, vanilla extract, cream cheese, coconut milk, chocolate chips, low sodium soy sauce, powdered sugar, sugar, muffin, milk, olive oil, bananas, large eggs, green onions, swiss cheese, butter, all-purpose flour, dark brown sugar, panko breadcrumbs, low-fat milk
# 
# Além disso, todas essas receitas contém uma extensa lista de ingredientes que, combinados, não parecem compor uma receita que possa, de fato, existir. Por exemplo, a primeira receita mexicana, apesar de só levar ingredientes "salgados" apresenta uma combinação de ingredientes que dificilmente correspondem a uma receita real (dentre seus ingredientes estão massa para lasanha, tortilhas, arroz, molho de soja e feijão preto).
# 
# Portanto, essas 6 receitas (incluindo o caso extremo da receita italiana de 65 ingredientes, que é a receita com o maior número de ingredientes presentes no conjunto de dados) serão consideradas outliers e removidas do conjunto de treinamento.

# In[20]:


# lista as receitas com mais de 40 ingredientes
train[train['ingredients_qtt'] > 40][['cuisine','ingredients_text','ingredients_qtt']].values


# ## Quantidade de ingredientes duplicados

# ## Quantidade de receitas duplicadas

# In[21]:


train.duplicated(subset=['cuisine','ingredients_text'], keep='first').sum()


# In[22]:


train[train.duplicated(subset=['cuisine','ingredients_text'], keep=False)].sort_values(['ingredients_text', 'cuisine'])


# São 190 receitas duplicadas no conjunto de dados. Para resolver isso, será mantido somente o primeiro exemplo de cada receita a aparecer no conjunto de treinamento.

# In[23]:


sns.boxplot(x = "ingredients_qtt", y = "cuisine", data = train)


# In[ ]:





# ### Ingredientes mais e menos utilizados nas receitas

# In[24]:


text = " ".join(str(review) for review in train.ingredients_text)
wordcloud = WordCloud(background_color="white", collocations=False).generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[25]:


shortrecipes = train[train['ingredients'].str.len() <= 2]
print("It seems that {} recipes consist of less than or equal to 2 ingredients!".format(len(shortrecipes)))


# In[26]:


shortrecipes


# ### Ingredientes mais utilizados por tipo de cozinha

# In[27]:


train.groupby("cuisine")["ingredients_qtt"].max()


# In[28]:


train[train['cuisine'] == 'brazilian']


# In[29]:


# list out all ingredients
allingredients = []
for item in train['ingredients']:
    for ingr in item:
        allingredients.append(ingr)


# In[30]:


from collections import Counter
count_ingr = Counter(allingredients)
# for ingr in allingredients:
#     count_ingr[ingr]
count_ingr.most_common(20)


# In[31]:


count_ingr = dict(count_ingr)
ingr_count = pd.Series(count_ingr)
fig, ax1 = plt.subplots(figsize=(16, 9))
ingr_count.sort_values(ascending=False)[:10].plot(kind='bar', ax=ax1)
plt.show()


# ## Conclusões da análise exploratória
# 
# * Não será necessário realizar um tratamento com relação aos valores faltantes porque não há casos assim no conjunto de treinamento.
# * As colunas `id` e `ingredients` não serão utilizadas para a criação do modelo. O `id` não será utilizado porque é apenas uma coluna identificadora e única para cada receita e, portanto, não possui poder de classificação e que permita distinguir os exemplos. Já a feature `ingredients` está em um formato um pouco mais díficil de se trabalhar por se tratar de uma lista de strings; em seu lugar, será utilizada a feature `ingredients_text`, que contém todos os ingredientes presentes na lista mas em uma única string, o que facilita o processo de modelagem.
# * Durante a fase de pré-processamento, será necessário realizar uma limpeza no conjunto de dados, visto que há receitas repetidas no conjunto de dados.

# In[ ]:




