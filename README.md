# Nanodegree Engenheiro de Machine Learning
## Proposta de projeto final
Vivian Mayumi Yamassaki Pereira  
20 de março de 2019

## Proposta

### Histórico do assunto
Com o crescimento do volume de dados, problemas de envolvendo texto têm aumentado consideravelmente, assim como sua complexidade. Tais problemas vão desde predizer se um dado e-mail é ou não é spam até problemas envolvendo análise de sentimento de textos publicados no Twitter. Como alguns algoritmos de machine learning não permitem utilizar dados de texto, são necessárias manipulações nesses dados para poder realizar tais predições.

Como nunca trabalhei em um projeto de machine learning somente com dados de texto anteriormente, acredito que esse projeto será interessante para aprimorar minhas habilidades nesse domínio. 

### Descrição do problema
O problema a ser resolvido é, a partir de uma lista de ingredientes fornecida, predizer de qual tipo de culinária (brasileira, japonesa, italiana, etc.) é a receita elaborado com tais ingredientes.
Para tanto, será necessário criar um modelo utilizando técnicas de machine learning que realize essa predição com a maior acurácia possível. Como o tipo de culinária descreve uma categoria para a qual a receita pertence, esse problema trata-se de um problema de classificação e, como não muitos tipos de culinárias, é um problema de classificação multiclasse.

Esse problema, denominado _What's cooking?_ é um desafio existente no Kaggle, plataforma com diversos conjuntos de dados e desafios para profissionais e entusiastas em machine learning. Ele possui uma particularidade de possuir dois versões do desafio no Kaggle, ambos com o mesmo conjunto de dados. A única diferença entre essas duas versões é a submissão. A [primeira versão do desafio](https://www.kaggle.com/c/whats-cooking) foi publicado há 3 anos atrás e a submissão é um arquivo contendo as predições do conjunto de teste fornecido. Já a [segunda versão do desafio](https://www.kaggle.com/c/whats-cooking-kernels-only), publicada 6 meses atrás, e que só aceita submissões dos scripts utilizados para criar os modelos e as predições.

Será na primeira versão do desafio que as predições dos modelos criados nesse projeto serão submetidos para suas acurácias serem calculadas. Entretanto, como nessa versão os primeiros colocados no ranking não forneceram o tipo de algoritmo utilizado, será considerado como benchmark o modelo com melhor acurácia fornecido na segunda versão do desafio.

### Conjuntos de dados e entradas
Os dados do desafio _What's cooking?_ foram obtidos [aqui](https://www.kaggle.com/c/whats-cooking/data). O problema possui dois conjuntos de dados: um de treino e outro de teste. O conjunto de treinamento contém 39.774 exemplos e possui 3 variáveis: uma contém o id da receita, outra com a lista de ingredientes e outra com o tipo de culinária do qual a receita pertence, que é a variável resposta:

![Amostra do conjunto de treino](/figuras/exemploTreino.PNG "Amostra do conjunto de treinamento")

Já o conjunto de teste contém 9.944 exemplos e apenas 2 colunas: uma com o id da receita e outra com a lista de ingredientes. Como o tipo de culinária é a variável resposta, ela foi removida do conjunto de teste: 

![Amostra do conjunto de teste](/figuras/exemploTeste.PNG "Amostra do conjunto de teste")

A variável resposta, que é dada pelo tipo de cozinha, pode ser descrita por múltiplos valores. Logo, trata-se de um problema multiclasse.

Além disso, como os exemplos dos conjuntos de dados de treino e teste estão em formato JSON, há um desafio extra em manipular esses dados de modo a se obter uma estrutura que permita treinar e testar o modelo com esses conjuntos. 

### Descrição da solução
A solução para o problema será utilizar alguns modelos de machine learning para serem treinados e testados com os conjuntos de dados fornecidos como entrada. Cada um desses modelos então serão avaliados com base em suas predições para verificar qual foi o modelo que melhor classificou as receitas ao observar seus ingredientes.

Como o conjunto de dados é formado apenas por texto, antes de treinar os modelos de machine learning será necessário realizar um tratamento desses dados, tais como utilizando as técnicas de TF-IDF (para calcular a frequência dos ingredientes nas receitas) ou aplicando o one-hot encoding de modo a converter essas descrições em forma de texto para números. 

Além desse tratamento com relação ao texto, será necessário realizar um pré-processamento antes de começar a modelagem. Por exemplo, será necessário verificar se não há erros de digitação nos ingredientes da receita, se não há outliers, etc.

### Modelo de referência (benchmark)
Quando esta proposta foi criada, o modelo com melhor acurácia postado na [página de scripts](https://www.kaggle.com/c/whats-cooking-kernels-only/kernels) do desafio possuía uma acurácia de 0.82803, que foi criado utilizando uma SVM e cujo script encontra-se [aqui](https://www.kaggle.com/oracool/natty-svc-better-score-than-the-first-place). Portanto, esse modelo e seu resultado que serão utilizados como benchmark.

### Métricas de avaliação
Como o ranking do Kaggle para este problema é dado pela acurácia, esta será a principal métrica de avaliação. A acurácia é dada pela proporção de exemplos que foram corretamente preditos pelo modelo.

Como o desafio do Kaggle não fornece a variável resposta do conjunto de teste, não é possível realizar outras métricas de avaliação com ele, visto que o retorno da plataforma contém apenas a acurácia do modelo calculada para classificá-lo com relação aos demais. Por esse motivo, para que seja possível fazer mais um tipo de avaliação, o modelo com a melhor acurácia obtida nesse projeto será treinado novamente dessa vez somente com uma porção do conjunto de treinamento e a porção restante do treinamento será utilizada como teste (já que é o único meio de obtermos a classe). Para esse novo modelo, será calculada a acurácia novamente (para verificar se diminuiu ou aumentou) e será gerada uma matriz de confusão para verificar quais tipos de cozinha o modelo mais acerta ou erra.

### Design do projeto
Para a resolução desse problema, o seguinte fluxo de trabalho será adotado na seguinte ordem:

**1 - Análise exploratória dos dados**

Primeiramente, como não foram fornecidas informações sobre os dados presentes nos conjuntos de treinamento e teste, será realizada uma análise exploratória para melhor entendimento do problema. Com essa análise, será possível verificar quais cozinhas possuem mais ou menos receitas, as distribuições dos ingredientes por tipo de cozinha, quantidade média de ingredientes por receita, quais ingredientes são específicos para uma determinada cozinha, etc. Além disso, também será possível verificar se existem outliers ou ingredientes escritos erroneamente.

**2 - Feature engineering**

Após a análise exploratória, será necessário realizar uma etapa de feature engineering para fazer o tratamento do conjunto de dados (por exemplo, dos ingredientes que estão com erro de digitação) e exclusão de outliers. Será nessa etapa em que serão aplicadas também técnicas para tratamento do texto, como o TF-IDF e o one-hot encoding.

**3 - Modelagem**

Em seguida, os modelos de machine learning serão treinados com o conjunto de dados resultante da etapa anterior. A princípio, os seguintes modelos foram escolhidos para serem treinados:

- Regressão logística, que é um modelo mais simples e muito utilizado para comparar a performance com outros algoritmos;

- SVM, visto que o benchmark foi criado com esse modelo;

- Random forest, visto que é um modelo que aceita problemas de classificação multiclasse, que é o caso do problema apresentado nesta proposta.

Diversos parâmetros serão testados utilizando a técnica de Gridsearch e os melhores modelos de cada um dos 3 tipos descritos acima serão considerados como aqueles que obtiverem a maior acurácia.

**4 - Avaliação**

Nessa etapa de avaliação, será calculada a acurácia para avaliar e escolher os melhores modelos. Além disso, para o modelo treinado e testado com o conjunto de treinamento dividido, também será criada uma matriz de confusão para que seja possível avaliar o quão bem o modelo está classificando determinadas receitas.

**5 - Comparação e discussão dos modelos**

Após a escolha do melhor modelo, suas predições serão submetidas ao Kaggle e o seu resultado será comparado com o benchmark. Como não temos as classes do conjunto de teste, um novo modelo com as mesmas configurações desse modelo anterior será treinado com uma proporção do conjunto de treinamento e o restante será utilizado para testá-lo, para que seja possível realizar outras análises desse problema. Com todas essas informações, os resultados serão discutidos e serão realizadas comparações entre os modelos criados nesse projeto com o benchmark.