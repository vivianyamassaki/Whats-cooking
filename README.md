Esse repositório contém o projeto desenvolvido para resolver o problema de classificação [What's cooking?](https://www.kaggle.com/c/whats-cooking), no qual, a partir de uma lista de ingredientes de uma receita, deve-se predizer o tipo de cozinha da qual a receita pertence. 

O projeto foi desenvolvido utilizando o **Python 3** e as principais bibliotecas utilizadas foram o **pandas** e o **scikit-learn**.

Este repositório contém os seguintes arquivos em sua raiz:
* **Pipfile**: arquivo contendo os pacotes que foram utilizados durante o projeto
*  **proposta.MD**: arquivo contendo a proposta que foi enviada anteriormente
*  **relatorio_final.MD**: relatório final do projeto. Além desse arquivo markdown, foi gerado um arquivo pdf como foi pedido (**relatorio_final.pdf**).

E contém os seguintes diretórios:
* **dados**: contém os dados de treino e teste, separados cada um em sua pasta
	* a pasta _train_ contém o conjunto de treino original e fornecido pelo Kaggle. Não foi possível subir os conjunto de treino após o pré-processamento para o Github por conta do limite de tamanho da plataforma
	* já a pasta _test_ contém o conjunto de teste original e fornecido pelo Kaggle e os arquivos .csv contendo as predições de cada modelo para o conjunto de teste. Foram estes arquivos que foram submetidos ao Kaggle para verificar a acurácia dos modelos
* **notebooks**: contém todos os notebooks criados durante a fase de análise exploratória, pré-processamento dos dados e modelagem 
	* o arquivo **eda.ipynb** contém a análise exploratória realizada no conjunto de dados
	* 	o arquivo **feature_engineering.ipynb** contém as tranformações realizadas no conjunto de dados durante a etapa de pré-processamento
	* Por fim, o diretório **models** contém 4 notebooks para cada um dos modelos feitos durante o projeto (os 3 modelos iniciais mais o modelo dos 3 treinado com outros parâmetros e conjunto de dados)
	* Para todos esses notebooks foram gerados arquivos _.html_ para o caso de ocorrer algum erro durante a leitura do notebook pelo Github, visto que há notebooks que estão extensos
* **figuras**: diretório contendo todas as figuras utilizadas nos relatórios e notebooks
