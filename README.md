
## Tech Challenge Fase 3 - FIAP - 4MLET

Este repositório foi criado como parte da entrega do **Tech Challenge da Fase 3** da **Pós-Graduação em Machine Learning Engineering - Pós Tech - 4MLET**, promovida pela **FIAP**.

O projeto foi desenvolvido com base no **Tech Challenge**, atividade integradora que visa consolidar os conhecimentos adquiridos ao longo da formação, por meio da criação de uma solução completa de **Machine Learning Engineering**. O desafio propõe o desenvolvimento de um sistema que englobe a **coleta e preparação de dados**, o **armazenamento estruturado** dessas informações, a **modelagem preditiva**, o **versionamento e documentação do código** em repositório público e a apresentação de um **modelo funcional**, capaz de alimentar uma aplicação ou dashboard.

Neste contexto, foi concebida a implementação de uma solução para a **previsão de próximos casos de uso em sistemas corporativos**, explorando técnicas de aprendizado supervisionado e arquiteturas de redes neurais aplicadas ao processamento de sequências, com o objetivo de demonstrar a capacidade de integração entre engenharia de dados, modelagem estatística e entrega de valor através de produtos de machine learning.

Como pode ser observado ao longo deste storytelling, inicialmente imaginamos que este seria um problema de **predição de sequencia**.  Contudo, após análises e testes, percebemos que o tamanho reduzido de sequencias utilizadas para o aprendizado acaba por tornando o problema mais de **classificação multiclasse**.  Tal fato nos levou a iniciar os testes implementado um modelo LSTM e concluimos o trabalho com um modelo Dense Layers.  Mantivemos os dois, já que se trata de um trabalho académico.

# Previsão de Próximos Casos de Uso no Sistema Corporativo

## Descrição do Projeto
Este projeto tem como objetivo a construção de um modelo de Machine Learning capaz de **prever os próximos casos de uso** que um usuário do sistema corporativo provavelmente irá executar, com base no **histórico de interações** registrado nos logs do sistema.

A motivação principal é **antecipar comportamentos**, permitindo otimizar processos, melhorar a experiência do usuário e fornecer subsídios para decisões estratégicas, como sugestões automatizadas ou alocação de recursos.

O modelo foi treinado utilizando uma **Rede Neural** com arquitetura baseada em camadas **Dense** e função de ativação **Softmax**, após testes comparativos com arquiteturas de LSTM. O pipeline do projeto envolve desde a extração dos dados diretamente do banco do sistema até o deploy do modelo para utilização prática, que será demonstrada através de um **protótipo simulando o sistema corporativo chamando a API**, visto a impossibilidade de se abrir o sistema real neste trabalho.  O treinamento e utilização do modelo LSTM foram mantidos como opção, apenas para efeito de comparação.

## Arquitetura do Projeto
O projeto foi estruturado em um pipeline de **cinco etapas principais**, integrando as diversas fases do desenvolvimento de soluções baseadas em Machine Learning:

1. **Extração:** Coleta dos dados de logs de uso diretamente do banco de dados do sistema corporativo e exportação para arquivos no formato `.CSV`.
2. **Transformação:** Processamento e limpeza dos dados, com remoção de inconsistências, **anonimização de informações sensíveis** e preparação das sequências de entrada para o modelo.
3. **Modelagem:** Desenvolvimento e treinamento de modelos preditivos baseados em redes neurais. Após testes comparativos, optou-se por uma **arquitetura Dense com função Softmax**.
4. **Validação:** Avaliação do modelo com métricas adequadas, buscando equilíbrio entre desempenho e eficiência.
5. **Deploy:** Disponibilização do modelo treinado por meio de uma **API**, que será consumida por um **protótipo simulando o sistema corporativo**, validando a aplicabilidade prática da solução.

## ETL - Extração, Transformação e Carga

### 1. Extração
Os dados utilizados neste projeto foram extraídos diretamente do **banco de dados transacional** do sistema corporativo, onde são registrados os logs de uso dos usuários.

A extração foi realizada utilizando-se **.NET C#** com o **framework ObjetoRelacional da DevExpress (XPO)**, que é amplamente utilizado corporativamente nos sistemas da empresa. Esta escolha garantiu total integração e compatibilidade com as infraestruturas e práticas já estabelecidas.

Além da extração dos dados brutos, nesta etapa também foram realizadas:
- A **exclusão de casos de uso considerados irrelevantes** para a modelagem.
- A **anonimização dos usuários**, substituindo os identificadores reais por códigos alfanuméricos gerados automaticamente.

As informações extraídas foram:
- **Identificação do usuário** (anonimizada).
- **Data e hora** da execução.
- **Identificador do caso de uso** realizado.

O resultado foi exportado para um arquivo `.CSV`, facilitando o processamento subsequente.

### 2. Transformação
A etapa de transformação foi realizada no ambiente de **Python**, utilizando as bibliotecas **Pandas** e **NumPy**.

As principais ações realizadas foram:
- **Limpeza:** Remoção de registros inconsistentes ou incompletos.
- **Estruturação das Sequências:** Organização dos dados em **sequências temporais** por usuário.

### 3. Carga
Os dados transformados foram carregados no ambiente de desenvolvimento em **Python**, utilizando principalmente as bibliotecas:
- **Pandas**
- **NumPy**
- **Scikit-learn**

## Modelagem
A modelagem do projeto teve como objetivo construir um modelo capaz de prever, a partir de uma sequência de interações anteriores de um usuário com o sistema, qual será o **próximo caso de uso** mais provável.

Foram realizados **testes comparativos** com:
- **LSTM**
- **Rede Neural com Camadas Dense e Softmax**

Também foi considerada a possiblidade de realizar um treino para cada usuário, o que geraria um modelo por usuário. Tal abordagem poderia se mostrar vantajosa, considerando que usuários distintos realizam operações próprias no sistema, de acordo com suas funções.  Porém, alguns testes mostraram que seria melhor gerar um treino único, para todos os usuários. Alguns usuários apresentam uma quantidade muito baixa de amostras.

Os arquivos presentes na pasta treinamento/saidas demonstram alguns testes realizados com a variação de modelo e também uma avaliação da viabilidade de se treinar um modelo para cada usuário.

Após testes, a arquitetura baseada em **camadas Dense com função de ativação Softmax** foi escolhida e treinando-se com as **amostras de todos os usuários**, executando-se apenas alguns com utilizações muito restritas de casos de uso e que não colaboravam com o treinamento.

### Tecnologias utilizadas:
- **Python**
- **TensorFlow/Keras**

## Considerações sobre a Comparação entre LSTM e Dense
Durante o processo de modelagem, realizamos **testes comparativos** entre arquiteturas baseadas em **LSTM (Long Short-Term Memory)** e uma rede **Dense Layers com função de ativação Softmax**. Embora o LSTM seja uma escolha clássica para problemas envolvendo **sequências temporais**, neste contexto específico ele apresentou desempenho inferior. As principais razões identificadas para esse resultado foram:

### 1. Tamanho e complexidade das sequências
- O histórico de interações analisado continha **sequências relativamente curtas (2 casos de uso como histórico para cada previsão)** e com **pouca variabilidade temporal** complexa, características para as quais a arquitetura LSTM pode ser **subaproveitada**.
- Modelos LSTM são mais vantajosos em **contextos onde existe dependência de longo prazo** ou **padrões temporais complexos**, o que não se evidenciou neste dataset.

### 2. Overfitting
- Devido à sua **maior complexidade** e ao número superior de parâmetros, a LSTM demonstrou uma **tendência maior ao overfitting**, principalmente dado o tamanho limitado do conjunto de dados disponível.
- Mesmo com regularização, como dropout e ajustes nos hiperparâmetros, o modelo LSTM apresentou **piora na generalização**. 

### 3. Eficiência e Simplicidade
- A arquitetura com **camadas Dense** foi capaz de capturar os **padrões relevantes** das sequências de uso de forma mais **eficiente e simples**.
- Além disso, apresentou uma **inferência mais rápida** e com **menor custo computacional**, o que é altamente relevante para aplicações práticas que exigem **baixa latência**.

### 4. Natureza do problema
- O problema de previsão do **próximo caso de uso** se comportou de maneira mais próxima de uma **classificação categórica sobre um espaço de estados discretos**, onde modelos como Dense se adequam muito bem, sem a necessidade de um mecanismo explícito de memória de longo prazo como o fornecido pelo LSTM.

**FALAR SOBRE bom desempenho** na tarefa de prever o próximo caso de uso.

## Treinamento e Validação
- **Divisão dos dados:** treino, validação e teste.
- **Hiperparâmetros:** épocas, batch size, função de perda categórica, otimização com Adam.
- **Métricas:** Acurácia e Top-K Accuracy.
  
## Deploy
O modelo foi exportado gerando:
- Um arquivo **`modelo.keras`**.
- Arquivos **`.pkl`** com objetos auxiliares.

Como foram mantidos tanto os modelos LSTM como Dense Layers, houve a exportação para os dois modelos.  O protótipo foi implementado podendo escolher o modelo, para efeito de comparação.

Uma **API REST** foi desenvolvida com **FastAPI**, que:
- Recebe uma **sequência de 2 casos de uso** como entrada.
- Retorna os **`n` casos de uso mais prováveis** como próxima opção do usuário.

A utilização prática foi demonstrada com um **protótipo** que simula o sistema corporativo.

## Como Executar o Projeto

### Requisitos
- **Python 3.x**
- **Pandas**, **NumPy**, **Scikit-learn**, **TensorFlow/Keras**, **FastAPI**, **Uvicorn**

### Passos
1. Clonar o repositório.
2. Configurar o ambiente.
3. Executar a API.
4. Testar com o protótipo.

### Para realizar o treino novamente
1. Clonar o repositório
2. Configurar o ambiente
3. Executar **treinamento/treina_dense_layers.py**.  (Isso atualiza o modelo na pasta **modelos/**)

## Resultados
- **Boa acurácia** nas previsões.
- **Velocidade de inferência adequada**.
- Capacidade de fornecer **múltiplas sugestões** (Top-N).

A validação prática reforçou a **viabilidade de integração** da solução.

## Próximos Passos
- Integração direta com o sistema corporativo.
- Aprimoramento do modelo.
- Ampliação do conjunto de dados.
- Monitoramento contínuo.
- Exploração de **Explainable AI**.

## Autor
Este projeto foi desenvolvido por:

**[Seu Nome Completo]**
- E-mail: [seu.email@empresa.com]
- LinkedIn: [https://www.linkedin.com/in/seu-perfil](https://www.linkedin.com/in/seu-perfil)
- GitHub: [https://github.com/seu-usuario](https://github.com/seu-usuario)



## Estrutura de Diretórios

O projeto está organizado da seguinte forma para garantir clareza, modularidade e facilidade de manutenção:

```
├── .git/                         
├── .gitignore                    
├── .venv/                        # Ambiente virtual Python
├── .vscode/                      
│   ├── launch.json               
│   └── settings.json             
├── README.md                     # Documentação principal do projeto
├── dados/                        # Dados brutos e processados
│   ├── brutos/                   # Dados brutos do sistema
│   │   ├── Dados_TechChallenge_Fase3_bruto.csv # Dados brutos retirados do sistema
│   │   └── ObterLogsSistema.cs   # Código c# que le os logs do sistema e gera dados brutos (já anonimizados)
│   └── processados/              # Dados já tratados para aprendizagem dos modelos
│       └── Dados_TechChallenge_Fase3.csv   # Base de dados principal sobre a qual são geradas as sequencias temporais de casos de uso
├── fase3_fiap_4mlet/             # Pacote principal do projeto (código fonte)
│   ├── __init__.py               
│   ├── __pycache__/              # Cache de bytecode Python
│   ├── modelo_para_uso.py        # Biblioteca para uso do modelo treinado
│   └── treino.py                 # Biblioteca de treinamento dos modelos
├── fase3_fiap_4mlet.egg-info/    # Metadados do pacote Python instalado
├── modelos/                      # Modelos treinados e artefatos
│   ├── modelo-dense.keras        # Modelo DENSE salvo (Keras)
│   ├── modelo-lstm.keras         # Modelo LSTM salvo (Keras)
│   ├── ohe_x-dense.pkl           # OneHotEncoder X para modelo DENSE
│   ├── ohe_x-lstm.pkl            # OneHotEncoder X para modelo LSTM
│   ├── ohe_y-dense.pkl           # OneHotEncoder Y para modelo DENSE
│   ├── ohe_y-lstm.pkl            # OneHotEncoder Y para modelo LSTM
│   ├── scaler_epoca-dense.pkl    # Scaler de época(antes_folha, dia_folha, apos_folha) para DENSE
│   ├── scaler_epoca-lstm.pkl     # Scaler de época(antes_folha, dia_folha, apos_folha) para LSTM
│   ├── scaler_seq-dense.pkl      # Scaler de sequência para DENSE
│   └── scaler_seq-lstm.pkl       # Scaler de sequência para LSTM
├── notebooks/                    # Notebooks Jupyter para análises e experimentos
│   ├── analise_usuarios.ipynb    # Notebook de análise de usuários
│   └── treino_modelos.ipynb      # Notebook de treinamento de modelos
├── requirements.txt              # Lista de dependências do projeto
├── scripts/                      # Scripts utilitários e experimentais
│   ├── compara_modelos_para_uso.py      # Script para comparar modelos
│   ├── saidas/                   # Saídas/resultados de execuções na fase de treinamento
│   │   ├── saida_treina_dense_layers.txt   # Log/output do treino DENSE
│   │   └── saida_treina_lstm.txt           # Log/output do treino LSTM
│   ├── testa_modelo_para_uso.py  # Script para testar modelos em produção
│   ├── treina_dense_layers.py    # Script de treino do modelo DENSE
│   └── treina_lstm.py            # Script de treino do modelo LSTM
├── setup.py                      # Script de instalação do pacote Python
└── teste_venv.py                 # Script de teste do ambiente virtual (após clone do repositório e configuração do ambiente)
```

