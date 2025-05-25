
## Sobre este Repositório

Este repositório foi criado como parte da entrega do **Trabalho de Conclusão da Fase 3** da **Pós-Graduação Machine Learning Engineering Pós Tech - 4MLET**, promovida pela **FIAP**.

O projeto foi desenvolvido com base no **Tech Challenge**, atividade integradora que visa consolidar os conhecimentos adquiridos ao longo da formação, por meio da criação de uma solução completa de **Machine Learning Engineering**. O desafio propõe o desenvolvimento de um sistema que englobe a **coleta automatizada de dados**, o **armazenamento estruturado** dessas informações, a **modelagem preditiva**, o **versionamento e documentação do código** em repositório público e a apresentação de um **modelo funcional**, capaz de alimentar uma aplicação ou dashboard.

Neste contexto, foi concebida a implementação de uma solução para a **previsão de próximos casos de uso em sistemas corporativos**, explorando técnicas de aprendizado supervisionado e arquiteturas de redes neurais aplicadas ao processamento de sequências, com o objetivo de demonstrar a capacidade de integração entre engenharia de dados, modelagem estatística e entrega de valor através de produtos de machine learning.



# Previsão de Próximos Casos de Uso no Sistema Corporativo

## Descrição do Projeto
Este projeto tem como objetivo a construção de um modelo de Machine Learning capaz de **prever os próximos casos de uso** que um usuário do sistema corporativo provavelmente irá executar, com base no **histórico de interações** registrado nos logs do sistema.

A motivação principal é **antecipar comportamentos**, permitindo otimizar processos, melhorar a experiência do usuário e fornecer subsídios para decisões estratégicas, como sugestões automatizadas ou alocação de recursos.

O modelo foi treinado utilizando uma **Rede Neural** com arquitetura baseada em camadas **Dense** e função de ativação **Softmax**, após testes comparativos com arquiteturas de LSTM. O pipeline do projeto envolve desde a extração dos dados diretamente do banco do sistema até o deploy do modelo para utilização prática, que será demonstrada através de um **protótipo simulando o sistema corporativo chamando a API**, visto a impossibilidade de se abrir o sistema real neste trabalho.

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

Após testes, a arquitetura baseada em **camadas Dense com função de ativação Softmax** foi escolhida.

### Tecnologias utilizadas:
- **Python**
- **TensorFlow/Keras**

## Treinamento e Validação
- **Divisão dos dados:** treino, validação e teste.
- **Hiperparâmetros:** épocas, batch size, função de perda categórica, otimização com Adam.
- **Métricas:** Acurácia e Top-K Accuracy.

O modelo demonstrou **bom desempenho** na tarefa de prever o próximo caso de uso.

## Deploy
O modelo foi exportado gerando:
- Um arquivo **`modelo.keras`**.
- Arquivos **`.pkl`** com objetos auxiliares.

Uma **API REST** foi desenvolvida com **FastAPI**, que:
- Recebe uma **sequência de casos de uso** como entrada.
- Retorna os **`n` casos de uso mais prováveis**.

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
├── dados/
│   ├── brutos/             # Dados brutos extraídos (ex: CSV original)
│   └── processados/        # Dados transformados, prontos para modelagem
│
├── modelos/
│   ├── modelo.keras        # Arquivo do modelo treinado
│   └── pre_processador.pkl # Objetos auxiliares, como encoders, scalers
│
├── treinamento/
│   ├── treinar_modelo.py   # Script para treinamento do modelo
│   └── utilitarios.py      # Funções auxiliares de pré-processamento
│
├── inferencia/
│   ├── api/                # Código da API de inferência (FastAPI)
│   │   └── principal.py
│   └── prototipo/          # Aplicação de teste que simula o sistema corporativo
│       └── aplicativo.py
│
├── notebooks/              # Jupyter Notebooks usados na análise e desenvolvimento
│
├── README.md
├── requisitos.txt          # Dependências do projeto
└── .gitignore              # Arquivos e pastas a serem ignorados no Git
```

Cada diretório possui uma finalidade específica, garantindo que o fluxo de trabalho — desde a coleta e transformação dos dados, passando pelo treinamento e até a disponibilização do modelo — seja organizado e facilmente compreendido.
