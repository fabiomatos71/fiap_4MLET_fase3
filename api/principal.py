#Para rodar execute:
# estando no diretório raiz do projeto
# source .venv/bin/activate
# uvicorn api.principal:app --reload --host 0.0.0.0 --port 8000

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from fase3_fiap_4mlet.modelo_para_uso import ModeloParaUso

# Inicializa o FastAPI
app = FastAPI(title="API de Previsão de Casos de Uso", version="0.1")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instancia o modelo (por padrão, usa Dense)
modelo = ModeloParaUso(path_base="modelos", use_lstm=False)

# Modelo de dados de entrada
class EntradaPrevisao(BaseModel):
    casoDeUso_1: str
    casoDeUso_2: str
    periodo_mes: str = 'dia_folha'
    top_n: int = 5

# Endpoint de health check
@app.get("/health")
def health_check():
    return {"status": "API funcionando corretamente"}

# Endpoint de previsão
@app.post("/prever")
def prever(entrada: EntradaPrevisao):
    try:
        previsoes = modelo.realizar_previsao(
            casoDeUso_1=entrada.casoDeUso_1,
            casoDeUso_2=entrada.casoDeUso_2,
            periodo_mes=entrada.periodo_mes,
            top_n=entrada.top_n
        )
        # Formata a saída como um dicionário
        resultado = {caso: float(prob) for caso, prob in previsoes}
        return {"previsoes": resultado}
    except Exception as e:
        return {"erro": str(e)}
