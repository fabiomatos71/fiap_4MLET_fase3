from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar arquivos estáticos
app.mount("/static", StaticFiles(directory="inferencia/static"), name="static")

# Configurar templates
templates = Jinja2Templates(directory="inferencia/templates")

# Carregar casos de uso únicos
df = pd.read_csv('dados/processados/Dados_TechChallenge_Fase3.csv', sep=';')
casos_de_uso = sorted(df['casoDeUso'].unique())

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "casos_de_uso": casos_de_uso}
    )
