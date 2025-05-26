import numpy as np
import pandas as pd
import pickle
import os
from tensorflow.keras.models import load_model

class ModeloParaUso:
    def __init__(self, path_base='modelos', use_lstm=False):
        """
        Inicializa a classe, carregando modelo e transformadores.
        Seleciona qual modelo carregar baseado em use_lstm
        """
        self.path_base = path_base
        self.tipo_modelo = "lstm" if use_lstm == True else "dense"
        self.model = load_model(os.path.join(self.path_base, f"modelo-{self.tipo_modelo}.keras"))

        with open(os.path.join(self.path_base,f"scaler_seq-{self.tipo_modelo}.pkl"), "rb") as f:
            self.scaler_seq = pickle.load(f)
        with open(os.path.join(self.path_base,f"scaler_epoca-{self.tipo_modelo}.pkl"), "rb") as f:
            self.scaler_epoca = pickle.load(f)
        with open(os.path.join(self.path_base,f"ohe_x-{self.tipo_modelo}.pkl"), "rb") as f:
            self.ohe_x = pickle.load(f)
        with open(os.path.join(self.path_base,f"ohe_y-{self.tipo_modelo}.pkl"), "rb") as f:
            self.ohe_y = pickle.load(f)

        # Guarda as possíveis classes para uso posterior
        self.nomes_classes = self.ohe_y.categories_[0]

    def realizar_previsao(self, casoDeUso_1, casoDeUso_2, periodo_mes='dia_folha', top_n=5):
        """
        Recebe dois casos de uso (strings) e retorna os 5 casos mais prováveis.
        Args:
            casoDeUso_1: string, caso de uso mais recente
            casoDeUso_2: string, segundo caso de uso mais recente
            periodo_mes: string, pode ser 'antes_folha', 'dia_folha', 'apos_folha'
        Returns:
            List[Tuple(str, float)]: lista dos 5 casos de uso mais prováveis e suas probabilidades
        """
        # 1. Monte o DataFrame de entrada (ajuste os nomes conforme o encoder do seu modelo!)
        entrada = pd.DataFrame([{
            "casoDeUso_1": casoDeUso_1,
            "casoDeUso_2": casoDeUso_2
        }])

        contexto = pd.DataFrame([{"PeriodoDoMes": periodo_mes}])

        # 2. Pré-processamento
        entrada_ohe = pd.DataFrame(
            self.ohe_x.transform(entrada),
            columns=self.ohe_x.get_feature_names_out(["casoDeUso_1", "casoDeUso_2"])
        )
        entrada_scaled = self.scaler_seq.transform(entrada_ohe)

        contexto["PeriodoDoMes"] = contexto["PeriodoDoMes"].map({'antes_folha': 0, 'dia_folha': 1, 'apos_folha': 2})
        contexto_scaled = self.scaler_epoca.transform(contexto)

        # 3. Predição
        pred_proba = self.model.predict([entrada_scaled, contexto_scaled])

        # 4. Top top_n índices e nomes de classes
        top_n_idx = np.argsort(pred_proba[0])[::-1][:top_n]
        top_n_casos = self.nomes_classes[top_n_idx]
        top_n_probs = pred_proba[0][top_n_idx]

        # 5. Retorna lista de tuplas (caso, probabilidade)
        return list(zip(top_n_casos, top_n_probs))


from modelo_para_uso import ModeloParaUso

def realizar_previsao(dado_previsao):
    resultados = modelo.realizar_previsao(casoDeUso_1=dado_previsao["caso_de_uso_1"], casoDeUso_2=dado_previsao["caso_de_uso_2"], periodo_mes=dado_previsao["periodo_mes"], top_n=5)
    print(f"Casos de uso mais prováveis para a sequencia {dado_previsao["caso_de_uso_1"]}, {dado_previsao["caso_de_uso_2"]}, no período {dado_previsao["periodo_mes"]}:")
    for i, (caso, prob) in enumerate(resultados, 1):
        print(f"{i}º: {caso}  (probabilidade: {prob:.2%})")


# Exemplo de uso:
if __name__ == "__main__":
    modelo = ModeloParaUso(path_base=f"{os.getcwd()}/modelos", use_lstm=False)

    print("\nPrevisões utilizando modelo Dense Layers\n")
    realizar_previsao({"caso_de_uso_1": "uc2036", "caso_de_uso_2": "uc2062", "periodo_mes": "dia_folha"})

    realizar_previsao({"caso_de_uso_1": "uc0222", "caso_de_uso_2": "uc0181", "periodo_mes": "dia_folha"})

    realizar_previsao({"caso_de_uso_1": "uc0222", "caso_de_uso_2": "uc0181", "periodo_mes": "antes_folha"})

    realizar_previsao({"caso_de_uso_1": "uc0222", "caso_de_uso_2": "uc0181", "periodo_mes": "apos_folha"})


    modelo = ModeloParaUso(path_base=f"{os.getcwd()}/modelos", use_lstm=True)

    print("\nPrevisões utilizando modelo LSTM\n")
    realizar_previsao({"caso_de_uso_1": "uc2036", "caso_de_uso_2": "uc2062", "periodo_mes": "dia_folha"})

    realizar_previsao({"caso_de_uso_1": "uc0222", "caso_de_uso_2": "uc0181", "periodo_mes": "dia_folha"})

    realizar_previsao({"caso_de_uso_1": "uc0222", "caso_de_uso_2": "uc0181", "periodo_mes": "antes_folha"})

    realizar_previsao({"caso_de_uso_1": "uc0222", "caso_de_uso_2": "uc0181", "periodo_mes": "apos_folha"})

    