import os
from modelo_para_uso import ModeloParaUso

def comparar_modelos(dados_previsao, top_n=5):
    # Instancie ambos os modelos apenas uma vez
    modelo_dense = ModeloParaUso(path_base=f"{os.getcwd()}/modelos", use_lstm=False)
    modelo_lstm = ModeloParaUso(path_base=f"{os.getcwd()}/modelos", use_lstm=True)

    for idx, dado in enumerate(dados_previsao, 1):
        print(f"\n========== COMPARAÇÃO {idx} ==========")
        print(f"Entrada: CasoDeUso_1 = {dado['caso_de_uso_1']}, CasoDeUso_2 = {dado['caso_de_uso_2']}, Período = {dado['periodo_mes']}")

        # Previsão modelo Dense
        preds_dense = modelo_dense.realizar_previsao(
            casoDeUso_1=dado["caso_de_uso_1"],
            casoDeUso_2=dado["caso_de_uso_2"],
            periodo_mes=dado["periodo_mes"],
            top_n=top_n
        )
        # Previsão modelo LSTM
        preds_lstm = modelo_lstm.realizar_previsao(
            casoDeUso_1=dado["caso_de_uso_1"],
            casoDeUso_2=dado["caso_de_uso_2"],
            periodo_mes=dado["periodo_mes"],
            top_n=top_n
        )

        # Printando lado a lado
        print(f"\n{'Rank':<5} {'Dense Model':<35} {'Probabilidade':<17} || {'LSTM Model':<35} {'Probabilidade'}")
        print("-"*100)
        for i in range(top_n):
            caso_dense, prob_dense = preds_dense[i]
            caso_lstm, prob_lstm = preds_lstm[i]
            print(f"{i+1:<5} {caso_dense:<35} {prob_dense:.2%}         || {caso_lstm:<35} {prob_lstm:.2%}")
        print("="*100)

if __name__ == "__main__":
    # Liste todos os dados de previsão que quiser comparar
    dados_previsao = [
        {"caso_de_uso_1": "uc2036", "caso_de_uso_2": "uc2062", "periodo_mes": "dia_folha"},
        {"caso_de_uso_1": "uc0222", "caso_de_uso_2": "uc0181", "periodo_mes": "dia_folha"},
        {"caso_de_uso_1": "uc0222", "caso_de_uso_2": "uc0181", "periodo_mes": "antes_folha"},
        {"caso_de_uso_1": "uc0222", "caso_de_uso_2": "uc0181", "periodo_mes": "apos_folha"},
    ]
    comparar_modelos(dados_previsao, top_n=5)
