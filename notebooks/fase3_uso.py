from modelo_para_uso import ModeloParaUso

def realizar_previsao(dado_previsao):
    resultados = modelo.realizar_previsao(casoDeUso_1=dado_previsao["caso_de_uso_1"], casoDeUso_2=dado_previsao["caso_de_uso_2"], periodo_mes=dado_previsao["periodo_mes"], top_n=5)
    print(f"Casos de uso mais prováveis para a sequencia {dado_previsao["caso_de_uso_1"]}, {dado_previsao["caso_de_uso_2"]}, no período {dado_previsao["periodo_mes"]}:")
    for i, (caso, prob) in enumerate(resultados, 1):
        print(f"{i}º: {caso}  (probabilidade: {prob:.2%})")



if __name__ == "__main__":
    modelo = ModeloParaUso(path_base="artefatos_modelo")


    realizar_previsao({"caso_de_uso_1": "uc0222", "caso_de_uso_2": "uc0181", "periodo_mes": "dia_folha"})

    realizar_previsao({"caso_de_uso_1": "uc0222", "caso_de_uso_2": "uc0181", "periodo_mes": "antes_folha"})

    realizar_previsao({"caso_de_uso_1": "uc0222", "caso_de_uso_2": "uc0181", "periodo_mes": "apos_folha"})
