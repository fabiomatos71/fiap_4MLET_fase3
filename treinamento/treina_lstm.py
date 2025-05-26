import os
from treino import processar_pipeline

print("\nIniciando pipeline de Machine Learning - modelo LSTM ")
print("=" * 60)

path_base = os.getcwd()
print(f"\nRodando em {path_base}.\n")

# Executar todo o pipeline principal
modelo, historico, resultados = processar_pipeline(
    data_path=f'{path_base}/dados/processados/Dados_TechChallenge_Fase3.csv', 
    usuario='*',  
    usuarios_exclusao=["usuario_02"],
    use_lstm=True,  
    epochs=50,
    plotar_resultado=True,
    salvar_modelo=True,
    modelo_path=f'{path_base}/modelos'
)

print("\n\n\nPipeline concluído com sucesso!")
print(f"Acurácia final: {resultados['accuracy']*100:.2f}%")