from treino import processar_pipeline

print("\nIniciando pipeline de Machine Learning - modelo Dense Layers ")
print("=" * 60)

# Executar todo o pipeline principal
modelo, historico, resultados = processar_pipeline(
    data_path='dados/processados/Dados_TechChallenge_Fase3.csv', 
    usuario='*',  
    usuarios_exclusao=["usuario_00", "usuario_01", "usuario_02", "usuario_04", "usuario_06", "usuario_08", "usuario_11", "usuario_12", "usuario_13"],
    use_lstm=False,  
    epochs=50,
    plotar_resultado=True  
)

print("\n\n\nPipeline concluído com sucesso!")
print(f"Acurácia final: {resultados['accuracy']*100:.2f}%")