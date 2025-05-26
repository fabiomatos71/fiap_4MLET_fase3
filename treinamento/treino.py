import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate, Dropout, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.optimizers import Adam
import pickle
import warnings
warnings.filterwarnings('ignore')

def filtra_usuario_separa_x_epoca_y(df, username="*", usuarios_exclusao=[]):
    """
    Prepara os dados filtrando por usuário e criando features históricas

    Separa as propriedades históricas para análise da sequencia: x, 
    
    Args:
        df: DataFrame com os dados originais
        username: Nome do usuário para filtrar ("*" para todos os usuários)
    
    Returns:
        tuple: (features_sequenciais, features_contextuais, target)
    """
    print(f"Processando dados para usuário: {username}")
    
    # Filtrar por usuário se especificado

    if username != "*":
        df_filtro = df[df["usuario"] == username].copy()
        print(f"Registros após filtro de usuário: {len(df_filtro)}")
    else:
        df_filtro = df[~df["usuario"].isin(usuarios_exclusao)].copy()
        print(f"Processando todos os usuários (excluindo {len(usuarios_exclusao)} usuários): {len(df_filtro)} registros")

        # Ordenar por data/hora para manter sequência temporal
        df_filtro = df_filtro.sort_values(["usuario", "Dia", "Mes", "Ano", "DataHoraCriacao"])

    # Criar features históricas (últimos 3 casos de uso)
    print("Criando features históricas...")
    for shift in range(1, 3):
        df_filtro[f"casoDeUso_{shift}"] = df_filtro.groupby(["usuario", "Dia", "Mes", "Ano"])["casoDeUso"].shift(shift).fillna("vazio")

    # Remover registros com NaN e resetar índice
    df_filtro = df_filtro.dropna().reset_index(drop=True)
    print(f"Registros após limpeza: {len(df_filtro)}")

    # Separar target
    df_target = df_filtro["casoDeUso"]

    # Preparar features sequenciais (remover colunas não necessárias)
    cols_a_remover = ["DataHoraCriacao", "Dia", "Mes", "Ano", "casoDeUso", "usuario", "PeriodoDoMes"]
    df_x = df_filtro.drop(columns=cols_a_remover)

    # Preparar features contextuais (período do mês)
    df_epoca = df_filtro[["PeriodoDoMes"]].copy()

    return df_x, df_epoca, df_target

def aplica_OneHotEncoder_x(df, cols):
    """
    Aplica One-Hot Encoding nas colunas especificadas
    
    Args:
        df: DataFrame para processar
        cols: Lista de colunas para aplicar encoding
    
    Returns:
        tuple: (DataFrame com encoding aplicado, OneHotEncoder fitted)
    """
    print(f"Aplicando One-Hot Encoding em: {cols}")
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    df_ohe = pd.DataFrame(
        ohe.fit_transform(df[cols]), 
        columns=ohe.get_feature_names_out(cols), 
        index=df.index
    )
    return pd.concat([df.drop(columns=cols), df_ohe], axis=1), ohe

def preprocessar_dados(data_path, username="*", usuarios_exclusao=[], test_size=0.4, random_state=42):
    """
    Pipeline completo de preprocessamento de dados
    
    Args:
        data_path: Caminho para o arquivo CSV
        username: Usuário para filtrar
        test_size: Proporção para teste
        random_state: Seed para reprodutibilidade
    
    Returns:
        tuple: Dados de treino e teste preprocessados
    """
    print("=== INICIANDO PREPROCESSAMENTO ===")
    
    # Carregar dados
    print(f"Carregando dados de: {data_path}")
    df = pd.read_csv(
        data_path, 
        sep=';', 
        encoding='utf-8', 
        parse_dates=['DataHoraCriacao'], 
        dayfirst=True
    )
    print(f"Dataset carregado: {df.shape}")
    
    # Verificar distribuição das classes
    print("\nDistribuição das classes:")
    print(df['casoDeUso'].value_counts().head(10))
    
    # Processar dados
    df_x, df_epoca, serie_y = filtra_usuario_separa_x_epoca_y(df, username, usuarios_exclusao)
    
    # Aplicar One-Hot Encoding nas features históricas
    # df_x, ohe_x = aplica_OneHotEncoder_x(df_x, ["casoDeUso_1", "casoDeUso_2", "casoDeUso_3"])
    df_x, ohe_x = aplica_OneHotEncoder_x(df_x, ["casoDeUso_1", "casoDeUso_2"])
    
    # Mapear período do mês para valores numéricos e normalizar
    print("Processando features contextuais...")
    df_epoca["PeriodoDoMes"] = df_epoca["PeriodoDoMes"].map({
        'antes_folha': 0, 
        'dia_folha': 1, 
        'apos_folha': 2
    })
    
    # Normalizar features contextuais
    scaler_epoca = StandardScaler()
    df_epoca_scaled = pd.DataFrame(
        scaler_epoca.fit_transform(df_epoca), 
        columns=df_epoca.columns,
        index=df_epoca.index
    )
    
    # Converter target para One-Hot
    y_one_hot, ohe_y = aplica_OneHotEncoder_x(serie_y.to_frame(), ["casoDeUso"])
    
    # Normalizar features sequenciais
    print("Normalizando features sequenciais...")
    scaler_seq = StandardScaler()
    df_x_scaled = pd.DataFrame(
        scaler_seq.fit_transform(df_x),
        columns=df_x.columns,
        index=df_x.index
    )
    
    # Verificar se é possível fazer divisão estratificada
    print("Dividindo dados em treino e teste...")
    
    # Verificar classes com poucas amostras
    class_counts = serie_y.value_counts()
    classes_com_poucas_amostras = class_counts[class_counts < 2]
    
    if len(classes_com_poucas_amostras) > 0:
        print(f"⚠️ Aviso: {len(classes_com_poucas_amostras)} classes com apenas 1 amostra:")
        print(classes_com_poucas_amostras.head())
        print("Removendo classes com poucas amostras para permitir estratificação...")
        
        # Filtrar classes com pelo menos 2 amostras
        classes_validas = class_counts[class_counts >= 2].index
        mask = serie_y.isin(classes_validas)
        
        df_x_scaled = df_x_scaled[mask]
        df_epoca_scaled = df_epoca_scaled[mask]
        y_one_hot = y_one_hot[mask]
        serie_y = serie_y[mask]
        
        print(f"Dados após filtro: {len(df_x_scaled)} amostras, {len(serie_y.unique())} classes")
        
        # Recriar one-hot encoding para classes restantes
        y_one_hot, ohe_y = aplica_OneHotEncoder_x(serie_y.to_frame(), ["casoDeUso"])
    
    # Tentar divisão estratificada, se falhar usar divisão simples
    try:
        X_train_seq, X_test_seq, X_train_epoch, X_test_epoch, y_train, y_test = train_test_split(
            df_x_scaled, df_epoca_scaled, y_one_hot, 
            test_size=test_size, 
            random_state=random_state,
            stratify=serie_y  # Estratificação para manter proporção das classes
        )
        print("✅ Divisão estratificada realizada com sucesso")
    except ValueError as e:
        print(f"⚠️ Não foi possível fazer divisão estratificada: {str(e)}")
        print("Realizando divisão simples...")
        X_train_seq, X_test_seq, X_train_epoch, X_test_epoch, y_train, y_test = train_test_split(
            df_x_scaled, df_epoca_scaled, y_one_hot, 
            test_size=test_size, 
            random_state=random_state
        )
    
    print(f"Treino: {X_train_seq.shape[0]} samples")
    print(f"Teste: {X_test_seq.shape[0]} samples")
    
    return (X_train_seq, X_test_seq, X_train_epoch, X_test_epoch, 
            y_train, y_test, scaler_seq, scaler_epoca, ohe_x, ohe_y)

def criar_modelo_hibrido(input_seq_shape, input_epoch_shape, output_shape, 
                        use_lstm=True, lstm_units=64, dense_units=[128, 64], 
                        dropout_rate=0.3):
    """
    Cria modelo híbrido com opção de usar LSTM ou Dense Layers

    Args:
        input_seq_shape: Shape das features sequenciais
        input_epoch_shape: Shape das features contextuais
        output_shape: Número de classes de saída
        use_lstm: Se True, usa LSTM; se False, usa Dense
        lstm_units: Unidades LSTM
        dense_units: Lista com unidades das camadas Dense
        dropout_rate: Taxa de dropout
    
    Returns:
        Model: Modelo compilado
    """
    print(f"=== CRIANDO MODELO {'LSTM' if use_lstm else 'DENSE'} HÍBRIDO ===")
    
    # Input para features sequenciais
    input_seq = Input(shape=(input_seq_shape,), name='input_sequence')
    input_epoch = Input(shape=(input_epoch_shape,), name='input_epoch')
    
    if use_lstm:
        # Para LSTM, precisamos reshapear para 3D (samples, timesteps, features)
        # Assumindo que cada feature é um timestep
        x = Reshape((input_seq_shape, 1))(input_seq)
        x = LSTM(lstm_units, return_sequences=False, dropout=dropout_rate)(x)
        print(f"Usando LSTM com {lstm_units} unidades")
    else:
        # Usar camadas Dense tradicionais
        x = input_seq
        for i, units in enumerate(dense_units):
            x = Dense(units, activation='relu', name=f'dense_{i+1}')(x)
            x = Dropout(dropout_rate, name=f'dropout_{i+1}')(x)
        print(f"Usando Dense layers: {dense_units}")
    
    # Combinar features sequenciais com contextuais
    combined = Concatenate(name='concatenate')([x, input_epoch])
    
    # Camada de saída
    output = Dense(output_shape, activation='softmax', name='output')(combined)
    
    # Criar modelo
    model = Model(inputs=[input_seq, input_epoch], outputs=output)
    
    # Compilar modelo
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer, 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    print("Modelo criado e compilado!")
    print(f"Parâmetros totais: {model.count_params():,}")
    
    return model

def treinar_modelo(model, X_train_seq, X_train_epoch, y_train, 
                  epochs=50, validation_split=0.2, verbose=1):
    """
    Treina o modelo com callbacks para early stopping e redução de learning rate
    
    Args:
        model: Modelo a ser treinado
        X_train_seq, X_train_epoch: Features de treino
        y_train: Target de treino
        epochs: Número máximo de épocas
        validation_split: Proporção para validação
        verbose: Verbosidade do treinamento
    
    Returns:
        History: Histórico do treinamento
    """
    print("=== INICIANDO TREINAMENTO ===")
    
    # Callbacks para melhorar o treinamento
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Treinar modelo
    history = model.fit(
        [X_train_seq, X_train_epoch], 
        y_train,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=verbose,
        batch_size=32
    )
    
    print("Treinamento concluído!")
    return history

def avaliar_modelo(model, X_test_seq, X_test_epoch, y_test, ohe_y):
    """
    Avalia o modelo com múltiplas métricas
    
    Args:
        model: Modelo treinado
        X_test_seq, X_test_epoch: Features de teste
        y_test: Target de teste
        ohe_y: OneHotEncoder do target para obter nomes das classes
    
    Returns:
        dict: Dicionário com métricas de avaliação
    """
    print("=== AVALIANDO MODELO ===")
    
    # Predições
    y_pred_proba = model.predict([X_test_seq, X_test_epoch])
    y_pred = np.argmax(y_pred_proba, axis=-1)
    y_test_labels = np.argmax(y_test.values, axis=-1)
    
    # Métricas básicas
    accuracy = accuracy_score(y_test_labels, y_pred)
    print(f"Acurácia: {accuracy * 100:.2f}%")
    
    # Relatório de classificação
    class_names = ohe_y.get_feature_names_out(['casoDeUso'])
    class_names = [name.replace('casoDeUso_', '') for name in class_names]
    
    print("\n=== RELATÓRIO DE CLASSIFICAÇÃO ===")
    print(classification_report(y_test_labels, y_pred, target_names=class_names))
    
    # Matriz de confusão
    cm = confusion_matrix(y_test_labels, y_pred)
    
    return {
        'accuracy': accuracy,
        'y_pred': y_pred,
        'y_test': y_test_labels,
        'confusion_matrix': cm,
        'class_names': class_names,
        'y_pred_proba': y_pred_proba
    }

def salvar_modelo_parametros(modelo, scaler_seq, scaler_epoca, ohe_x, ohe_y, path_base='modelos', tipo_modelo=""):
    """
    Salva o modelo Keras e os transformadores necessários para inferência futura.
    
    Args:
        model: Modelo Keras treinado.
        scaler_seq: Scaler das features sequenciais.
        scaler_epoca: Scaler das features contextuais.
        ohe_x: OneHotEncoder das features históricas.
        ohe_y: OneHotEncoder do target.
        path_base: Pasta/caminho base para salvar os arquivos.
    """
    import os
    os.makedirs(path_base, exist_ok=True)

    # Salva o modelo keras
    modelo.save(os.path.join(path_base, f'modelo-{tipo_modelo}.keras'))

    # Salva os transformadores com pickle
    with open(os.path.join(path_base, f'scaler_seq-{tipo_modelo}.pkl'), 'wb') as f:
        pickle.dump(scaler_seq, f)

    with open(os.path.join(path_base, f'scaler_epoca-{tipo_modelo}.pkl'), 'wb') as f:
        pickle.dump(scaler_epoca, f)

    with open(os.path.join(path_base, f'ohe_x-{tipo_modelo}.pkl'), 'wb') as f:
        pickle.dump(ohe_x, f)

    with open(os.path.join(path_base, f'ohe_y-{tipo_modelo}.pkl'), 'wb') as f:
        pickle.dump(ohe_y, f)

    print(f"\n✅ Modelo e transformadores para o modelo {tipo_modelo} salvos na pasta '{path_base}'!")

def plotar_resultados(history, results):
    """
    Plota gráficos de treinamento e matriz de confusão
    
    Args:
        history: Histórico do treinamento
        results: Resultados da avaliação
    """
    print("Gerando visualizações...")
    
    # Configurar estilo dos gráficos
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Acurácia durante treinamento
    ax1 = plt.subplot(2, 3, 1)
    plt.plot(history.history['accuracy'], label='Treino', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validação', linewidth=2)
    plt.title('Acurácia Durante o Treinamento', fontsize=14, fontweight='bold')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Perda durante treinamento
    ax2 = plt.subplot(2, 3, 2)
    plt.plot(history.history['loss'], label='Treino', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validação', linewidth=2)
    plt.title('Perda Durante o Treinamento', fontsize=14, fontweight='bold')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Learning Rate (se disponível)
    ax3 = plt.subplot(2, 3, 3)
    if 'lr' in history.history:
        plt.plot(history.history['lr'], linewidth=2, color='red')
        plt.title('Learning Rate', fontsize=14, fontweight='bold')
        plt.xlabel('Épocas')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Learning Rate\nnão disponível', 
                ha='center', va='center', fontsize=12)
        plt.title('Learning Rate', fontsize=14, fontweight='bold')
    
    # 4. Matriz de confusão
    ax4 = plt.subplot(2, 3, (4, 6))
    sns.heatmap(
        results['confusion_matrix'], 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=results['class_names'],
        yticklabels=results['class_names']
    )
    plt.title('Matriz de Confusão', fontsize=14, fontweight='bold')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    
    # 5. Distribuição de confiança das predições
    ax5 = plt.subplot(2, 3, 5)
    max_proba = np.max(results['y_pred_proba'], axis=1)
    plt.hist(max_proba, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribuição de Confiança das Predições', fontsize=14, fontweight='bold')
    plt.xlabel('Confiança Máxima')
    plt.ylabel('Frequência')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def processar_pipeline(data_path='dados.csv', 
         usuario='*', # Nome do usuário ou '*' para todos
         use_lstm=False,  # Usar Dense layers ou LSTM
         epochs=50,
         plotar_resultado=True, # Se deve ou não chamar o método plotar_resultados()
         usuarios_exclusao=[], # No caso de usar '*' em usuario, lista de exclusão da base de dados.  Não carregar os dados desses.
         salvar_modelo=False, # Salva o modelo para posterior utilização nas previsões
         modelo_path=''
         ):  
    """
    Função principal que executa todo o pipeline
    
    Args:
        data_path: Caminho para os dados
        usuario: Usuário para filtrar
        use_lstm: Se usar LSTM ou Dense layers
        epochs: Número de épocas para treinamento
        plotar_resultado: Se deve ou não chamar o método plotar_resultados() ao final do treino e testes
        usuarios_exclusao: Lista de usuários a não serem considerados se o usuario='*'
        salvar_modelo: Salva o modelo para posterior utilização nas previsões
        modelo_path: pasta onde deverão ser salvos os dados do modelo
    """
    try:
        # Preprocessamento
        (X_train_seq, X_test_seq, X_train_epoch, X_test_epoch, 
         y_train, y_test, scaler_seq, scaler_epoca, ohe_x, ohe_y) = preprocessar_dados(data_path, usuario, usuarios_exclusao)
        
        # Criar modelo
        modelo = criar_modelo_hibrido(
            input_seq_shape=X_train_seq.shape[1],
            input_epoch_shape=X_train_epoch.shape[1],
            output_shape=y_train.shape[1],
            use_lstm=use_lstm
        )
        
        print("\n\n=== ARQUITETURA DO MODELO ===")
        modelo.summary()
        
        # Treinar modelo
        historico = treinar_modelo(
            modelo, X_train_seq, X_train_epoch, y_train, 
            epochs=epochs
        )
        
        # Avaliar modelo
        resultados = avaliar_modelo(
            modelo, X_test_seq, X_test_epoch, y_test, ohe_y
        )
        
        # Plotar resultados
        if plotar_resultado:
            plotar_resultados(historico, resultados)
        
        if salvar_modelo:
            salvar_modelo_parametros(modelo, scaler_seq, scaler_epoca, ohe_x, ohe_y, path_base=modelo_path, tipo_modelo="lstm" if use_lstm == True else "dense")

        return modelo, historico, resultados
        
    except Exception as e:
        print(f"Erro durante execução: {str(e)}")
        raise
