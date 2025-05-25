import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

# Simulação de dados de opções de menu
menu_options = ['Home', 'Perfil', 'Configurações', 'Suporte', 'Relatórios', 'Mensagens']
data = []

# Gerando dados simulados
np.random.seed(42)
for _ in range(5000):
    seq = np.random.choice(menu_options, 4, replace=True)
    data.append(seq)

# Convertendo para DataFrame
df = pd.DataFrame(data, columns=['Menu_1', 'Menu_2', 'Menu_3', 'Next_Menu'])

# Codificação das opções de menu
menu_to_int = {menu: idx for idx, menu in enumerate(menu_options)}
int_to_menu = {idx: menu for menu, idx in menu_to_int.items()}

for col in ['Menu_1', 'Menu_2', 'Menu_3', 'Next_Menu']:
    df[col] = df[col].map(menu_to_int)

# Separando X e y
X = df[['Menu_1', 'Menu_2', 'Menu_3']].values
y = df['Next_Menu'].values

# Dividindo em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo de Cadeia de Markov
class MarkovModel:
    def __init__(self):
        self.transitions = {}

    def fit(self, X, y):
        for seq, target in zip(X, y):
            key = tuple(seq)
            if key not in self.transitions:
                self.transitions[key] = {}
            if target not in self.transitions[key]:
                self.transitions[key][target] = 0
            self.transitions[key][target] += 1

    def predict(self, X):
        predictions = []
        for seq in X:
            key = tuple(seq)
            if key in self.transitions:
                predictions.append(max(self.transitions[key], key=self.transitions[key].get))
            else:
                predictions.append(np.random.choice(list(menu_to_int.values())))
        return np.array(predictions)

# Treinando o modelo de Markov
markov_model = MarkovModel()
markov_model.fit(X_train, y_train)
y_pred_markov = markov_model.predict(X_test)

# Avaliação do modelo de Markov
markov_accuracy = accuracy_score(y_test, y_pred_markov)

# Configurando o modelo LSTM
model = Sequential()
model.add(Embedding(len(menu_options), 8, input_length=3))
model.add(LSTM(32, activation='relu'))
model.add(Dense(len(menu_options), activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinando o LSTM
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2, verbose=1)

# Avaliando o LSTM
y_pred_lstm = np.argmax(model.predict(X_test), axis=-1)
lstm_accuracy = accuracy_score(y_test, y_pred_lstm)

print(f"Acurácia Markov: {markov_accuracy * 100:.2f}%")
print(f"Acurácia LSTM: {lstm_accuracy * 100:.2f}%")
