# Classificação da dieta de dinossauros com MLP

# Autores: Guilherme Altmeyer Soares; Igor Correa Domingues de Almeida; Kauan Carlos Campos; Maria Eduarda Quevedo

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import joblib
# ======================
# Configuração do Pandas para exibição
# ======================
# Garante que todas as linhas sejam mostradas
pd.set_option('display.max_rows', None)
# Garante que todas as colunas sejam mostradas
pd.set_option('display.max_columns', None)
# Ajusta a largura da exibição para evitar quebras de linha indesejadas
pd.set_option('display.width', None)


# Carregar o dataset
data = pd.read_csv("./dinoDatasetCSV.csv")

# Seleção de features e alvo
features = ["length_m", "weight_kg", "height_m",
            "locomotion", "geological_period", "lived_in"]
target = "diet"

# Remover linhas com valores nulos nas colunas selecionadas
data = data.dropna(subset=features + [target])

X = data[features]
y = data[target]


# Pré-processamento
num_features = ["length_m", "weight_kg", "height_m"]
cat_features = ["locomotion", "geological_period", "lived_in"]

# Pré-processador: escala nos numéricos + one-hot nos categóricos
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)

# Aplicar transformação nos atributos
X_processed = preprocessor.fit_transform(X)

joblib.dump(preprocessor, "dino_preprocessor.joblib")
print("\nPreprocessor salvo")

# Codificação da saída (diet)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_encoded = to_categorical(y_encoded)

joblib.dump(label_encoder, "dino_label_encoder.joblib")
print("Label encoder salvo com sucesso!")

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_encoded, test_size=0.2, random_state=42
)


# Construção da MLP
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(y_encoded.shape[1], activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# Treinamento
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16,
    verbose=1
)


# Avaliação
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nAcurácia no conjunto de teste: {acc:.2f}")

# Gráficos de Treinamento
plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Treino")
plt.plot(history.history["val_loss"], label="Validação")
plt.title("Evolução da Loss")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Treino")
plt.plot(history.history["val_accuracy"], label="Validação")
plt.title("Evolução da Acurácia")
plt.xlabel("Épocas")
plt.ylabel("Acurácia")
plt.legend()

plt.tight_layout()
plt.show()


# Matriz de Confusão e Relatório de Classificação
# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Nomes das classes (dietas)
class_names = label_encoder.classes_

# Gerar o relatório de classificação
print("\nRelatório de Classificação:")

print(classification_report(
    y_true,
    y_pred_classes,
    target_names=class_names,
    labels=np.arange(len(class_names))  
))

# Gerar a matriz de confusão
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Plotar a matriz de confusão para melhor visualização
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusão')
plt.ylabel('Classe Verdadeira')
plt.xlabel('Classe Prevista')
plt.show()


# Salvar o Modelo
model.save("modelo_dieta_dinossauros.h5")
print("\nModelo salvo com sucesso!")



# Previsão com Novos Dados

# Criar um DataFrame com um novo dinossauro (exemplo)
novo_dino = pd.DataFrame({
    "length_m": [12.0],
    "weight_kg": [7000.0],
    "height_m": [5.0],
    "locomotion": ["Bipedal"],
    "geological_period": ["Cretaceous"],
    "lived_in": ["North America"]
})


novo_dino_processado = preprocessor.transform(novo_dino)

# Fazer a previsão
previsao = model.predict(novo_dino_processado)
classe_prevista_idx = np.argmax(previsao, axis=1)[0]
dieta_prevista = label_encoder.inverse_transform([classe_prevista_idx])[0]

print(f"\nA dieta prevista para o novo dinossauro é: {dieta_prevista}")
print(f"Confiança da previsão: {previsao[0]}")
