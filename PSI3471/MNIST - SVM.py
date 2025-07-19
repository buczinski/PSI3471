import numpy as np
from scipy.ndimage import shift
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
from tensorflow.keras.datasets import mnist

def augment_data(images, labels, shifts):

    augmented_images = list(images)
    augmented_labels = list(labels)
    
    print(f"Iniciando Data Augmentation para {len(shifts)} direções...")
    
    for dx, dy in shifts:
        for image, label in zip(images, labels):
            shifted_image = shift(image, shift=(dx, dy), cval=0)
            augmented_images.append(shifted_image)
            augmented_labels.append(label)
            
    return np.array(augmented_images), np.array(augmented_labels)

# --- 1. Carregamento e Preparação dos Dados ---
print("Carregando o dataset MNIST...")
(X_train_orig, y_train), (X_test_orig, y_test) = mnist.load_data()

# Normaliza os pixels para o intervalo [0, 1]
X_train_normalized = X_train_orig / 255.0
X_test_normalized = X_test_orig / 255.0

# --- 2. Data Augmentation (Menor) ---
# Para o SVM, usamos um conjunto menor para manter o tempo de treino razoável.
# Conjunto final 2x maior (1 original + 1 deslocado para a direita).
shifts_to_apply_svm = [(0, 1)] # Apenas um deslocamento
X_train_augmented, y_train_augmented = augment_data(X_train_normalized, y_train, shifts_to_apply_svm)

print(f"Tamanho do conjunto de treino original: {len(X_train_orig)} imagens")
print(f"Tamanho do conjunto de treino aumentado para SVM: {len(X_train_augmented)} imagens")

# Achata as imagens para vetores de 784 pixels
n_samples_train = len(X_train_augmented)
n_samples_test = len(X_test_normalized)
X_train_flat = X_train_augmented.reshape((n_samples_train, -1))
X_test_flat = X_test_normalized.reshape((n_samples_test, -1))


# --- 3. Treinamento e Avaliação do Modelo SVM ---
print("\n--- Iniciando Parte 2: Classificador SVM ---")
# Cria o classificador SVM
# C=5: Parâmetro de regularização. Um valor mais alto tenta classificar corretamente mais exemplos de treino.
# kernel='rbf': Kernel não-linear, muito poderoso para dados de imagem.
# gamma='scale': Estratégia padrão e eficaz para o coeficiente do kernel.
svm_clf = SVC(kernel='rbf', C=5, gamma='scale')

# Mede o tempo de processamento
start_time = time.time()

# Treina o modelo com os dados aumentados
svm_clf.fit(X_train_flat, y_train_augmented)

# Faz predições no conjunto de teste (não aumentado)
print("Realizando predições no conjunto de teste...")
y_pred = svm_clf.predict(X_test_flat)

end_time = time.time()
processing_time = end_time - start_time

# --- 4. Cálculo dos Resultados ---
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy

print("\n--- Resultados (SVM) ---")
print(f"Acurácia: {accuracy * 100:.2f}%")
print(f"Taxa de Erro: {error_rate * 100:.2f}%")
print(f"Tempo de Processamento Total: {processing_time:.2f} segundos")