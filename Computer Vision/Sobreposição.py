import cv2
import numpy as np
from sklearn import tree
from scipy.ndimage import uniform_filter, generic_filter
import sys

def extract_features(image, window_size=5):
    """
    Extrai atributos de cada pixel com base em sua vizinhança.
    Retorna uma matriz onde cada linha é um pixel e cada coluna é um atributo.
    """
    # 1. O valor do próprio pixel
    pixel_value = image.astype(np.float32)


    # 2. A média da vizinhança (usando um filtro rápido)
    local_mean = uniform_filter(pixel_value, size=window_size)

    # 3. O desvio padrão da vizinhança
    local_std = generic_filter(pixel_value, np.std, size=window_size)

    # Empilha os atributos em uma única matriz de features
    # Formato: (N_pixels, N_features)
    features = np.vstack((
        pixel_value.ravel(),
        local_mean.ravel(),
        local_std.ravel()
    )).T
    
    return features


ax = cv2.imread("janei.pgm", cv2.IMREAD_GRAYSCALE)
ay = cv2.imread("janei-1.pgm", cv2.IMREAD_GRAYSCALE)


features = extract_features(ax)
labels = (ay / 255).ravel()

clf = tree.DecisionTreeClassifier()
clf.fit(features, labels)


qx = cv2.imread("julho.pgm", cv2.IMREAD_GRAYSCALE)
query_features = extract_features(qx)

qp = clf.predict(query_features)
qp_image = qp.reshape(qx.shape).astype(np.uint8)

cv2.medianBlur(qp_image, 7,qp_image)  


cv2.imwrite("julho-p1.pgm", qp_image.astype(np.uint8) * 255)

# Criação da Imagem Final com Sobreposição Vermelha ---
julho_colorida = cv2.cvtColor(qx, cv2.COLOR_GRAY2BGR)
julho_colorida[qp_image == 0] = [0, 0, 255] # Pinta de vermelho onde a máscara é 0
cv2.imwrite("julho-c1.png", julho_colorida)
