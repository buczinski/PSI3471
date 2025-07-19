#Pedro Buczinski - 12555266
#Média de acurácia: 92% 

import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Normalization, Dropout
from tensorflow.keras import optimizers
import numpy as np; import sys
from time import time

AX = np.load("kmnist-train-imgs.npz")['arr_0']
AY = np.load("kmnist-train-labels.npz")['arr_0']
QX = np.load("kmnist-test-imgs.npz")['arr_0']
QY = np.load("kmnist-test-labels.npz")['arr_0']

AX=255-AX; QX=255-QX
nclasses = 10
nl, nc = AX.shape[1], AX.shape[2] #28, 28

model = Sequential()
model.add(Normalization(input_shape=(nl, nc)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))  
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu')) 
model.add(Dropout(0.3)) 
model.add(Dense(nclasses))

model.summary()

opt = optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt,
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.get_layer(index=0).adapt(AX) #Calcula media e desvio
t0=time()
model.fit(AX, AY, batch_size=100, epochs=30, verbose=2)
t1=time(); print("Tempo de treino: %.2f s"%(t1-t0))
score = model.evaluate(QX, QY, verbose=0)
print('Test loss: %.4f'%(score[0]))
print('Test accuracy: %.2f %%'%(100*score[1]))
model.save('densa.keras')

