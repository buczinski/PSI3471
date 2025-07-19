#Pedro Buczinski - 12555266
#Média de acurácia: 97% 

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras import optimizers
import numpy as np; import sys; import os; from time import time

AX = np.load("kmnist-train-imgs.npz")['arr_0']
AY = np.load("kmnist-train-labels.npz")['arr_0']
QX = np.load("kmnist-test-imgs.npz")['arr_0']
QY = np.load("kmnist-test-labels.npz")['arr_0']



AX=255-AX; QX=255-QX
nclasses = 10
nl, nc = AX.shape[1], AX.shape[2] #28, 28
AX = (AX.astype('float32') / 255.0)-0.5 # -0.5 a +0.5
QX = (QX.astype('float32') / 255.0)-0.5 # -0.5 a +0.5
AX = np.expand_dims(AX,axis=3) # AX [70000,28,28,1]
QX = np.expand_dims(QX,axis=3)

model = Sequential()  # 28x28
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu',
                 input_shape=(nl, nc, 1)))  # 24x24x32
model.add(MaxPooling2D(pool_size=(2, 2)))  # 12x12x32
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))  # 8x8x64
model.add(MaxPooling2D(pool_size=(2, 2)))  # 4x4x64
model.add(Dropout(0.25))
model.add(Flatten())  # 4*4*64 = 1024
model.add(Dense(256, activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(nclasses))

model.summary()
opt=optimizers.Adam()
model.compile(optimizer=opt, loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
t0=time()
model.fit(AX, AY, batch_size=100, epochs=30, verbose=2)
t1=time(); print("Tempo de treino: %.2f s"%(t1-t0))
score = model.evaluate(QX, QY, verbose=False)
print('Test loss: %.4f'%(score[0]))
print('Test accuracy: %.2f %%'%(100*score[1]))
print('Test error: %.2f %%'%(100*(1-score[1])))
t2=time()
QP2=model.predict(QX); QP=np.argmax(QP2,1)
t3=time(); print("Tempo de predicao: %.2f s"%(t3-t2))
nerro=np.count_nonzero(QP-QY); print("nerro=%d"%(nerro))
model.save('convol.keras')