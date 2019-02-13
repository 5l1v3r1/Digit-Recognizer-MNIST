'GHOST'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation,Flatten
from keras.models import Sequential

data_train=pd.read_csv('G:\\PYTHON\\Digit Recogniser MNIST\\train.csv')
data_train=np.array(data_train)
data_trainX=data_train[:,1:]
X=data_trainX.reshape(-1,784)
Y=data_train[:,0]


data_test=pd.read_csv('G:\\PYTHON\\Digit Recogniser MNIST\\train.csv')
X_test=np.array(data_test)
X_test=X.reshape(-1,784)

y=[]
for i in range(len(Y)):
    fy=np.zeros((10))
    fy[Y[i]]=1
    y.append(fy)

y=np.array(y)


model=Sequential()
#model.add(Dense(Flatten()))
model.add(Dense(1024,activation='tanh'))
model.add(Dense(512, activation='tanh'))
model.add(Dense(256,activation='tanh'))
model.add(Dense(128,activation='tanh'))
model.add(Dense(64,activation='tanh'))
model.add(Dense(32,activation='tanh'))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x=X,y=y,epochs=1)
np.set_printoptions(suppress="True")
ind = np.random.randint(0,28000)
a=model.predict(X_test[ind].reshape(1,784))
plt.imshow(X_test[ind].reshape(28,28))
print(np.argmax(a))


print(ind)

model.summary()
