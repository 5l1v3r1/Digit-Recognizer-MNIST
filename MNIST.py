'GHOST'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation,Flatten, Dropout, Conv2D, BatchNormalization, MaxPool2D
from keras.models import Sequential
import os
import csv

data_train=pd.read_csv('G:\\PYTHON\\Digit Recogniser MNIST\\train.csv')
data_train=np.array(data_train)
data_trainX=data_train[:,1:]
X=data_trainX.reshape(-1,28,28,1)
Y=data_train[:,0]


data_test=pd.read_csv('G:\\PYTHON\\Digit Recogniser MNIST\\test.csv')
data_test=np.array(data_test)
print(data_test.shape)
X_test =[]
for i in data_test:
    X_test.append(i.reshape(28,28))

X_test = np.array(X_test)
y=[]
for i in Y:
    fy=np.zeros((10))
    fy[i]=1
    y.append(fy)

y=np.array(y)

model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu',input_shape = (28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
model.fit(x=X,y=y,epochs=10)

ind = np.random.randint(0,28000)

a=model.predict(X_test[ind].reshape(1,28,28,1))
plt.imshow(X_test[ind].reshape(28,28))
print(np.argmax(a))


predictions[ind]
model.save("G:\\PYTHON\\Digit Recogniser MNIST\\99weights.h5")
        

predictions = []
for i in X_test:
    predictions.append(np.argmax(model.predict(i.reshape(1,28,28,1)))) 

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)), "Label": predictions})
submissions.to_csv("Final.csv", index=False, header=True)
