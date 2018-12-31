import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.models import load_model
from keras.models import Model

x_train = pd.read_csv('./semeion.data',header=None,sep='\s').as_matrix()
x_train,y_train=np.split(x_train[:],[len(x_train[0])-10],axis=1)
x_train=x_train.reshape(x_train.shape[0],16,16,1)
# y_oneH=np_utils.to_categorical(y_train)
print('training data size: ',x_train.shape)

model=Sequential()
model.add(Conv2D(filters=16,kernel_size=(5,5),padding='same',input_shape=(16,16,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=36,kernel_size=(5,5),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu',name="Dense_2"))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax',name="Dense_3"))
# print(model.summary())
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history=model.fit(x=x_train,y=y_train,validation_split=0.2,epochs=20,batch_size=30,verbose=2)
# show_train_history('acc','val_cc')
model.save('my_modelCNN.h5')