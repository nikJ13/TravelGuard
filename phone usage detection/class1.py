import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

NAME= "phone-call-cnn-64x2-{}".format(int(time.time()))
#gpu_options=tf.GPUoptions(per_process_gpu_memory_fraction=0.333)
#sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tensorboard=TensorBoard(log_dir='logs/{}'.format(NAME))

X=pickle.load(open("X.pickle","rb"))
y=pickle.load(open("y.pickle","rb"))
X=X/255.0
model=Sequential()
model.add(Conv2D(64,(3,3),input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

model.fit(X,y,batch_size=32, epochs=50,validation_split=0.1, callbacks=[tensorboard])
