import warnings
import tensorflow as tf
from tensorflow.keras import datasets,layers,models
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()


classes =["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)

x_test = x_test/255.0
x_train = x_train/255.0

print(x_train[0])
print(x_train[0].shape)


model = models.Sequential(
    [
        layers.Conv2D(filters=32,kernel_size=(3,3), activation='relu', input_shape=(32,32,3)),
        layers.MaxPool2D((2,2)),
        layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(32,32,3)),
        layers.MaxPool2D((2,2)),
        layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=(32,32,3)),
        layers.MaxPool2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ]
)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=["accuracy"]
)

model.fit(x_train,y_train, epochs=15)

model.evaluate(x_test,y_test)

model.save("ciafar10.model")
