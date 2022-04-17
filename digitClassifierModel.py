import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) #1st Hidden Layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) #2nd Hidden Layer

model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) #Output Layer, 0-9 digits so, used 10 as parameter

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(x_train, y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)

model.save('Digit_Classifier.model')

new_model = tf.keras.models.load_model('Digit_Classifier.model') #load/retrieve the model

predictions = new_model.predict(x_test)
print(predictions)


#print(np.argmax(predictions[1000]))



converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
tflite_model = converter.convert()

# Save the TF Lite model as file
f = open('DigitClassifier.tflite', "wb")
f.write(tflite_model)
f.close()

print('TF Lite model:', os.path.join(os.getcwd(), 'DigitClassifier.tflite'))
