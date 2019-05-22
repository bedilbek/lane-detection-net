import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model_file("models/classification_third.h5")
tflite_model = converter.convert()
open("models/classification_third.tflite", "wb").write(tflite_model)
