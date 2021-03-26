import tensorflow as tf
import zipfile
import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DESIRED_ACCURACY = 0.9
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs = {}):
    if(logs.get("accuracy") >= DESIRED_ACCURACY):
      print("Reached 90% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

categories = 8

model = tf.keras.applications.DenseNet121(
    include_top=True, weights=None, input_tensor=None,
    input_shape=(300,300,3), pooling=None, classes=5)

train_datagen = ImageDataGenerator(rotation_range = 90, 
                                   width_shift_range = 50,
                                   height_shift_range = 50,
                                   shear_range = .5,
                                   zoom_range = .5,
                                   horizontal_flip = True,
                                   vertical_flip = True,
                                   rescale = 1/255,
                                   validation_split = .2)

batch_size = 2

train_generator = train_datagen.flow_from_directory("../data/prototype_5_class", target_size = (300, 300), batch_size = batch_size, class_mode = "categorical", subset = "training")
validation_generator = train_datagen.flow_from_directory("../data/prototype_5_class", target_size = (300, 300), batch_size = batch_size, class_mode = "categorical", subset = "validation")

#!rm -rf ./logs/

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir, histogram_freq = 1)

#%tensorboard --logdir logs/fit

model.compile(loss = "categorical_crossentropy", optimizer = tf.keras.optimizers.Adam(learning_rate = .001), metrics = ["accuracy"])
history = model.fit_generator(train_generator, steps_per_epoch = 1, epochs = 100, verbose = 1, callbacks = [callbacks, tensorboard_callback], validation_data = validation_generator)