
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL
from PIL import Image

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

# # Data Augumentation - Creating more data from existing data

# Resizing and Rescaling


batch_size = 20

training_datagenarator= ImageDataGenerator(rescale=1./255, horizontal_flip=True,
                                           vertical_flip=True ,shear_range=0.2,
                                           zoom_range=0.2, width_shift_range=0.2,
                                           height_shift_range=0.2, validation_split=0.1)


# dividing the data into training and validation


train =training_datagenarator.flow_from_directory('D:\smoke and fire dectection\Tranining_data',
                                                 target_size=(224,224), color_mode='rgb',
                                                 class_mode='binary', batch_size=batch_size ,subset='training')

validation =training_datagenarator.flow_from_directory('D:\smoke and fire dectection\Tranining_data',
                                                      target_size=(224,224) ,color_mode='rgb',
                                                      class_mode='binary', batch_size=batch_size, subset='validation')


# # Now its time to make our CNN


# Initializing CNN
cnn = Sequential()

# adding first layer
cnn.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=[224,224,3]))
cnn.add(MaxPool2D(pool_size=2))

# adding second layer
cnn.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
cnn.add(MaxPool2D(pool_size=2))

# adding third layer
cnn.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
cnn.add(MaxPool2D(pool_size=2))

# Flattening
cnn.add(Flatten())

# Fully connected layer
cnn.add(Dense(units=128, activation='relu'))

# Output layers
cnn.add(Dense(units=1, activation='sigmoid'))


cnn.summary()


# # CNN Model
# ### Compile and Train

checkpoint = tf.keras.callbacks.ModelCheckpoint('D:\smoke and fire dectection\models\model.h5',
                                             monitor='val_loss', mode="min",
                                             save_best_only=True)
callbacks = checkpoint


# In[ ]:


cnn.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

cnn.fit_generator(train, validation_data=validation, epochs=1,
                  steps_per_epoch=train.samples//batch_size,
                  validation_steps=validation.samples//batch_size,
                  callbacks=callbacks
                  )

cnn.save("model.h5")


# # Test It


cnn = load_model('model.h5')


cnn.summary()

image_for_testing = r'D:\smoke and fire dectection\Tranining_data\smoke\14.png'


test_image = image.load_img(image_for_testing, target_size=(224 ,224))
test_image = image.img_to_array(test_image)
test_image = test_image/255
test_image = np.expand_dims(test_image, axis=0)


# result = cnn.predict(test_image)

result = cnn.predict(test_image)
result = np.round(result[0][0], 2)
# 0 is fire; 1 is smoke
if result > 0.5:
    label = 'Smoke'
else:
    label = 'Fire'

image_show =PIL.Image.open(image_for_testing)
plt.imshow(image_show)
plt.title(label)
plt.show()
