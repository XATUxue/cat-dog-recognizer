import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from IPython.display import display
from PIL import Image


#Initializing the CNN
classifier = Sequential()

#Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

#Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Flattening
classifier.add(Flatten())

#Full Connection
classifier.add(Dense(output_dim =128, activation='relu'))
classifier.add(Dense(output_dim =1, activation='sigmoid'))

#Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the CNN to the images
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('Convolutional_Neural_Networks/dataset/training_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
test_set = test_datagen.flow_from_directory('Convolutional_Neural_Networks/dataset/test_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
classifier.fit_generator(training_set, steps_per_epoch = 8000, epochs = 10, validation_data = test_set, validation_steps = 800)

#Testing
test_image = image.load_img('random.jpg', target_size = (64, 64))
test_image = img.image_to_array (test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)

if[0][0] >= 0.5:
    prediction = 'Dog'
else:
    prediction = 'Cat'

print(prediction)





