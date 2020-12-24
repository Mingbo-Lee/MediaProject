import numpy as np
import keras
from keras.models import Sequential
from keras import layers, models
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
        'dataset/glint360k_100id',
        target_size=(112, 112),
        batch_size=32,
        subset='training',
        class_mode='sparse')

validation_generator=train_datagen.flow_from_directory(
        'dataset/glint360k_100id',
        target_size=(112, 112),
        batch_size=32,
        subset='validation',
        class_mode='sparse')


teachermodel=models.load_model('model/teacher/vgg19_teacher.h5')
teacher_label=teachermodel.predict_generator(train_generator)
