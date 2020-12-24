# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 14:33:16 2020

@author: LiMingbo
"""


import numpy as np
import keras
from keras.models import Sequential
from keras import layers, models
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
import datetime
import matplotlib.pyplot as plt
import os


#给定模仿层的特征向量维数
D=4608

#模型训练参数给定
BATCH_SIZE=64
EPOCHS=1000
VERBOSE=1
VALIDATIOIN_SPLIT=0.2 #数据集，验证集 8:2
SHUFFLE=True #数据新一轮开始时打乱

#模型训练参数给定结束

#加载数据
train_x=np.load('data/glint_trainImgs_10.npy')
train_y2=np.load('data/glint_trainLabels_10.npy')
train_y1=np.load('data/glint_vgg19Labels_10.npy')
#对数据集进行打乱
train_x,train_y2 = shuffle(train_x,train_y2, random_state=1337) 
#生成预测标签

  

input_shape=(112,112,3)
img_input = layers.Input(shape=input_shape)

x=layers.Conv2D(16, (3, 3), activation='relu')(img_input)
x=layers.MaxPooling2D((2, 2))(x)

x=layers.Conv2D(32, (3, 3), activation='relu')(x)
x=layers.MaxPooling2D((2, 2))(x)
    
x=layers.Conv2D(16, (1, 1), activation='relu')(x)
x=layers.Conv2D(128, (3, 3), activation='relu')(x)
x=layers.Conv2D(16, (1, 1), activation='relu')(x)
x=layers.Conv2D(128, (3, 3), activation='relu')(x)
    
x=layers.MaxPooling2D((2, 2))(x)

x=layers.Conv2D(32, (1, 1), activation='relu')(x)
x=layers.Conv2D(256, (3, 3), activation='relu')(x)
x=layers.Conv2D(32, (1, 1), activation='relu')(x)

x=layers.MaxPooling2D((2, 2))(x)
x=layers.Conv2D(32, (1, 1), activation='relu')(x)
x=layers.Conv2D(256, (3, 3), activation='relu')(x)

x=layers.MaxPooling2D((2, 2))(x)
x=layers.Conv2D(D, (1, 1), activation='relu')(x)
MinicOutput=layers.Flatten(name='Minic_output')(x)
    
x=layers.Dense(128,activation="relu",name="Identity")(MinicOutput)
softmaxOutput=layers.Dense(10,activation="softmax",name="Softmax_output")(x)

model=Model(input=img_input,output=[MinicOutput,softmaxOutput])


model.summary()


model.compile(optimizer='adam',
              loss=[ keras.losses.mean_squared_error
                    ,keras.losses.SparseCategoricalCrossentropy(from_logits=True)],
              loss_weights=[0,1],
              metrics=['accuracy'])



#对模型的保存进行规范命名
curr_time = datetime.datetime.now()
year=curr_time.year
month=curr_time.month
day=curr_time.day
hour=curr_time.hour
minute=curr_time.minute
now=str(year)+str(month)+str(day)+str(hour)+str(minute)
modelname='model/student/vgg19_student_EPOCHS_{}_BATCH_SIZE_{}_time_{}.h5'.format(EPOCHS,BATCH_SIZE,now)

#进行训练
history=model.fit(x=train_x,y=train_y2,
          batch_size=BATCH_SIZE,epochs=EPOCHS,shuffle=SHUFFLE,verbose=VERBOSE)
model.save(modelname)
#对训练图像进行数据保存


NowPath=os.path.abspath('.')
EverylogName=os.path.join(NowPath,'logs',now)
os.makedirs(EverylogName)
accuImgName='logs/{}/vgg19_student_accu_EPOCHS_{}_BATCH_SIZE_{}_time_{}.jpg'.format(now,EPOCHS,BATCH_SIZE,now)
                                                                                          
lossImgName='logs/{}/vgg19_student_loss_{}_BATCH_SIZE_{}_time_{}.jpg'.format(now,EPOCHS,BATCH_SIZE,now)
 


#绘制loss
plt.Figure()
plt.plot(history.history['loss'])
plt.plot(history.history['Minic_output_loss'])
plt.plot(history.history['Softmax_output_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['loss','Minic_output_loss','Softmax_output_loss'],loc='upper right')
plt.legend(['loss','Softmax_output_loss'],loc='upper right')
plt.savefig(lossImgName)

                 
#绘制准确率                                                                                         


plt.plot(history.history['Softmax_output_accuracy'])
plt.title('model accuracy')

plt.ylim(0,1)
plt.ylabel('accuracy')
plt.xlabel('epochs')
#plt.legend(['Softmax_output_accuracy'],loc='upper right')
plt.savefig(accuImgName)

del model








