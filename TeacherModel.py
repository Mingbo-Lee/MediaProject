# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 08:38:32 2020

@author: LiMingbo
"""

from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras import layers
import numpy as np 


# teacher model 构造
teacher_model=VGG19(weights="imagenet",include_top=False,input_shape=(112,112,3))
for i ,layer in enumerate(teacher_model.layers):
    print(i,layer.name,layer.output_shape)
    
y_vgg=teacher_model.get_layer('block5_pool').output
y=layers.Flatten(name='Flatten')(y_vgg)
model=Model(input=teacher_model.input,output=y)


#加载训练集 进行预测 生成标签
train_x=np.load('data/glint_trainImgs_6.npy')
train_y1=model.predict(train_x)

#保存实验结果
np.save('data/glint_vgg19Labels_6.npy',train_y1)



    
    