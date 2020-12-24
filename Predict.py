

import numpy as np
import keras
from keras.models import Sequential
from keras import layers, models
from keras.models import Model
from keras.preprocessing import image

#加载模型
model=models.load_model('model/student/vgg19_student_EPOCHS_100_BATCH_SIZE_64_time_20201210174.h5')
#处理加载图片
img_path='dataset/glint360k_100id/id_4/4_262.jpg'
img=image.load_img(img_path)
x=image.img_to_array(img)
x=np.array([x])

result=model.predict(x)
#得到两个预测输出，获取第二个(one-hot)输出，即可获得结果
for i in range(result[1].shape[1]):
    if result[1][0][i]!=0:
        print("The predict result is:id_{}".format(i))


