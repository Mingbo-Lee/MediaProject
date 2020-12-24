# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 14:54:02 2020

@author: LiMingbo
"""

import numpy as np
import cv2 as cv
import os 
from  os import path
#系列函数的定义

#获得当前文件夹下的文件夹数目
def CountFileNumber(DatasetPath):
    count=0
    for filedirs in os.listdir(DatasetPath):
        count=count+1
    return count

#整体思路:对LFW数据集(或派生的数据集) 以文件夹的个数生成one-hot 向量
raw_resolution=112

imgsData=[]
labels=[]

DatasetPath="E:\\LiMingbo\\faceRecognitionDemo\\dataset\\glint360k_6id"

#进行文件夹的计数，确定one-hot编码的大小
labelsSize=CountFileNumber(DatasetPath)

#数据集里照片的张数，即处理对象的总个数
FileDirCount=0
objectCount=0

print("Start process {}".format(DatasetPath))
for filedirs in os.listdir(DatasetPath):
    #获取文件夹数目
    imgsdir=os.path.join(DatasetPath,filedirs)
    imgs=os.listdir(imgsdir)
    imgs.sort()
    for singleImg in imgs:
        labels.append(FileDirCount)
        ImgWholeName=os.path.join(DatasetPath,filedirs,singleImg)
        photo=cv.imread(ImgWholeName)
        print("Prceoss: {}".format(os.path.join(filedirs,singleImg)))
        #提取图片和标签
        imgsData.append(photo)
        objectCount=objectCount+1
    FileDirCount=FileDirCount+1
print("Finish process {}".format(DatasetPath)) 
LFWImgs=np.array(imgsData).reshape((objectCount,raw_resolution,raw_resolution,3))
LFWLabels=np.array(labels).reshape((objectCount,1))
 
       
np.save('data/glint_trainImgs_6.npy',LFWImgs)
print("Shape of glint_trainImgs:{}".format(LFWImgs.shape))
print("Save glint_trainImgs successfully!")
np.save('data/glint_trainLabels_6.npy',LFWLabels)
print("Shape of glint_trainLabel:{}".format(LFWLabels.shape))
print("Save glint_trainLabel successfully!")
 
    









