import numpy as np
import pandas as pd
import cv2
from random import shuffle

df = pd.read_csv('training.csv')
c1 = 0
d1 = 0
for index, row in df.iterrows():
    path = './images/' + row['image_name']
    img = cv2.imread(path)
    #edges = np.multiply(0.25,cv2.Canny(img,50,99)) + np.multiply(0.5,cv2.Canny(img,100,149)) + np.multiply(0.75,cv2.Canny(img,150,199)) + cv2.Canny(img,200,255)
    #added extra channel edges as our 4th channel which incresed our iou in 2nd level from 0.62 to 0.696396
    edges = cv2.Canny(img,50,255)
    #print(edges[:,:,np.newaxis].shape)
    img = np.concatenate((img,edges[:,:,np.newaxis]), axis = 2)
    #print(img.shape)
    if c1 == 0:
        data_img = img[np.newaxis,:]
        c1 = 1
    else:
        img = img[np.newaxis,:]
        data_img = np.concatenate((data_img,img), axis = 0)
    labelx = np.divide([row['x1'], row['x2']],640)
    labely = np.divide([row['y1'], row['y2']],480)
    label = np.concatenate((labelx,labely), axis = None)
    label = np.multiply(label,1)
    #print(label)
    if d1 == 0:
        data_lab = label[np.newaxis,:]
        d1 = 1
    else:
        label = label[np.newaxis,:]
        data_lab = np.concatenate((data_lab,label), axis = 0)
    print(index)
    if (index+1)%500 == 0:
        """ if index  == 12814:
            index = idxx + 1 """
        ax = 'data_imgtrain'+str(index+1)+'.npy'
        al = 'data_lab'+str(index+1)+'.npy'
        np.save(ax,data_img)
        np.save(al,data_lab)
        data_img = None
        data_lab = None
        c1 = 0
        d1 = 0
        idxx = index + 1
