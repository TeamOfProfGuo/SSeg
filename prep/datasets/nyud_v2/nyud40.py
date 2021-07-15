import cv2
import scipy.io as scio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

dataFile ="../dataset/NYUD_v2/nyuv2_meta/labels40.mat"
data = scio.loadmat(dataFile)
labels = np.array(data["labels40"])

path_converted = '../../../dataset/NYUD_v2/nyu_labels40/'
if not os.path.isdir(path_converted):
    os.makedirs(path_converted)

labels_number = []
for i in range(1449):
    labels_number.append(labels[:, :, i].transpose((1, 0)))  # 转置
    labels_0 = np.array(labels_number[i])
    # print labels_0.shape
    print(type(labels_0))
    label_img = Image.fromarray(np.uint8(labels_number[i]))
    # label_img = label_img.rotate(270)
    label_img = label_img.transpose(Image.ROTATE_270)

    iconpath = path_converted + str(i) + '.png'
    label_img.save(iconpath, optimize=True)

unique, counts = np.unique(labels, return_counts=True)
class_cnt = dict(zip(unique, counts))
class_cnt1 = pd.Series(class_cnt,index=class_cnt.keys())



# to check the labels for one image (left right flipped)
label0 = labels[:, :, 0]
label_img = Image.fromarray(np.uint8(label0))
label_img.show()
label_df = pd.DataFrame(label0)


# extract all names
f = scio.loadmat("../../../dataset/NYUD_v2/nyuv2_meta/classMapping40.mat")
all_names = []
names = np.squeeze(f['className'])
all_names = []
for e in names:
    all_names.append(e[0])
all_names = pd.DataFrame(all_names)


# train test split
splitFile ="../dataset/NYUD_v2/nyuv2_meta/splits.mat"
data = scio.loadmat(splitFile)
train = data['trainNdxs']
train_df = pd.DataFrame(train)
train_df.to_csv('../dataset/NYUD_v2/train.txt', index=False, header=False)
test = data['testNdxs']
test_df = pd.DataFrame(test)
test_df.to_csv('../dataset/NYUD_v2/test.txt', index=False, header=False)

