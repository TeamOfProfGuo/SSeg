# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py
import os
from PIL import Image
import pandas as pd

f = h5py.File("../dataset/NYUD_v2/nyu_depth_v2_labeled.mat")   # h5py is container for datasets and groups


def get_images():
    images = f["images"]
    images = np.array(images)

    path_converted = '../../../dataset/NYUD_v2/nyu_images'
    if not os.path.isdir(path_converted):
        os.makedirs(path_converted)

    # extract rgb images
    images_number = []
    for i in range(len(images)):
        print(i)
        images_number.append(images[i])
        a = np.array(images_number[i])

        # convert array to image
        r = Image.fromarray(a[0]).convert('L')
        g = Image.fromarray(a[1]).convert('L')
        b = Image.fromarray(a[2]).convert('L')
        img = Image.merge("RGB", (r, g, b))
        img = img.transpose(Image.ROTATE_270)
        # plt.imshow(img)
        # plt.show()
        iconpath = os.path.join(path_converted, str(i) + '.jpg')
        img.save(iconpath, optimize=True)


#Extract depth images
def get_depth():
    depths = f["depths"]
    depths = np.array(depths)

    path_converted = '../../../dataset/NYUD_v2/nyu_depths/'
    if not os.path.isdir(path_converted):
        os.makedirs(path_converted)

    max = depths.max()
    print('depths shape {}, max {}, min {}'.format(depths.shape, depths.max(), depths.min()))

    depths = depths / max * 255
    depths = depths.transpose((0, 2, 1))
    print('after transformation')
    print('depths shape {}, max {}, min {}'.format(depths.shape, depths.max(), depths.min()))

    for i in range(len(depths)):
        print(str(i) + '.png')
        depths_img = Image.fromarray(np.uint8(depths[i]))
        depths_img = depths_img.transpose(Image.FLIP_LEFT_RIGHT)

        iconpath = path_converted + str(i) + '.png'
        depths_img.save(iconpath, 'PNG', optimize=True)


# Extract ground truth
labels = f["labels"]
labels = np.array(labels)

path_converted = '../../../dataset/NYUD_v2/nyu_labels/'
if not os.path.isdir(path_converted):
    os.makedirs(path_converted)

labels_number = []
for i in range(len(labels)):
    labels_number.append(labels[i])
    labels_0 = np.array(labels_number[i])

    label_img = Image.fromarray(np.uint8(labels_number[i]))
    # label_img = label_img.rotate(270)
    label_img = label_img.transpose(Image.ROTATE_270)

    iconpath = path_converted + str(i) + '.png'
    label_img.save(iconpath, 'PNG', optimize=True)
    print(str(i)+'png')


# extract all names

def get_names():
    all_names = []

    names = f['names']
    names = np.squeeze(np.array(names))

    for i in range(len(names)):
        name = names[i]
        v = np.squeeze(f[name])
        name = ''.join(chr(i) for i in v)
        all_names.append(name)

        print('{} th name is {}'.format(i, name))
    return all_names

# write the names to file
all_names = pd.DataFrame(get_names())
out_path = '../../../dataset/NYUD_v2/meta_data/'
if not os.path.exists(out_path):
    os.makedirs(out_path)
all_names.to_csv(out_path+'names.txt', header=False)

