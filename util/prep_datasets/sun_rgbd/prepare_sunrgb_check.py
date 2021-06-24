
import urllib.request
from zipfile import ZipFile

import h5py
import numpy as np
import scipy.io
from tqdm import tqdm
import os

# argument parser
output_path= '../dataset/sunrgbd'
output_path = os.path.expanduser(output_path)
# toolbox
toolbox_dir = os.path.join(output_path, 'SUNRGBDtoolbox')

# extract labels from SUNRGBD toolbox
print('Extract labels from SUNRGBD toolbox')
SUNRGBDMeta_dir = os.path.join(toolbox_dir, 'Metadata/SUNRGBDMeta.mat')
allsplit_dir = os.path.join(toolbox_dir, 'traintestSUNRGBD/allsplit.mat')
SUNRGBD2Dseg_dir = os.path.join(toolbox_dir, 'Metadata/SUNRGBD2Dseg.mat')
img_dir_train = []
depth_dir_train = []
label_dir_train = []
img_dir_test = []
depth_dir_test = []
label_dir_test = []

SUNRGBD2Dseg = h5py.File(SUNRGBD2Dseg_dir, mode='r', libver='latest')

# load the data from the matlab file
SUNRGBDMeta = scipy.io.loadmat(SUNRGBDMeta_dir, squeeze_me=True,
                               struct_as_record=False)['SUNRGBDMeta']
split = scipy.io.loadmat(allsplit_dir, squeeze_me=True,
                         struct_as_record=False)
split_train = split['alltrain']

seglabel = SUNRGBD2Dseg['SUNRGBD2Dseg']['seglabel']

for i, meta in tqdm(enumerate(SUNRGBDMeta)):
    meta_dir = '/'.join(meta.rgbpath.split('/')[:-2])
    real_dir = meta_dir.split('/n/fs/sun3d/data/SUNRGBD/')[1]
    depth_bfx_path = os.path.join(real_dir, 'depth_bfx/' + meta.depthname)
    rgb_path = os.path.join(real_dir, 'image/' + meta.rgbname)

    label_path = os.path.join(real_dir, 'label/label.npy')
    label_path_full = os.path.join(output_path, 'SUNRGBD', label_path)

    # save segmentation (label_path) as numpy array
    if not os.path.exists(label_path_full):
        os.makedirs(os.path.dirname(label_path_full), exist_ok=True)
        label = np.array(
            SUNRGBD2Dseg[seglabel[i][0]][:].transpose(1, 0)).\
            astype(np.uint8)
        np.save(label_path_full, label)

    if meta_dir in split_train:
        img_dir_train.append(os.path.join('SUNRGBD', rgb_path))
        depth_dir_train.append(os.path.join('SUNRGBD', depth_bfx_path))
        label_dir_train.append(os.path.join('SUNRGBD', label_path))
    else:
        img_dir_test.append(os.path.join('SUNRGBD', rgb_path))
        depth_dir_test.append(os.path.join('SUNRGBD', depth_bfx_path))
        label_dir_test.append(os.path.join('SUNRGBD', label_path))

# write file lists
def _write_list_to_file(list_, filepath):
    with open(os.path.join(output_path, filepath), 'w') as f:
        f.write('\n'.join(list_))
    print('written file {}'.format(filepath))

_write_list_to_file(img_dir_train, 'train_rgb.txt')
_write_list_to_file(depth_dir_train, 'train_depth.txt')
_write_list_to_file(label_dir_train, 'train_label.txt')
_write_list_to_file(img_dir_test, 'test_rgb.txt')
_write_list_to_file(depth_dir_test, 'test_depth.txt')
_write_list_to_file(label_dir_test, 'test_label.txt')



from encoding.datasets import SUNRGBD
from PIL import Image

sun = SUNRGBD()
img_dir, depth_dir, label_dir = sun.img_dir['list'], sun.depth_dir['list'], sun.label_dir['list']

idx = 0

img_path = os.path.join(sun.BASE_DIR, img_dir[idx])
img = Image.open(img_path).convert('RGB')
i = np.asarray(img)
im = ImageOps.expand(img, border=(0, 0, 20, 40), fill=(255, 255, 255))

dep_path = os.path.join(sun.BASE_DIR, depth_dir[idx])
dep = Image.open(dep_path)

d = np.asarray(dep).astype(np.float)
d_new = (d - 8784) / (52456 - 8784) * 255
im = Image.fromarray(d_new)
im.show()

target_path = os.path.join(sun.BASE_DIR, label_dir[idx])
target = np.load(target_path).astype(np.int8)
label = Image.fromarray(target, 'L')
l = np.asarray(label)
sp = tuple(list(target.shape) + [3])
t = np.zeros(sp).astype(np.int8)
for i in range(len(t)):
    for j in range(len(t[i])):
        t[i, j] = sun.CLASS_COLORS[target[i, j]]
im = Image.fromarray(t, 'RGB')
im.show()

a = np.array([[1, 2], [3, 4]])
for index, x in np.ndenumerate(a):
    print(index, x)

d = sun.img_dir['dict']
for k in d.keys():
    print('type {} len {}'.format(k, len(d[k])))



