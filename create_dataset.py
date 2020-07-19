import os
import pandas as pd
import h5py
import cv2

DATASET_PATH = 'data'

meta_csv = pd.read_csv(os.path.join(DATASET_PATH, 'Meta.csv'))
train_csv = pd.read_csv(os.path.join(DATASET_PATH, 'Train.csv'))
test_csv = pd.read_csv(os.path.join(DATASET_PATH, 'Test.csv'))

meta_csv = meta_csv.drop(['ShapeId', 'ColorId', 'SignId'], axis=1)
train_csv = train_csv.drop(['Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2'], axis=1)
test_csv = test_csv.drop(['Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2'], axis=1)

meta_images = list()
meta_labels = list()

for i, j in enumerate(meta_csv['Path']):
    path = os.path.join(DATASET_PATH, j)
    label = meta_csv['ClassId'][i]
    img = cv2.imread(path)
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    meta_images.append(img)
    meta_labels.append(label)

train_images = list()
train_labels = list()

for i, j in enumerate(train_csv['Path']):
    path = os.path.join(DATASET_PATH, j)
    label = train_csv['ClassId'][i]
    img = cv2.imread(path)
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    train_images.append(img)
    train_labels.append(label)

test_images = list()
test_labels = list()

for i, j in enumerate(test_csv['Path']):
    path = os.path.join(DATASET_PATH, j)
    label = test_csv['ClassId'][i]
    img = cv2.imread(path)
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    test_images.append(img)
    test_labels.append(label)

train = h5py.File('train.h5', 'w')
train.create_dataset('Images', data=train_images)
train.create_dataset('Labels', data=train_labels)
train.close()

test = h5py.File('test.h5', 'w')
test.create_dataset('Images', data=test_images)
test.create_dataset('Labels', data=test_labels)
test.close()

meta = h5py.File('meta.h5', 'w')
meta.create_dataset('Images', data=meta_images)
meta.create_dataset('Labels', data=meta_labels)
meta.close()

