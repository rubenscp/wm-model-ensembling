import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset

from model_transunet.wm_utils import WM_Utils

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        print(f'Synapse_dataset')
        print(f'list_dir: {list_dir}')
        print(f"self.split: {self.split+'.txt'}")
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        print(f'')
        print(f'class Synapse_dataset, self.sample_list: {self.sample_list}')
        print(f'')
        self.data_dir = base_dir

    def __len__(self):
        print(f'class Synapse_dataset, len(self.sample_list): {len(self.sample_list)}')
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            print(f'')
            print(f'class Synapse_dataset, __getitem__, train, slice_name: {slice_name}')
            print(f'class Synapse_dataset, __getitem__, train, self.data_dir: {self.data_dir}')            
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            print(f'class Synapse_dataset, __getitem__, train, data: {data}')
            print(f'class Synapse_dataset, __getitem__, train, data.keys(): {data.keys()}')
            print(f'class Synapse_dataset, __getitem__, train, data["image"].shape: {data["image"].shape}')
            # print(f'class Synapse_dataset, __getitem__, train, data["image"].dtype: {data["image"].dtype}')
            # print(f'class Synapse_dataset, __getitem__, train, data["image"]: {data["image"]}')
            # print(f'class Synapse_dataset, __getitem__, train, data["image"].sum(): {data["image"].sum()}')
            print(f'class Synapse_dataset, __getitem__, train, data["label"].shape: {data["label"].shape}')
            # print(f'class Synapse_dataset, __getitem__, train, data["label"].dtype: {data["label"].dtype}')
            # print(f'class Synapse_dataset, __getitem__, train, data["label"]: {data["label"]}')
            print(f'class Synapse_dataset, __getitem__, train, data["label"].sum(): {data["label"].sum()}')
            image, label = data['image'], data['label']
          
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            print(f'class Synapse_dataset, __getitem__, not train, vol_name: {vol_name}')
            print(f'class Synapse_dataset, __getitem__, not train, filepath: {filepath}')
            data = h5py.File(filepath)
            print(f'class Synapse_dataset, __getitem__, not train, data: {data}')
            image, label = data['image'][:], data['label'][:]

        # saving image and label for debugging
        if 'case0005' in slice_name:
            print(f'class Synapse_dataset, __getitem__, saving image and mask for debugging 1')
            path = '/home/lovelace/proj/proj939/rubenscp/research/white-mold-applications/TransUNet/results/train'       
            print(f'class Synapse_dataset, __getitem__, path: {path}')        
            image_filename = slice_name + '_image.jpeg'
            path_and_filename = os.path.join(path, image_filename)
            WM_Utils.save_image(path_and_filename, image)
            WM_Utils.save_to_excel(path_and_filename.replace('.jpeg', '.xlsx'), 'image', image)

            label_filename = slice_name + '_label.jpeg'
            path_and_filename = os.path.join(path, label_filename)
            WM_Utils.save_image(path_and_filename, label)
            WM_Utils.save_to_excel(path_and_filename.replace('.jpeg', '.xlsx'), 'label-mask', label)           
        # end of saving image and label for debugging
        
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

    def save_image(self, path, filename, image):
        import cv2    
        path_and_filename = os.path.join(path, filename)
        cv2.imwrite(path_and_filename, image)

        import pandas as pd
        df = pd.DataFrame(image)
        path_and_filename = os.path.join(path, filename.replace('.png', '.xlsx'))
        df.to_excel(path_and_filename, index=False)

