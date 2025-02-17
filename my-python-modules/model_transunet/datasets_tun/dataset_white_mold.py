import os
import random
import h5py
import numpy as np
import torch
import cv2

from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset

from model_transunet.wm_utils import WM_Utils
import xml.etree.ElementTree as ET
 
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
    # Probably interpolation create some noise in the label image, as example 
    # some pixels of zero value can become labeled with the same class id.
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

# count the number of pixels per label of the labeled mask image 
def count_label_pixels(label):
    # h, w, c = label.shape
    h, w = label.shape
    number_of_labels = np.zeros(9, dtype=int)
    for lin in range(h):
        for col in range(w):
            # number_of_labels[label[lin][col][0]] += 1
            number_of_labels[label[lin][col]] += 1

    # returning the number of pixels of all labels 
    return number_of_labels

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # print(f'')
        # print(f'image.shape: {image.shape}')        
        # print(f'label.shape: {label.shape}')     

        # number_of_labels = count_label_pixels(label)
        # print(f'number_of_labels 01: {number_of_labels}')

        # print(f'Random Generator class 1 ')
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        # adapted by Rubens for Graysclae and RGB input images
        if len(image.shape) == 2: # grayscale 
            x, y = image.shape
            print(f'x:{x} y:{y}')
            if x != self.output_size[0] or y != self.output_size[1]:
                image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
                label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            
            # print(f'after zoom gray image.shape 01: {image.shape}')        
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            # print(f'after zoom gray image.shape 02: {image.shape}')        
            # print(f'after zoom gray label.shape 01: {label.shape}')        

        else: # RGB image
            x, y, channels = image.shape
            # print(f'x:{x} y:{y} channels:{channels}')
            if x != self.output_size[0] or y != self.output_size[1]:
                scale = (self.output_size[0] / x, self.output_size[1] / y)
                image = zoom(image, scale + (1,), order=3)  # why not 3?
                label = zoom(label, scale, order=0)

            # print(f'after zoom rgb image.shape 01: {image.shape}')        
            image = torch.from_numpy(image.astype(np.float32)).permute(2,0,1)  # Convert (H, W, C) to (C, H, W)
            # print(f'after zoom rgb image.shape 02: {image.shape}')        
            # print(f'after zoom rgb label.shape 01: {label.shape}')        

        # number_of_labels = count_label_pixels(label)
        # print(f'number_of_labels 02: {number_of_labels}')

        # if x != self.output_size[0] or y != self.output_size[1]:
        #     image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
        #     label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        # image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        # print(f'Random Generator class 2 ')
        return sample


class WhiteMold_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        print(f'')
        print(f'class WhiteMold_dataset, constructor called')
        print(f'class WhiteMold_dataset, base_dir: {base_dir}')
        print(f'class WhiteMold_dataset, list_dir: {list_dir}')
        print(f'class WhiteMold_dataset, split: {split}')
        print(f'class WhiteMold_dataset, transform: {transform}')        

        self.transform = transform  # using transform in torch!
        self.split = split
        # getting list of images
        folder = os.path.join(base_dir, split)
        # extract image files list only for the original images 
        self.sample_list = [f for f in os.listdir(folder) if f.endswith('jpg') and 'mask' not in f]
        # self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

        # print(f'class WhiteMold_dataset, self.transform: {self.transform}')
        # print(f'class WhiteMold_dataset, self.split: {self.split}')
        # print(f'class WhiteMold_dataset, self.sample_list: {self.sample_list}')
        # print(f'class WhiteMold_dataset, len(self.sample_list): {len(self.sample_list)}')
        # print(f'class WhiteMold_dataset, self.data_dir: {self.data_dir}')
        # print(f'end constructor')
        # print(f'')

    def __len__(self):
        # print(f'class WhiteMold_dataset, len(self.sample_list) into class method: {len(self.sample_list)}')
        return len(self.sample_list)

    def __getitem__(self, idx):
        # print(f'class WhiteMold_dataset, __getitem__, 1 - idx: {idx}')
        # print(f'class WhiteMold_dataset, __getitem__, 1 - self.sample_list[idx]: {self.sample_list[idx]}')
        # print(f'class WhiteMold_dataset, __getitem__, 1 - self.split: {self.split}')

        # creating bounding boxes list 
        bounding_boxes = []

        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')           
            # print(f'')
            # print(f'class WhiteMold_dataset, __getitem__, train, idx: {idx} - slice_name: {slice_name}')
            # print(f'class WhiteMold_dataset, __getitem__, train, self.data_dir: {self.data_dir}')            

            image_filename = os.path.join(self.data_dir, self.split, slice_name)
            # print(f'class WhiteMold_dataset, __getitem__, train, image_filename: {image_filename}')             
            # original_image = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
            original_image = cv2.imread(image_filename)
            normalized_image = original_image / 255.0 # normalize to [0, 1]

            # print(f'slice_name: {slice_name}')
            # print(f'original_image.shape: {original_image.shape}')
            # print(f'normalized_image.shape: {normalized_image.shape}')

            # ATENÇÃO: VER COMO TRABALHAR COM IMAGENS COLORIDAS --> TRÊS CANAIS

            label_filename = os.path.join(self.data_dir, self.split, slice_name.replace('.jpg', '_mask.png'))
            # print(f'class WhiteMold_dataset, __getitem__, train, label_filename: {label_filename}')
            # rgb_label = cv2.imread(label_filename)
            # print(f'')
            # print(f'label_filename: {label_filename}')
            # print(f'rgb_label.shape: {rgb_label.shape}')
            # for i in range(0,299):
            #     for j in range(0,299):
            #         if (rgb_label[i][j][0] > 0 or \
            #            rgb_label[i][j][1] > 0 or \
            #            rgb_label[i][j][2] > 0):
            #            print(f'{i},{j}: ({rgb_label[i][j][0]},{rgb_label[i][j][1]},{rgb_label[i][j][2]})')

            # label = cv2.imread(label_filename, cv2.IMREAD_GRAYSCALE)
            original_label = cv2.imread(label_filename)
            label = original_label[:, :, 0] # the mask image must to have only 2 dimensions as it's a grayscale image 
            # print(f'printing original label image')
            # print(f'original_label.shape: {original_label.shape}')
            # print(f'label.shape: {label.shape}')

            # h, w, c = original_label.shape
            # for lin in range(h):
            #     for col in range(w):
            #         r = original_label[lin][col][0]
            #         g = original_label[lin][col][1]
            #         b = original_label[lin][col][2]
            #         if r > 0  or  g > 0  or  b > 0:
            #             print(f'({lin},{col}): ({r}, {g}, {b}) - {label_filename}')
            # h, w, c = label.shape
            # for lin in range(h):
            #     for col in range(w):
            #         r = label[lin][col][0]
            #         g = label[lin][col][1]
            #         b = label[lin][col][2]
            #         if r>0 or g>0 or b>0:
                        # print(f'({lin},{col}): ({r}) - read again {image_filename}')                  
                        # print(f'({lin},{col}): ({r}, {g}, {b}) - {image_filename}')                  
                        # if r != 5 or g != 5 or b != 5:
                        #     print(f'({lin},{col}): ({r}, {g}, {b}) - read again - valor diferente de 5')

            # print(f'anormal finish.')
            # print(f'label.shape: {label.shape}')

            # print(f'trainer_white_mold checking max value of the class ids')
            # exit()

            # showing max and min label values 
            # max_val = np.max(label)
            # min_val = np.min(label)
            # if max_val > 5:
            #     print(f'class WhiteMold_dataset, __getitem__, train, label value after read label image - max: {max_val}, min: {min_val}')
            #     print(f'class WhiteMold_dataset, __getitem__, self.sample_list[idx]: {self.sample_list[idx]}')

            # print(f'class WhiteMold_dataset, __getitem__, train, normalized_image.shape: {normalized_image.shape}')
            # print(f'class WhiteMold_dataset, __getitem__, train, label.shape: {label.shape}')
            
        elif self.split == "test":

            # print(f'class WhiteMold_dataset, __getitem__, split=test 01')

            slice_name = self.sample_list[idx].strip('\n')           

            image_filename = os.path.join(self.data_dir, self.split, slice_name)
            original_image = cv2.imread(image_filename)
            # normalized_image = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
            normalized_image = cv2.imread(image_filename)
            normalized_image = normalized_image / 255.0 # normalize to [0, 1]
            print(f'001 normalized_image.shape: {normalized_image.shape}')
            # print(f'class WhiteMold_dataset, __getitem__, test, image.shape: 01 {image.shape}')
            normalized_image = np.expand_dims(normalized_image, axis=0)
            # normalized_image = normalized_image.unsqueeze(0)
            print(f'002 normalized_image.shape: {normalized_image.shape}')
            # normalized_image = normalized_image.permute(0, 3, 1, 2).cpu().detach().numpy()  # Move channel to last dimension
            # normalized_image = normalized_image.permute(0, 3, 1, 2).cpu().detach().numpy()  # Move channel to last dimension
            normalized_image = torch.from_numpy(normalized_image.astype(np.float32)).permute(0, 3, 1, 2)  # Convert (H, W, C) to (C, H, W)

            print(f'003 normalized_image.shape: {normalized_image.shape}')

            # print(f'class WhiteMold_dataset, __getitem__, test, image.shape: 02 {image.shape}')

            label_filename = os.path.join(self.data_dir, self.split, slice_name.replace('.jpg', '_mask.png'))
            original_label = cv2.imread(label_filename)
            label = original_label[:, :, 0]
            print(f'004 label.shape: {label.shape}')

            # label = cv2.imread(label_filename, cv2.IMREAD_GRAYSCALE)
            # label = np.expand_dims(label, axis=0)

            # getting list of bounding boxes from the original image      
            path_and_annotation_filename = os.path.join(self.data_dir, self.split, 'xml', slice_name.replace('.jpg', '.xml'))
            # print(f'path_and_annotation_filename: {path_and_annotation_filename}')
            bounding_boxes = []
            bounding_boxes = get_annotation_from_xml_file(path_and_annotation_filename)

            # print(f'class WhiteMold_dataset, __getitem__, train, image.shape: {normalized_image.shape}')            
            # print(f'class WhiteMold_dataset, __getitem__, train, original_image.shape: {original_image.shape}')
            # print(f'class WhiteMold_dataset, __getitem__, train, label.shape: {label.shape}')

        else:
            print(f'class WhiteMold_dataset, __getitem__, 2 - else do if')
            exit()

            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        # saving image and label for debugging
        # if 'ds-2023-09-07-santa-helena-de-goias-go-fazenda-sete-ilhas-pivo-02-IMG_20230907_104618' in slice_name:
        #     print(f'class WhiteMold_dataset, __getitem__, saving image and mask for debugging 1')
        #     path = '/home/lovelace/proj/proj939/rubenscp/research/white-mold-applications/TransUNet/results/train'       
        #     print(f'class WhiteMold_dataset, __getitem__, path: {path}')        
        #     image_filename = slice_name + '_image.jpeg'
        #     path_and_filename = os.path.join(path, image_filename)
        #     WM_Utils.save_image(path_and_filename, image)
        #     WM_Utils.save_to_excel(path_and_filename.replace('.jpeg', '.xlsx'), 'image', image)

        #     label_filename = slice_name + '_label.jpeg'
        #     path_and_filename = os.path.join(path, label_filename)
        #     WM_Utils.save_image(path_and_filename, label)
        #     WM_Utils.save_to_excel(path_and_filename.replace('.jpeg', '.xlsx'), 'label-mask', label)           
        # end of saving image and label for debugging

        sample = {'image': normalized_image, 
                  'label': label, 
                  'original_image': original_image, 
                  'bbox': bounding_boxes}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

def get_annotation_from_xml_file(xml_filename: str):
    '''
    Get list of bounding boxes from xml file
    '''
    # reading annotation xml
    ann_tree = ET.parse(xml_filename)
    ann_root = ann_tree.getroot()

    # getting list of bounding boxes
    bounding_boxes = []
    
    # print(f'def get_annotation_from_xml_file(xml_filename: str): 01')
    # print(f'ann_root: {ann_root}') 

    # reading objects from xml
    for object in ann_root.findall('object'):
        # print(f'def get_annotation_from_xml_file(xml_filename: str): 02')
        class_name = object.find('name').text
        class_name = ''.join(class_name)

        # print(f'def get_annotation_from_xml_file(xml_filename: str): 03')
        # print(f'class_name: {class_name}')

        # # print(f'rubens 00 len(class_name): {len(class_name_array)}')
        # # print(f'rubens 01 class_name_array: {class_name_array}')
        # # class_name = ''.join(class_name_array)
        # # print(f'rubens 02 class_name: {class_name}')
        bndbox = object.find('bndbox')
        # print(f'def get_annotation_from_xml_file(xml_filename: str): 04')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
        # print(f'def get_annotation_from_xml_file(xml_filename: str): 05')

        bounding_box = {
            "bndbox": {
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            },
            "class_name": class_name,
        }
        # print(f'def get_annotation_from_xml_file(xml_filename: str): 06')
        # print(f'whitemold dataset - bounding_box: {bounding_box}')
        # print(f'def get_annotation_from_xml_file(xml_filename: str): 07')
        bounding_boxes.append(bounding_box)
        # print(f'def get_annotation_from_xml_file(xml_filename: str): 08')

    # print(f'def get_annotation_from_xml_file(xml_filename: str): 09')

    # returning list of bounding boxes
    return bounding_boxes
