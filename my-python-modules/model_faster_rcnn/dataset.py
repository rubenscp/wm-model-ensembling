import os
import numpy as np
import torch
import cv2
from xml.etree import ElementTree as et

from   model_faster_rcnn.augmentation import *
import model_faster_rcnn.utils as utils

# Importing python modules
from common.manage_log import *

class WhiteMoldImagesDatasetFRCNN(torch.utils.data.Dataset):

    def __init__(self, files_dir, width, height, classes, transforms=None):
        self.transforms = transforms
        self.files_dir = files_dir
        self.height = height
        self.width = width

        # sorting the images for consistency
        # To get images, the extension of the filename is checked to be jpg
        self.imgs = [image for image in sorted(os.listdir(files_dir))
                        if image[-4:]=='.jpg']

        # classes: 0 index is reserved for background
        # self.classes = [_, 'Apothecium','banana','orange']
        self.classes = classes

    def __getitem__(self, idx):

        img_name = self.imgs[idx]
        image_path = os.path.join(self.files_dir, img_name)

        # reading the images and converting them to correct size and color
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        # diving by 255
        img_res /= 255.0

        # annotation file
        annot_filename = img_name[:-4] + '.xml'
        # annot_filename = img_name + '.xml'
        annot_file_path = os.path.join(self.files_dir, annot_filename)

        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        # cv2 image gives size as height x width
        wt = img.shape[1]
        ht = img.shape[0]

        # box coordinates for xml files are extracted and corrected for image size given
        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))

            # bounding box
            xmin = int(member.find('bndbox').find('xmin').text)
            xmax = int(member.find('bndbox').find('xmax').text)

            ymin = int(member.find('bndbox').find('ymin').text)
            ymax = int(member.find('bndbox').find('ymax').text)

            xmin_corr = (xmin/wt)*self.width
            xmax_corr = (xmax/wt)*self.width
            ymin_corr = (ymin/ht)*self.height
            ymax_corr = (ymax/ht)*self.height

            boxes.append([xmin_corr, ymin_corr, xmax_corr, ymax_corr])

        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # getting the areas of the boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        # image_id
        # image_id = torch.tensor([idx])
        image_id = idx  # Changed by Rubens the tensor assign - 22/11/2023
        target["image_id"] = image_id

        if self.transforms:
            # print(f'target["boxes"]: {target["boxes"]}')
            # print(f'labels: {labels}')
            # print(f'labels.numpy(): {labels.numpy()}')
            # converting labels tensor to numpy array by Rubens
            labels = labels.numpy()

            sample = self.transforms(image = img_res,
                                     bboxes = target['boxes'],
                                     labels = labels)

            img_res = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        return img_res, target

    def __len__(self):
        return len(self.imgs)

# # check dataset
# dataset_to_check = WhiteMoldImagesDatasetFRCNN(local_image_dataset_faster_rcnn_train_path, WHITE_MOLD_IMAGE_DATASET_WIDTH, WHITE_MOLD_IMAGE_DATASET_WIDTH)
# print('length of dataset = ', len(dataset_to_check), '\n')

# # getting the image and target for a test index.  Feel free to change the index.
# img, target = dataset_to_check[78]
# print(img.shape, '\n',target)
    
def get_train_and_valid_datasets_and_dataloaders_faster_rcnn(parameters):

    # setting parameters value 
    width  = parameters['input']['input_dataset']['input_image_size']
    height = parameters['input']['input_dataset']['input_image_size']
    batch_size = parameters['model_faster_rcnn']['neural_network_model']['batch_size']
    number_workers = parameters['model_faster_rcnn']['neural_network_model']['number_workers']    

    # use our dataset and defined transformations
    dataset_train = WhiteMoldImagesDatasetFRCNN(
        parameters['processing']['image_dataset_folder_train'], 
        width=width, height=height,
        classes=parameters['neural_network_model']['classes'],
        transforms= get_transform(train=True))
    
    dataset_valid = WhiteMoldImagesDatasetFRCNN(
        parameters['processing']['image_dataset_folder_valid'], 
        width=width, height=height,
        classes=parameters['neural_network_model']['classes'],
        transforms= get_transform(train=False))
    
    # dataset_test  = get_test_datasets_and_dataloaders_faster_rcnn(parameters)
   
    # define training and validation data loaders
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=number_workers,
        collate_fn=utils.collate_fn)

    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=batch_size, shuffle=False, num_workers=number_workers,
        collate_fn=utils.collate_fn)
   
    return dataset_train, dataset_valid, dataloader_train, dataloader_valid


def get_test_datasets_and_dataloaders_faster_rcnn(parameters):

    # setting parameters value 
    width  = parameters['input']['input_dataset']['input_image_size']
    height = parameters['input']['input_dataset']['input_image_size']    
    batch_size = parameters['model_faster_rcnn']['neural_network_model']['batch_size']
    number_workers = parameters['model_faster_rcnn']['neural_network_model']['number_workers']    

    print(f"parameters['processing']['image_dataset_folder_test']: {parameters['processing']['image_dataset_folder_test']}")
    
    # use our dataset and defined transformations    
    dataset_test  = WhiteMoldImagesDatasetFRCNN(
        parameters['processing']['image_dataset_folder_test'], 
        width=width, height=height,
        classes=parameters['neural_network_model']['classes'],
        transforms= get_transform(train=False))

    # print(f'Test dataset size >> dentro do dataset.py: {len(dataset_test)}')
    # print(f'len(dataset_test): {len(dataset_test)}')        
    # print(f'len(dataset_test.classes): {len(dataset_test.classes)}')      
    
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, num_workers=number_workers,
        collate_fn=utils.collate_fn)

   
    # returning test datasets and dataloaders    
    return dataset_test, dataloader_test
