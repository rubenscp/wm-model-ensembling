import torchvision
import os
import torch
from torch.utils.data import DataLoader

class CocoDetection(torchvision.datasets.CocoDetection):

    def __init__(self, img_folder, processor, dataset_type):
        ann_file = os.path.join(img_folder, 'custom_' + dataset_type + '.json')      
        # ann_file = os.path.join(img_folder, "custom_train.json" if train else "custom_val.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        # feel free to add data augmentation here before passing them to the next step
        img, target = super(CocoDetection, self).__getitem__(idx)
        
        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]

        target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target

def collate_fn(batch, processor):
    pixel_values = [item[0] for item in batch]
    encoding = processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = labels
    return batch

def convert_normalized_bbox_to_original_bbox(normalized_boxes, normalized_size, original_size):
    # normalized_boxes: [x, y, w, h]
    # original_size: (width, height)
    
    # getting the original and normalized sizes
    original_width, original_height = original_size
    normalized_width, normalized_height = normalized_size

    # converting normalized boxes to original boxes
    original_boxes = []
    for normalized_box in normalized_boxes:

        # getting max and min values of the box points
        x_min_norm, y_min_norm, width_norm, height_norm = normalized_box
        x_min = x_min_norm * normalized_width
        y_min = y_min_norm * normalized_height      
        x_max = x_min + (width_norm * normalized_width)
        y_max = y_min + (height_norm * normalized_height)

        # computing the scaling factor
        scale_x = original_width / normalized_width
        scale_y = original_height / normalized_height

        # scaling the box points
        x_min = x_min * scale_x
        y_min = y_min * scale_y
        x_max = x_max * scale_x
        y_max = y_max * scale_y

        # appending the original box to the list
        original_boxes.append([x_min.item(), y_min.item(), x_max.item(), y_max.item()])

    # returning original boxes in format [x_min, y_min, x_max, y_max] according the original size
    return torch.tensor(original_boxes)