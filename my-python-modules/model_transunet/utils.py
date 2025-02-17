import os
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk

from model_transunet.wm_utils import WM_Utils

from torchvision.ops import masks_to_boxes

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):     
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list


def test_single_image_white_mold(image, label, net, classes, patch_size=[256, 256], 
                       test_save_path=None, case=None, 
                       z_spacing=1, dataset=None, original_image=None, 
                       annotated_bounding_boxes=None, inference_metric=None,
                       tested_folder=None, test_image_folder=None, all_predictions=None):

    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()

    # creating predictions dictionary of all images 
    # image_predictions = {}

    print(f'test_single_image_white_mold')
    print(f'001 - len(image.shape): {len(image.shape)}')
    print(f'002 - image.shape: {image.shape}')
    # if dataset == 'WhiteMold':
    #     image = image.squeeze(0).squeeze(0).cpu().detach().numpy() # removing two dimensions 
    #     print(f'003 - len(image.shape): {len(image.shape)}')
    #     print(f'004 - image.shape: {image.shape}')
    #     image = torch.from_numpy(image.astype(np.float32)).permute(2,0,1)  # Convert (H, W, C) to (C, H, W)
    #     print(f'005 - len(image.shape): {len(image.shape)}')
    #     print(f'006 - image.shape: {image.shape}')
    #     label = label.squeeze(0).cpu().detach().numpy()        
    # else:
    #     image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()

    # print(f'007 - len(image.shape): {len(image.shape)}')
    # print(f'008 - image.shape: {image.shape}')

     # Check if the image is grayscale (single channel) or RGB (three channels)
    # if image.shape[1] == 1:  # Grayscale
    #     image = image.squeeze(0).cpu().detach().numpy()  # Squeeze batch dimension
    #     label = label.squeeze(0).cpu().detach().numpy()
    # elif image.shape[1] == 3:  # RGB
    #     image = image.permute(1, 2, 3, 0).cpu().detach().numpy()  # Move channel to last dimension
    #     label = label.squeeze(0).cpu().detach().numpy()
    # else:
    #     raise ValueError("Unsupported number of channels. Expected 1 or 3 channels for grayscale or RGB.")

    
    if len(image.shape) == 3:
        print(f'')
        print(f'inferencing image.shape equal 3')
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            print(f'input.shape: {input.shape}')

            net.eval()
            with torch.no_grad():
                outputs = net(input)
                print(f'outputs.shape: {outputs.shape}')
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        print(f'')
        print(f'inferencing image.shape not equal 3')
        # image = image.numpy()
        print(f'len(image.shape): {len(image.shape)}')
        print(f'image.shape 01: {image.shape}')
        # print(f'image: {image}')

        # input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        # input = torch.from_numpy(image).float().cuda()
        # print(f'input.shape: {input.shape}')

        x, y = image.shape[2], image.shape[3]
        print(f'x:{x}  y:{y}')
        if x != patch_size[0] or y != patch_size[1]:
            scale = (1,) + (1,) + (patch_size[0] / x, patch_size[1] / y)
            print(f'scale: {scale}')
            print(f'image.shape 02: {image.shape}')
            image = zoom(image, scale, order=3)            
        
        print(f'image.shape 03: {image.shape}')
        # input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        input = torch.from_numpy(image).float().cuda()
        print(f'input.shape: {input.shape}')

        net.eval()
        with torch.no_grad():
            outputs = net(input)
            print(f'outputs.shape: {outputs.shape}')
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            print(f'out.shape: {out.shape}')
            out = out.cpu().detach().numpy()
            if x != patch_size[0] or y != patch_size[1]:
                prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            else:
                prediction = out
            # prediction = pred.cpu().detach().numpy()
            print(f'prediction.shape: {prediction.shape}')
            print(f'prediction: {prediction}')           

    metric_list = []
    # for i in range(1, classes):
    #     metric_list.append(calculate_metric_percase(prediction == i, label == i))

    # processing inferenced images by model  
    if dataset == 'WhiteMold':
        # setting target and predicted bounding boxes for metrics 
        new_targets = []
        original_target_boxes, original_target_labels = get_original_boxes(annotated_bounding_boxes)
        original_target_boxes = torch.tensor(original_target_boxes)
        original_target_labels = torch.tensor(original_target_labels)
        print(f'original_target_boxes: {original_target_boxes}')
        print(f'original_target_labels: {original_target_labels}')
        item_target = {
            "boxes": original_target_boxes,
            "labels": original_target_labels
            }
        new_targets.append(item_target)

        # getting the bounding boxes of the model predictions
        # creating bounding boxes list for each class
        # bounding_boxes = []
        print(f'classes: {classes}')
        print(f'prediction.shape: {prediction.shape}')
        # predicted_boxes, predicted_scores, predicted_labels = \
        #     get_bounding_boxes_from_predicted_mask_grayscale(prediction[0], classes)
        predicted_boxes, predicted_scores, predicted_labels = \
            get_bounding_boxes_from_predicted_mask_rgb(prediction, classes)
        predicted_boxes = torch.tensor(predicted_boxes)
        predicted_scores = torch.tensor(predicted_scores)
        predicted_labels = torch.tensor(predicted_labels)
        print(f'predicted_boxes: {predicted_boxes}')
        print(f'predicted_scores: {predicted_scores}')
        print(f'predicted_labels: {predicted_labels}')

        new_predicteds = []
        # print(f'test_single_volume - prediction: {prediction}')
        item_predicted = {
            "boxes": predicted_boxes,
            "scores": predicted_scores, 
            "labels": predicted_labels,
            }
        new_predicteds.append(item_predicted)
        print(f'new_targets: {new_targets}')
        print(f'new_predicteds: {new_predicteds}')            

        # setting target and predicted bounding boxes for metrics
        inference_metric.set_details_of_inferenced_image(
            case, new_targets, new_predicteds)

        # setting prediction in all predictions dictionary
        key = case
        value = (new_targets, new_predicteds)
        all_predictions[key] = value 

        # reading original image used for inference (test)
        path_and_original_rgb_image_filename = os.path.join(test_image_folder, case)
        print(f'path_and_original_rgb_image_filename: {path_and_original_rgb_image_filename}')   
        original_rgb_image = WM_Utils.read_image(case, test_image_folder)
        print(f'original_rgb_image.shape: {original_rgb_image.shape}')
        
        # copying original rgb image to tested folder 
        WM_Utils.copy_file_same_name(case, test_image_folder, tested_folder)

        # path_and_original_image_filename = os.path.join(tested_folder, case)
        # print(f'path_and_original_image_filename: {path_and_original_image_filename}')
        # print(f'original_image.shape: {original_image.shape}')
        # print(f'original_image.squeeze(0).shape: {original_image.squeeze(0).shape}')
        # WM_Utils.save_image(path_and_original_image_filename, original_image.squeeze(0))
        # WM_Utils.save_to_excel(path_and_original_image_filename.replace('.jpeg', '.xlsx'), 'image', image.squeeze(0))

        # create new inferenced image with its bounding boxes 
        predicted_image = draw_bounding_boxes_into_image(original_rgb_image, predicted_boxes, predicted_labels)

        # saving inferenced (tested) images in the working folder 
        path_and_predicted_image_filename = os.path.join(tested_folder, case.replace('.jpg', '') + "_predicted.jpg")
        WM_Utils.save_image(path_and_predicted_image_filename, predicted_image)
        # WM_Utils.save_to_excel(path_and_predicted_image_filename.replace('.jpg', '.xlsx'), 'prediction', prediction.squeeze(0))

        
        
    if test_save_path is not None:
        if dataset == 'WhiteMoldxxxxxxxx':

            path_and_filename = os.path.join(test_save_path, case + "_image.jpeg")
            WM_Utils.save_image(path_and_filename, image.squeeze(0))
            WM_Utils.save_to_excel(path_and_filename.replace('.jpeg', '.xlsx'), 'image', image.squeeze(0))
            path_and_filename = os.path.join(test_save_path, case + "_prediction.jpeg")
            WM_Utils.save_image(path_and_filename, prediction.squeeze(0))
            WM_Utils.save_to_excel(path_and_filename.replace('.jpeg', '.xlsx'), 'prediction', prediction.squeeze(0))
            
            path_and_label_filename = os.path.join(test_save_path, case + "_label.jpeg")
            WM_Utils.save_image(path_and_label_filename, label.squeeze(0))
            WM_Utils.save_to_excel(path_and_label_filename.replace('.jpeg', '.xlsx'), 'label', label.squeeze(0))

            # adjusting the shape of the image and prediction
            image = image.squeeze(0)
            prediction = prediction.squeeze(0)
            original_image = original_image.squeeze(0).numpy()
            print(f'image.shape: {image.shape}')
            print(f'prediction.shape: {prediction.shape}')
            print(f'original_image.shape: {original_image.shape}')
            print(f'annotated_bounding_boxes: {annotated_bounding_boxes}')  
            print(f'len(annotated_bounding_boxes): {len(annotated_bounding_boxes)}')  
            print(f'annotated_bounding_boxes[0]: {annotated_bounding_boxes[0]}')  


            # creating bounding boxes list for each class
            bounding_boxes = []
            colors = [(0, 0, 255), (255, 255, 0), (255, 0, 0),
                      (0, 255, 255), (0, 255, 0)]
            print(f'classes: {classes}')            
            for i in range(1, classes):
                # get bounding boxes from mask prediction
                label_value = i
                bounding_boxes_per_class = get_bounding_boxes_from_predicted_mask(prediction, label_value)
                
                ###############################################################

                # np_seg = np.array(prediction)
                # print(f'np_seg: {np_seg}')
                # segmentation = np.where(np_seg == label_value)
                # print(f'len(segmentation): {len(segmentation)}')
                # print(f'segmentatio[0]: {segmentation[0]}')
                # print(f'segmentatio[1]: {segmentation[1]}')
                # print(f'segmentatio[2]: {segmentation[2]}')                
                
                # exit()

                # print(f'segmentation: {segmentation}')
                # # print(f'segmentation.shape: {segmentation.shape}')
                # segmentation_file = os.path.join(test_save_path, case + '_segmentation_' + str(label_value) + '.xlsx')
                # WM_Utils.save_to_excel(segmentation_file, 'label', segmentation)
                # print(f'segmentation_file: {segmentation_file}')
                # exit()

                ###############################################################

                print(f'bounding_boxes_per_class: {bounding_boxes_per_class}')
                print(f'len(bounding_boxes_per_class): {len(bounding_boxes_per_class)}')
                if len(bounding_boxes_per_class) > 0:
                    new_image_with_bbox_drawn = original_image
                    for bounding_box_per_class in bounding_boxes_per_class:
                        print(f'bounding_box_per_class: {bounding_box_per_class}')
                        # creating new image to check the new coordinates of bounding box                    
                        bgrBoxColor = colors[i-1]
                        thickness = 1
                        new_image_with_bbox_drawn = WM_Utils.draw_bounding_box(
                            new_image_with_bbox_drawn,
                            bounding_box_per_class['ymin'], bounding_box_per_class['xmin'],
                            bounding_box_per_class['ymax'], bounding_box_per_class['xmax'],
                            bgrBoxColor, thickness, str(label_value))
                    path_and_filename = os.path.join(test_save_path, case + "_image_predicted.jpeg")
                    WM_Utils.save_image(path_and_filename, new_image_with_bbox_drawn)
            
        elif dataset == 'Synapse':
            img_itk = sitk.GetImageFromArray(image.astype(np.float32))
            prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
            lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
            img_itk.SetSpacing((1, 1, z_spacing))
            prd_itk.SetSpacing((1, 1, z_spacing))
            lab_itk.SetSpacing((1, 1, z_spacing))
            sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
            sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
            sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")

    # print(f'all_predictions: {len(all_predictions)} - {all_predictions}')

    return metric_list

def get_bounding_boxes_from_predicted_mask_rgb(prediction_image, classes):

    # print(f'get_bounding_boxes_from_predicted_mask - prediction_image.shape: {prediction_image.shape}')

    # initializing results 
    predicted_boxes = []
    predicted_scores = []
    predicted_labels = []

    # get predicted bounding boxes for all classes
    for label_value in range(1, classes):
        # the classes with id 2 and 4 aren't considered in this experiment
        # if label_value == 2 or 4:
        #     continue

        # creating a numpy array from the image
        print(f'prediction_image.shape: {prediction_image.shape}')
        np_seg = np.array(prediction_image)
        segmentation = np.where(np_seg == label_value)
        print(f'segmentation: {segmentation}')
        # print(f'get_bounding_boxes_from_predicted_mask - segmentation: {segmentation}')

        # bounding box
        bounding_boxes = []
        if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
            x_min = int(np.min(segmentation[1]))
            x_max = int(np.max(segmentation[1]))
            y_min = int(np.min(segmentation[0]))
            y_max = int(np.max(segmentation[0]))

            # print(f'x_min: {x_min} y_min: {y_min} x_max: {x_max} y_max: {y_max}')

            bbox_item = [x_min, y_min, x_max, y_max]

            predicted_boxes.append(bbox_item)
            predicted_scores.append(0.8)
            predicted_labels.append(label_value)
    
    return predicted_boxes, predicted_scores, predicted_labels

def get_bounding_boxes_from_predicted_mask_grayscale(prediction_image, classes):

    # print(f'get_bounding_boxes_from_predicted_mask - prediction_image.shape: {prediction_image.shape}')

    # initializing results 
    predicted_boxes = []
    predicted_scores = []
    predicted_labels = []

    # get predicted bounding boxes for all classes
    for label_value in range(1, classes):
        # the classes with id 2 and 4 aren't considered in this experiment
        # if label_value == 2 or 4:
        #     continue

        # creating a numpy array from the image
        print(f'prediction_image.shape: {prediction_image.shape}')
        np_seg = np.array(prediction_image)
        segmentation = np.where(np_seg == label_value)
        print(f'segmentation: {segmentation}')
        # print(f'get_bounding_boxes_from_predicted_mask - segmentation: {segmentation}')

        # bounding box
        bounding_boxes = []
        if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
            x_min = int(np.min(segmentation[1]))
            x_max = int(np.max(segmentation[1]))
            y_min = int(np.min(segmentation[0]))
            y_max = int(np.max(segmentation[0]))

            # print(f'x_min: {x_min} y_min: {y_min} x_max: {x_max} y_max: {y_max}')

            bbox_item = [x_min, y_min, x_max, y_max]

            predicted_boxes.append(bbox_item)
            predicted_scores.append(1)
            predicted_labels.append(label_value)
    
    return predicted_boxes, predicted_scores, predicted_labels

def get_bounding_boxes_from_predicted_mask_xxxxxxxxxxxx(prediction, label_value):

    print(f'get_bounding_boxes_from_predicted_mask - prediction.shape: {prediction.shape}')

    # creating a numpy array from the image
    np_seg = np.array(prediction)
    segmentation = np.where(np_seg == label_value)
    print(f'get_bounding_boxes_from_predicted_mask - segmentation: {segmentation}')

    # bounding box
    # bbox = 0, 0, 0, 0
    bounding_boxes = []
    if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
        x_min = int(np.min(segmentation[2]))
        x_max = int(np.max(segmentation[2]))
        y_min = int(np.min(segmentation[1]))
        y_max = int(np.max(segmentation[1]))

        print(f'x_min: {x_min} y_min: {y_min} x_max: {x_max} y_max: {y_max}')

        bounding_box = {}
        bounding_box['xmin'] = x_min
        bounding_box['ymin'] = y_min
        bounding_box['xmax'] = x_max
        bounding_box['ymax'] = y_max
        print(f'get_bounding_boxes_from_predicted_mask - bounding_box: {bounding_box}')

        bounding_boxes.append(bounding_box)    

    return bounding_boxes

# def get_bounding_boxes_from_predicted_mask_3(prediction, label_value):

#     # getting bounding boxes from masks
#     bounding_boxes = masks_to_boxes(prediction)
#     print(f'bounding_boxes: {bounding_boxes}')
#     if len(bounding_boxes) > 1:
#         print(f'varios bounding boxes na mesma imagem ')

#     return bounding_boxes


def get_original_boxes(bounding_boxes):
    # print(f'get_original_boxes - bounding_boxes: {bounding_boxes}')

    original_target_boxes = []
    original_target_labels = []
    for bounding_box in bounding_boxes:
        # print(f'get_original_boxes - bounding_box: {bounding_box}')
        # print(f'get_original_boxes - bounding_box[bndbox]: {bounding_box["bndbox"]}')
        # print(f'get_original_boxes - bounding_box[class_name]: {bounding_box["class_name"]}')
        bbox_item = [
            bounding_box['bndbox']['xmin'].item(), 
            bounding_box['bndbox']['ymin'].item(), 
            bounding_box['bndbox']['xmax'].item(), 
            bounding_box['bndbox']['ymax'].item(), 
            ]
        original_target_boxes.append(bbox_item)

        class_name = bounding_box['class_name']
        class_name = ''.join(class_name) # this instruction transforms an string array into string 
        if class_name == 'Apothecium':
            class_id = 1
        elif class_name == 'Imature Sclerotium':
            class_id = 2
        elif class_name == 'Mature Sclerotium':
            class_id = 3
        elif class_name == 'White Mold':
            class_id = 4
        elif class_name == 'Imature Sclerotium and White Mold':
            class_id = 5
        else:
            class_id = -1

        print(f'class_id: {class_id}')
        original_target_labels.append(class_id)

    # print(f'original_target_boxes: {original_target_boxes}')
    # print(f'original_target_labels: {original_target_labels}')

    # returning original boxes
    return original_target_boxes, original_target_labels


def draw_bounding_boxes_into_image(original_rgb_image, predicted_boxes, predicted_labels):

    # create a copy of original image 
    predicted_image = original_rgb_image.copy()

    # setting class colors for bounding boxes
    colors = [(0, 0, 255), (255, 255, 0), (255, 0, 0), (0, 255, 255), (0, 255, 0)]

    for predicted_box, predicted_label in zip(predicted_boxes, predicted_labels):
        predicted_box = predicted_box.numpy()
        predicted_label = predicted_label.numpy()

        # creating new image to check the new coordinates of bounding box                    
        bgrBoxColor = colors[predicted_label - 1]
        thickness = 1
        predicted_image = WM_Utils.draw_bounding_box(
            predicted_image,
            predicted_box[1], predicted_box[0],
            predicted_box[3], predicted_box[2],
            bgrBoxColor, thickness, str(predicted_label))

    return predicted_image

