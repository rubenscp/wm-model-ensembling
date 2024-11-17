"""
Project: White Mold 
Description: This module implements methods and functions related to metrics used in the models
Author: Rubens de Castro Pereira
Advisors: 
    Prof. Dr. Hélio Pedrini - advisor at IC-Unicamp
    Prof. Dr. Díbio Leandro Borges - coadvisor at CIC-UnB
    Prof. Dr. Murillo Lobo Jr. - coadvisor at Embrapa Rice and Beans
Date: 16/02/2024
Version: 1.0
This implementation is based on the web article "Intersection over Union (IoU) in Object Detection & Segmentation", 
from LearnOpenCV, and it can be accessed by:
- https://colab.research.google.com/drive/1wxIVwYQ6RPXRiYhGqhYoixa53CQPLKSz?authuser=1#scrollTo=a183afbb
- https://learnopencv.com/intersection-over-union-iou-in-object-detection-and-segmentation/
"""

# Importing Python libraries
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

# from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision.ops import * 

# Importing python modules
from common.manage_log import *

# ###########################################
# Constants
# ###########################################
LINE_FEED = '\n'

class Metrics:

    def __init__(self, model=None, number_of_classes=None):
        self.model = model
        self.number_of_classes = number_of_classes

        # This list store details of all inferenced images (image_name, targets_list, preds_list)
        self.inferenced_images = []
        self.images_bounding_boxes = []
        self.result = None
        self.full_confusion_matrix = None
        self.full_confusion_matrix_normalized = None
        self.confusion_matrix = None
        self.confusion_matrix_normalized = None
        self.confusion_matrix_summary = {
            'number_of_images': 0,
            'number_of_bounding_boxes_target': 0,
            'number_of_bounding_boxes_predicted': 0,
            'number_of_bounding_boxes_predicted_with_target': 0,
            'number_of_ghost_predictions': 0,
            'number_of_undetected_objects': 0,
        }
        self.counts_per_class = []

        self.tp_per_class = []
        self.fp_per_class = []
        self.fn_per_class = []
        self.tn_per_class = []
        self.tp_model = 0
        self.fp_model = 0
        self.fn_model = 0
        self.tn_model = 0
        self.accuracy_per_class = []
        self.precision_per_class = []
        self.recall_per_class = []
        self.f1_score_per_class = []
        self.dice_per_class = []

       

    def to_string(self):
        text = LINE_FEED + 'Metrics' + LINE_FEED + LINE_FEED
        text += f'len(self.inferenced_images): {len(self.inferenced_images)}' + LINE_FEED
        # text += f'preds: {len(self.preds)}' + LINE_FEED
        # text += str(self.preds) + LINE_FEED + LINE_FEED
        # text += f'target: {len(self.target)}' + LINE_FEED
        # text += str(self.target) + LINE_FEED
        return text

    def set_target(self, target):
        self.target = target 
        
    def set_preds(self, preds):
        self.preds = preds
      
    def get_predicted_bounding_boxes(self, boxes, scores, labels):
            
        # creating predicted object 
        predicted = []

        # getting bounding boxes in format fot predicted object 
        predicted_boxes = []
        predicted_scores = []
        predicted_labels = []
        for i, box in enumerate(boxes):        
            predicted_boxes.append(box)
            predicted_scores.append(scores[i])
            predicted_labels.append(labels[i])

        # setting predicted dictionary 
        item = {
            "boxes": torch.tensor(predicted_boxes, dtype=torch.float),
            "scores": torch.tensor(predicted_scores, dtype=torch.float),
            "labels": torch.tensor(predicted_labels)
            }
        predicted.append(item)

        # returning predicted object
        return predicted 

    def set_details_of_inferenced_image(self, image_name, targets, preds):
        '''
            The sample below shows data format used here.
            inferenced_image: {
                'image_name': 'IMG_1853-bbox-1526946742.jpg', 
                'targets_list': [
                    {
                    'boxes': tensor([[ 85.,  36., 215., 263.]]), 
                    'labels': tensor([3])
                    }
                    ], 
                'preds_list': [
                    {
                    'boxes': tensor([[105.,  59., 193., 235.],	[103.,  54., 197., 242.]]), 
                    'scores': tensor([0.7952, 0.5049]), 
                    'labels': tensor([3, 2])
                    }
                    ]		
            }
        '''

        item = {
            "image_name": image_name,
            "targets_list": targets,
            "preds_list": preds
            }
        self.inferenced_images.append(item)           


    def add_image_bounding_box(self, image_name, 
        target_bbox=None, target_label=None, 
        pred_bbox=None, pred_label=None, pred_score=None, 
        threshold=None, iou_threshold=None, iou=None, status=None):

        target_bbox_aux = target_bbox.numpy().squeeze() if torch.is_tensor(target_bbox) else ''
        target_label_aux = target_label.numpy().squeeze() if torch.is_tensor(target_label) else ''
        pred_bbox_aux = pred_bbox.numpy().squeeze() if torch.is_tensor(pred_bbox) else ''
        pred_label_aux = pred_label.numpy().squeeze() if torch.is_tensor(pred_label) else ''
        pred_score_aux = pred_score.numpy().squeeze() if torch.is_tensor(pred_score) else ''
        iou_aux = iou.numpy().squeeze() if torch.is_tensor(iou) else ''

        # adding one bounding box of an image 
        image_bounding_box = []
        image_bounding_box.append(image_name)
        image_bounding_box.append(target_bbox_aux)
        image_bounding_box.append(target_label_aux)
        image_bounding_box.append(pred_bbox_aux)
        image_bounding_box.append(pred_label_aux)
        image_bounding_box.append(pred_score_aux)
        image_bounding_box.append(threshold)
        image_bounding_box.append(iou_aux)
        image_bounding_box.append(iou_threshold)
        image_bounding_box.append(status)        
        self.images_bounding_boxes.append(image_bounding_box)


    def compute_confusion_matrix(self, model_name, num_classes, threshold, iou_threshold, 
                                 metrics_folder, running_id_text, tested_folder):

        # Inspired from:
        # https://medium.com/@tenyks_blogger/multiclass-confusion-matrix-for-object-detection-6fc4b0135de6
        # https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/

        # Step 4 and 5: Convert bounding box coordinates and apply thresholding for multi-label classification
        # (Assuming the output format of your model is similar to the torchvision Faster R-CNN model)

        self.full_confusion_matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.full_confusion_matrix_normalized = np.zeros((num_classes + 1, num_classes + 1))
        self.confusion_matrix_normalized = np.zeros((num_classes, num_classes))
        undetected_objects_index = ghost_predictions_index = num_classes

        logging_info(f'')
        logging_info(f'Computing Confusion Matrix')
        logging_info(f'')
        
        self.images_bounding_boxes = []

        number_of_bounding_boxes_target = 0
        number_of_bounding_boxes_predicted = 0        
        number_of_bounding_boxes_predicted_with_target = 0
        number_of_ghost_predictions = 0
        number_of_undetected_objects = 0

        # processing all inferenced images 
        for inferenced_image in self.inferenced_images:

            logging_info(f' inferenced_image: {inferenced_image}')

            # getting target and predictions bounding boxes for evaluation             
            targets = inferenced_image["targets_list"]
            preds = inferenced_image["preds_list"]
            logging_info(f'')
            logging_info(f'targets: {targets}')
            logging_info(f'preds: {preds}')
            number_of_bounding_boxes_target += len(targets[0]['boxes'])    

            # evaluating predictions bounding boxes
            if  len(preds) == 0 or len(preds[0]['boxes']) == 0:
                #  Counting undetected objects
                number_of_undetected_objects += 1
                for target in targets:
                    for t_label in target['labels']:
                        self.full_confusion_matrix[undetected_objects_index, t_label] += 1
                        self.add_image_bounding_box(
                            inferenced_image["image_name"],
                            target_bbox=target['boxes'], 
                            target_label=target['labels'],
                            pred_bbox=None, pred_label=None, pred_score=None,
                            threshold=threshold, 
                            iou_threshold=iou_threshold, 
                            iou=0,
                            status='undetected-fn'
                        )
            else:
                for pred in preds:
                    for p_box, p_label, p_score in zip(pred['boxes'], pred['labels'], pred['scores']):
                        number_of_bounding_boxes_predicted += 1
                        for target in targets:
                            for t_box, t_label in zip(target['boxes'], target['labels']):

                                # compute IoU of two boxes
                                # Both sets of boxes are expected to be in (x1, y1, x2, y2)
                                iou = box_iou(p_box.unsqueeze(0), t_box.unsqueeze(0))

                                # evaluate IoU threshold and labels
                                if iou >= iou_threshold:
                                    number_of_bounding_boxes_predicted_with_target += 1                           
                                    if p_label == t_label:
                                        # True Positive 
                                        self.full_confusion_matrix[t_label, p_label] += 1
                                        status = 'target-tp'
                                    else:                        
                                        # False Positive 
                                        self.full_confusion_matrix[t_label, p_label] += 1
                                        status = 'target-fp'
                                else:
                                    # Counting ghost predictions   
                                    number_of_ghost_predictions += 1                         
                                    self.full_confusion_matrix[t_label, ghost_predictions_index] += 1
                                    status = 'err-pred-fp'

                                # adding bounding box to list of statistics
                                self.add_image_bounding_box(
                                    inferenced_image["image_name"],
                                    target_bbox=t_box, 
                                    target_label=t_label,
                                    pred_bbox=p_box,
                                    pred_label=p_label, 
                                    pred_score=p_score,
                                    threshold=threshold,
                                    iou_threshold=iou_threshold, 
                                    iou=iou,
                                    status=status
                                )


        # saving images and bounding boxes inferenced
        path_and_filename = os.path.join(
            metrics_folder,
            model_name + '_' + running_id_text + '_images_bounding_boxes.xlsx'
        )
        self.save_inferenced_images(path_and_filename)

        # split tested images according status of confusion matrix 
        self.split_tested_images(tested_folder)

        # getting just confusion matrix whithout the background, ghost predictions and undetected objects
        self.confusion_matrix = np.copy(self.full_confusion_matrix[1:-1,1:-1])

        # normalizing values summarizing by rows
        self.confusion_matrix_normalized = np.copy(self.confusion_matrix)
        sum_columns_aux_1 = np.sum(self.confusion_matrix_normalized,axis=0)
        row, col = self.confusion_matrix_normalized.shape
        for i in range(col):
            if sum_columns_aux_1[i] > 0:
                self.confusion_matrix_normalized[:,i] = self.confusion_matrix_normalized[:,i] / sum_columns_aux_1[i]

        # normalizing values summarizing by rows
        self.full_confusion_matrix_normalized = np.copy(self.full_confusion_matrix)
        sum_columns_aux_2 = np.sum(self.full_confusion_matrix_normalized,axis=0)
        row, col = self.full_confusion_matrix_normalized.shape
        for i in range(col):
            if sum_columns_aux_2[i] > 0:
                self.full_confusion_matrix_normalized[:,i] = self.full_confusion_matrix_normalized[:,i] / sum_columns_aux_2[i]

        # summary of confusion matrix        
        self.confusion_matrix_summary["number_of_images"] = len(self.inferenced_images)
        self.confusion_matrix_summary["number_of_bounding_boxes_target"] = number_of_bounding_boxes_target         
        self.confusion_matrix_summary["number_of_bounding_boxes_predicted"] = number_of_bounding_boxes_predicted
        self.confusion_matrix_summary["number_of_bounding_boxes_predicted_with_target"] = number_of_bounding_boxes_predicted_with_target
        self.confusion_matrix_summary["number_of_ghost_predictions"] = number_of_ghost_predictions
        self.confusion_matrix_summary["number_of_undetected_objects"] = number_of_undetected_objects

        # computing metrics from confusion matrix 
        self.compute_metrics_from_confusion_matrix()

    def confusion_matrix_to_string(self):
        logging_info(f'')
        logging_info(f'FULL CONFUSION MATRIX')
        logging_info(f'---------------------')
        logging_info(f'{LINE_FEED}{self.full_confusion_matrix}')
        logging_info(f'')
        logging_info(f'CONFUSION MATRIX')
        logging_info(f'---------------------------')
        logging_info(f'{LINE_FEED}{self.confusion_matrix}')
        logging_info(f'')
        logging_info(f'SUMMARY OF CONFUSION MATRIX')
        logging_info(f'---------------------------')
        logging_info(f'')
        logging_info(f'Total number of images               : ' + \
                     f'{self.confusion_matrix_summary["number_of_images"]}')
        logging_info(f'Bounding boxes target                : ' + \
            f'{self.confusion_matrix_summary["number_of_bounding_boxes_target"]}')
        logging_info(f'Bounding boxes predicted             : ' + \
            f'{self.confusion_matrix_summary["number_of_bounding_boxes_predicted"]}')
        logging_info(f'Bounding boxes predicted with target : ' + \
                     f'{self.confusion_matrix_summary["number_of_bounding_boxes_predicted_with_target"]}')
        logging_info(f'Number of ghost preditions           : ' + \
            f'{self.confusion_matrix_summary["number_of_ghost_predictions"]}')
        logging_info(f'Number of undetected objects         : ' + \
            f'{self.confusion_matrix_summary["number_of_undetected_objects"]}')
        logging_info(f'')
        
    # Extract from: 
    # 1) https://stackoverflow.com/questions/43697980/is-there-something-already-implemented-in-python-to-calculate-tp-tn-fp-and-fn
    # 2) https://stackoverflow.com/questions/75478099/how-to-extract-performance-metrics-from-confusion-matrix-for-multiclass-classifi?newreg=c9549e71afff4f13982ca151adedfbd5
    # 3) https://www.youtube.com/watch?v=FAr2GmWNbT0
    # 4) https://www.linkedin.com/pulse/yolov8-projects-1-metrics-loss-functions-data-formats-akbarnezhad/ --> EXCCELENT
    def not_used_compute_metrics_from_confusion_matrix_deactivated(self):
        """
        Obtain TP, FN FP, and TN for each class in the confusion matrix
        """

        # getting a copy of confusion matrix 
        confusion = np.copy(self.confusion_matrix)
        logging_info(f'confusion: {LINE_FEED}{confusion}')

        self.counts_per_class = []
                 
        # Iterate through classes and store the counts
        for i in range(confusion.shape[0]):
            tp = confusion[i, i]

            fn_mask = np.zeros(confusion.shape)
            fn_mask[i, :] = 1
            fn_mask[i, i] = 0
            fn = np.sum(np.multiply(confusion, fn_mask))

            fp_mask = np.zeros(confusion.shape)
            fp_mask[:, i] = 1
            fp_mask[i, i] = 0
            fp = np.sum(np.multiply(confusion, fp_mask))

            tn_mask = 1 - (fn_mask + fp_mask)
            tn_mask[i, i] = 0
            tn = np.sum(np.multiply(confusion, tn_mask))

            self.counts_per_class.append( {'Class': i,
                                            'TP': tp,
                                            'FN': fn,
                                            'FP': fp,
                                            'TN': tn} )

            # logging_info(f'counts_per_class: {self.counts_per_class}')
            # logging_info({'Class': i,
            #               'TP': tp,
            #               'FN': fn,
            #               'FP': fp,
            #               'TN': tn})

        # counting for model 
        self.tp_model = 0
        self.fn_model = 0
        self.fp_model = 0
        self.tn_model = 0
        for count in self.counts_per_class:
            self.tp_model += count['TP']
            self.fn_model += count['FN']
            self.fp_model += count['FP']
            self.tn_model += count['TN']

        # self.counts_of_model = {'Model': self.model,
        #                      'TP': tp_model,
        #                      'FN': fn_model,
        #                      'FP': fp_model,
        #                      'TN': tn_model}

        logging_info(f'TP / FN / FP / TN from confunsion matrix: ')
        for count in self.counts_per_class:
            logging_info(f'count {count}')
        
        logging_info(f'self.tp_model:{self.tp_model}')
        logging_info(f'self.fn_model:{self.fn_model}')
        logging_info(f'self.fp_model:{self.fp_model}')
        logging_info(f'self.tn_model:{self.tn_model}')

        # logging_info(f'counts_model: {self.counts_model}')
        # logging_info(f'counts_list: {counts_list}')             


    def compute_metrics_from_confusion_matrix(self):
        """
        Obtain TP, FN FP, and TN for each class in the confusion matrix
        """

        logging_info(f'full_confusion_matrix: {LINE_FEED}{self.full_confusion_matrix}')

        # initializing variable 
        self.tp_per_class = [0 for i in range(self.number_of_classes + 1)]
        self.fp_per_class = [0 for i in range(self.number_of_classes + 1)]
        self.fn_per_class = [0 for i in range(self.number_of_classes + 1)]
        self.tn_per_class = [0 for i in range(self.number_of_classes + 1)]
        self.tp_model = 0
        self.fp_model = 0
        self.fn_model = 0
        self.tn_model = 0
        
        cm_fp = self.full_confusion_matrix[1:-1, 1:]
        self.tp_per_class = cm_fp.diagonal()
        self.fp_per_class = cm_fp.sum(1) - self.tp_per_class
        
        cm_fn = self.full_confusion_matrix[1:, 1:-1]
        # DO NOT USE below because it consider all values of matrix, but it mus 
        # use just the last row to calculate the false negatives
        # self.fn_per_class = cm_fn.sum(0) - self.tp_per_class
        self.fn_per_class = cm_fn[-1:,].squeeze()

        # TN will be calculate just per class as every bounding boxes predicted of other classes,
        # all predicted bbox minus the bounding boxes predicted considering TP and FP of that class.
        logging_info(f'self.full_confusion_matrix: {self.full_confusion_matrix}')
        cm_tn = self.full_confusion_matrix[1:-1, 1:]

        # computing all values of confusion matrix for true negative 
        cm_tn_all = cm_tn.sum()

        logging_info(f'cm_tn: {cm_tn}')
        logging_info(f'number_of_classes: {self.number_of_classes}')
        for i in range(self.number_of_classes):
            # sum all elements of the row "i"
            cm_tn_row = cm_tn[i,].sum()
            # sum all elements of the column "i"
            cm_tn_col = cm_tn[:,i].sum()
            # getting the true positive of the element "i"
            element_tp = cm_tn[i,i]
            # compute the true negative value 
            self.tn_per_class[i] = cm_tn_all - cm_tn_row - cm_tn_col + element_tp

        # summarizing TP, FP, FN       
        self.tp_model = self.tp_per_class.sum()
        self.fp_model = self.fp_per_class.sum()
        self.fn_model = self.fn_per_class.sum()
        self.tn_model = 0 

        # computing metrics accuracy, precision, recall, f1-score and dice per classes
        self.compute_accuracy_per_class()
        self.compute_precision_per_class()
        self.compute_recall_per_class()
        self.compute_f1_score_per_class()
        self.compute_dice_per_class()

    def get_value_metric(self, metric):
        value = 0
        for count in self.counts_per_class:
            value += count[metric]
        return value

    # https://docs.kolena.io/metrics/accuracy/
    def get_model_accuracy(self):
        # accuracy = (self.tp_model + self.tn_model) /  \
        #            (self.tp_model + self.tn_model + self.fp_model + self.fn_model)
        accuracy = 0                   
        return accuracy

    # https://docs.kolena.io/metrics/precision/
    def get_model_precision(self):
        precision = (self.tp_model) /  \
                    (self.tp_model + self.fp_model)
        return precision

    # https://docs.kolena.io/metrics/recall/
    def get_model_recall(self):
        recall = (self.tp_model) /  \
                 (self.tp_model + self.fn_model)
        return recall

    # https://docs.kolena.io/metrics/f1-score/
    def get_model_f1_score(self):
        f1_score = (2.0 * self.get_model_precision() * self.get_model_recall()) /  \
                   (self.get_model_precision() + self.get_model_recall())
        return f1_score

    # https://docs.kolena.io/metrics/specificity/
    def get_model_specificity(self):
        specificity = (self.tn_model) /  \
                      (self.tn_model + self.fp_model)
        return specificity

    # https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    def get_model_dice(self):
        dice = (2 * self.tp_model) /  \
               ((2 * self.tp_model) + self.fp_model + self.fn_model)
        return dice

    # compute accuracy per classes and for the model
    def compute_accuracy_per_class(self):
        # initializing variable 
        self.accuracy_per_class = [0 for i in range(self.number_of_classes + 1)]

        # compute precision for each class
        for i in range(len(self.tp_per_class)):
            denominator = self.tp_per_class[i] + self.tn_per_class[i] + \
                          self.fp_per_class[i] + self.fn_per_class[i]
            if denominator > 0:
                self.accuracy_per_class[i] = (self.tp_per_class[i] + self.tn_per_class[i]) / denominator

    # compute precision per classes and for the model
    def compute_precision_per_class(self):
        # initializing variable 
        self.precision_per_class = [0 for i in range(self.number_of_classes + 1)]

        # compute precision for each class
        for i in range(len(self.tp_per_class)):
            denominator = self.tp_per_class[i] + self.fp_per_class[i]
            if denominator > 0:
                self.precision_per_class[i] = (self.tp_per_class[i]) / denominator

        # compute precision of the model
        self.precision_per_class[i+1] = np.sum(self.precision_per_class) / self.number_of_classes

    # compute recall per classes and for the model
    def compute_recall_per_class(self):
        # initializing variable 
        self.recall_per_class = [0 for i in range(self.number_of_classes + 1)]

        # compute recall for each class
        for i in range(len(self.tp_per_class)):
            denominator = self.tp_per_class[i] + self.fn_per_class[i]
            if denominator > 0:
                self.recall_per_class[i] = (self.tp_per_class[i]) / denominator

        # compute recall of the model
        self.recall_per_class[i+1] = np.sum(self.recall_per_class) / self.number_of_classes

    # compute f1-score per classes and for the model
    def compute_f1_score_per_class(self):
        # initializing variable 
        self.f1_score_per_class = [0 for i in range(self.number_of_classes + 1)]

        # compute f1-score for each class
        for i in range(len(self.tp_per_class)):
            denominator = self.precision_per_class[i] + self.recall_per_class[i]
            if denominator > 0:
                self.f1_score_per_class[i] = \
                (2 * self.precision_per_class[i] * self.recall_per_class[i]) / denominator

        # compute f1-score of the model
        self.f1_score_per_class[i+1] = np.sum(self.f1_score_per_class) / self.number_of_classes

    # compute dice per classes and for the model
    def compute_dice_per_class(self):
        # initializing variable 
        self.dice_per_class = [0 for i in range(self.number_of_classes + 1)]

        # compute dice for each class
        for i in range(len(self.tp_per_class)):
            denominator = (2 * self.tp_per_class[i]) + self.fp_per_class[i] + self.fn_per_class[i]
            if denominator > 0:
                self.dice_per_class[i] = (2 * self.tp_per_class[i]) / denominator

        # compute dice of the model
        self.dice_per_class[i+1] = np.sum(self.dice_per_class) / self.number_of_classes


    def save_inferenced_images(self, path_and_filename):

        # preparing columns name to list
        column_names = [
            'image name',
            'target bbox',           
            'target label',
            'predict bbox',           
            'predict label',
            'predict score',
            'threshold',
            'iou',
            'iou threshold',
            'status',
        ]

        # creating dataframe from list 
        df = pd.DataFrame(self.images_bounding_boxes, columns=column_names)

        # writing excel file from dataframe
        df.to_excel(path_and_filename, sheet_name='bounding_boxes', index=False)


    def split_tested_images(self, tested_folder):

        # creating specific folders 
        target_tp = os.path.join(tested_folder, 'target-tp')
        Utils.create_directory(target_tp)

        target_fp = os.path.join(tested_folder, 'target-fp')
        Utils.create_directory(target_fp)

        err_pred_fp = os.path.join(tested_folder, 'err-pred-fp')
        Utils.create_directory(err_pred_fp)

        undetected_fn = os.path.join(tested_folder, 'undetected-fn')
        Utils.create_directory(undetected_fn)

        for image_bounding_box in self.images_bounding_boxes:
            filename, extension = Utils.get_filename_and_extension(image_bounding_box[0])
            filename = filename + '_predicted.' + extension
            input_path = tested_folder
            output_path = os.path.join(tested_folder, image_bounding_box[9])
            Utils.copy_file_same_name(filename, input_path, output_path)

    # def split_tested_images(self, tested_folder):

    #     logging_info(f'split_tested_images: {tested_folder}')

    #     # creating specific folders 
    #     target_tp = os.path.join(tested_folder, 'target-tp')
    #     Utils.create_directory(target_tp)

    #     target_fp = os.path.join(tested_folder, 'target-fp')
    #     Utils.create_directory(target_fp)

    #     err_pred_fp = os.path.join(tested_folder, 'err-pred-fp')
    #     Utils.create_directory(err_pred_fp)

    #     undetected_fn = os.path.join(tested_folder, 'undetected-fn')
    #     Utils.create_directory(undetected_fn)

    #     for image_bounding_box in self.images_bounding_boxes:

    #         logging_info(f'image_bounding_box: {image_bounding_box}')

    #         filename, extension = Utils.get_filename_and_extension(image_bounding_box[0])
    #         logging_info(f'filename: {filename}')
    #         logging_info(f'extension: {extension}')
    #         filename = filename + '_predicted.' + extension
    #         logging_info(f'filename: {filename}')
    #         input_path = tested_folder
    #         output_path = os.path.join(tested_folder, image_bounding_box[9])
            
    #         logging_info(f'filename: {filename}')
    #         logging_info(f'input_path: {input_path}')
    #         logging_info(f'output_path: {output_path}')

    #         Utils.copy_file_same_name(filename, input_path, output_path)
        
    def compute_metrics_sklearn(self):
        logging_info(f'Computing metrics using Sklearn')

        y_all_targets = []
        all_preds = []
        for inferenced_image in self.inferenced_images:
            logging_info(f' inferenced_image: {inferenced_image}')
            if len(inferenced_image['preds_list'][0]['boxes']) > 0:
                all_targets.append(inferenced_image["targets_list"])
                all_preds.append(inferenced_image["preds_list"])

        logging_info(f'')
        logging_info(f'all_targets: {all_targets}')
        logging_info(f'all_preds: {all_preds}')

        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='macro')
        recall = recall_score(all_targets, all_preds, average='macro')
        f1 = f1_score(all_targets, all_preds, average='macro')

        logging_info(f'accuracy: {accuracy}')
        logging_info(f'precision: {precision}')
        logging_info(f'recall: {recall}')
        logging_info(f'f1: {f1}')


    def remove_path_from_image_filename(self):
        logging_info(f'')
        logging_info(f'remove_path_from_image_filename')                
        for image_bounding_box in self.images_bounding_boxes:
            logging_info(f'image_bounding_box[0]: {image_bounding_box[0]}')
            path, filename_with_extension, filename, extension = Utils.get_filename(image_bounding_box[0])
            logging_info(f'path: {path}')
            logging_info(f'filename_with_extension: {filename_with_extension}')
            logging_info(f'filename: {filename}')
            logging_info(f'extension: {extension}')            
            image_bounding_box[0] = filename_with_extension

