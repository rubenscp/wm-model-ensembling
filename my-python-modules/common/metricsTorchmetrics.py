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
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# from torchvision import ops
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics.detection import MeanAveragePrecision
# from torchmetrics.detection import Accuracy
from torchmetrics.classification import MulticlassPrecision 
# from torchmetrics.detection import Recall 
# from torchmetrics.detection import F1Score
# from torchmetrics.detection import Dice
# from torchmetrics.detection import Specificity

# import seaborn as sns

# from torchvision.models.detection import box_iou
# from torchvision.ops import * 

# Importing python modules
from common.manage_log import *

# ###########################################
# Constants
# ###########################################
LINE_FEED = '\n'

class MetricsTorchmetrics:

    def __init__(self, model=None, number_of_classes=0):
        self.model = model
        self.preds = []
        self.targets = []
        self.number_of_classes = number_of_classes
        self.confusion_matrix = None
        self.mean_average_precision = None
        self.precision = None

        # self.inferenced_images = []
        # self.images_bounding_boxes = []
        # self.result = None
        # self.full_confusion_matrix = None
        # self.full_confusion_matrix_normalized = None
        # self.confusion_matrix = None
        # self.confusion_matrix_normalized = None
        # self.confusion_matrix_summary = {
        #     'number_of_images': 0,
        #     'number_of_bounding_boxes_target': 0,
        #     'number_of_bounding_boxes_predicted': 0,
        #     'number_of_bounding_boxes_predicted_with_target': 0,
        #     'number_of_ghost_predictions': 0,
        #     'number_of_undetected_objects': 0,
        # }
        # self.counts_per_class = []

        # self.tp_per_classes = []
        # self.fp_per_classes = []
        # self.fn_per_classes = []
        # self.tn_per_classes = []
        # self.tp_model = 0
        # self.fp_model = 0
        # self.fn_model = 0
        # self.tn_model = 0

    def to_string(self):
        text = LINE_FEED + 'Metrics' + LINE_FEED + LINE_FEED
        text += f'len(self.inferenced_images): {len(self.inferenced_images)}' + LINE_FEED
        # text += f'preds: {len(self.preds)}' + LINE_FEED
        # text += str(self.preds) + LINE_FEED + LINE_FEED
        # text += f'target: {len(self.target)}' + LINE_FEED
        # text += str(self.target) + LINE_FEED
        return text

    def add_targets(self, targets):
        # targets = torch.Tensor(targets)
        self.targets.append(targets)
        # logging_info(f'add_targets 1 targets: {targets}')
        # logging_info(f'add_targets 2 self.targets: {self.targets}')
        
    def add_preds(self, preds):
        # preds = torch.Tensor(preds)
        self.preds.append(preds)
        # logging_info(f'add_preds 1 preds: {preds}')
        # logging_info(f'add_preds 2 self.preds: {self.preds}')

    def compute_confusion_matrix(self):
        logging_info(f'Computing Confusion Matrix')

        target = torch.Tensor([2, 1, 0, 0])
        preds = torch.Tensor([[0.16, 0.26, 0.58],
                        [0.22, 0.61, 0.17],
                        [0.71, 0.09, 0.20],
                        [0.05, 0.82, 0.13]])
        logging_info(f'target: {target}')
        logging_info(f'target.shape: {target.shape}')
        logging_info(f'preds: {preds}')
        logging_info(f'preds.shape: {preds.shape}')
        metric = MulticlassConfusionMatrix(num_classes=3)
        cm = metric(preds, target)
        logging_info(f'cm: {cm}')

        logging_info(f'MultilabelConfusionMatrix')

        from torch import tensor
        from torchmetrics.classification import MultilabelConfusionMatrix
        # target = tensor([[0, 1, 0], [1, 0, 1]])
        target = tensor([[0], [1], [0], [1]])
        preds = tensor([[0, 0, 1], [1, 0, 1], [1], []])
        logging_info(f'target: {target}')
        logging_info(f'target.shape: {target.shape}')
        logging_info(f'preds: {preds}')
        logging_info(f'preds.shape: {preds.shape}')
        metric = MultilabelConfusionMatrix(num_labels=3)
        cml = metric(preds, target)
        logging_info(f'cml: {cml}')
        
        exit()

        target_labels = []
        preds_labels = []
        for item in self.targets:
            target_labels.append(item['labels'].numpy())
        for item in self.preds:
            preds_labels.append(item['labels'].numpy())
        logging_info(f'target_labels: {target_labels}')
        logging_info(f'preds_labels: {preds_labels}')
                
        metric = MulticlassConfusionMatrix(num_classes=self.number_of_classes)
        # logging_info(f'self.targets: {self.targets}')
        # logging_info(f'self.preds: {self.preds}')
        # self.confusion_matrix(self.preds, self.targets)
        self.confusion_matrix = metric(preds_labels, target_labels)
        logging_info(f'self.confusion_matrix: {self.confusion_matrix}')

        exit()


    def compute_mean_average_precision(self):
        logging_info(f'Computing Mean Average Precision')
        metric = MeanAveragePrecision()
        logging_info(f'self.targets: {self.targets}')
        logging_info(f'self.preds: {self.preds}')
        metric.update(self.preds, self.targets)
        self.mean_average_precision = metric.compute()
        logging_info(f'self.mean_average_precision: {self.mean_average_precision}')

    def compute_precision(self):
        logging_info(f'Computing Multiclass Precision')
        metric = MulticlassPrecision(num_classes=self.number_of_classes)
        logging_info(f'self.targets: {self.targets}')
        logging_info(f'self.preds: {self.preds}')
        self.precision = metric(self.preds, self.targets)
        logging_info(f'self.precision: {self.precision}')

    # def compute_precision(self):
    #     logging_info(f'Computing Precision')
    #     precision = Precision(task='multiclass', average='macro', num_classes=self.number_of_classes)
    #     logging_info(f'self.targets: {self.targets}')
    #     logging_info(f'self.preds: {self.preds}')
    #     self.precision = precision(self.preds, self.targets)        
    #     logging_info(f'self.precision: {self.precision}')




    # =============================================================================


    # def get_target_size(self):
    #     # counting number of bounding boxes
    #     count = 0
    #     for target in self.target:
    #         count += len(target['labels'])

    #     # returning number of bounding boxes target
    #     return count
        

    # def get_preds_size(self):
    #     # counting number of bounding boxes
    #     count = 0
    #     for pred in self.preds:
    #         count += len(pred['labels'])

    #     # returning number of bounding boxes predicted
    #     return count

    # def get_predicted_bounding_boxes(self, boxes, scores, labels):
            
    #     # creating predicted object 
    #     predicted = []

    #     # getting bounding boxes in format fot predicted object 
    #     predicted_boxes = []
    #     predicted_scores = []
    #     predicted_labels = []
    #     for i, box in enumerate(boxes):        
    #         predicted_boxes.append(box)
    #         predicted_scores.append(scores[i])
    #         predicted_labels.append(labels[i])

    #     # setting predicted dictionary 
    #     item = {
    #         "boxes": torch.tensor(predicted_boxes, dtype=torch.float),
    #         "scores": torch.tensor(predicted_scores, dtype=torch.float),
    #         "labels": torch.tensor(predicted_labels)
    #         }
    #     predicted.append(item)

    #     # returning predicted object
    #     return predicted 

    def set_details_of_inferenced_image(self, image_name, targets, preds):

        # The sample below shows data format used here.
        #    inferenced_image: {
        #        'image_name': 'IMG_1853-bbox-1526946742.jpg', 
        #        'targets_list': [
        #            {
        #            'boxes': tensor([[ 85.,  36., 215., 263.]]), 
        #            'labels': tensor([3])
        #            }
        #            ], 
        #        'preds_list': [
        #            {
        #            'boxes': tensor([[105.,  59., 193., 235.],	[103.,  54., 197., 242.]]), 
        #            'scores': tensor([0.7952, 0.5049]), 
        #            'labels': tensor([3, 2])
        #            }
        #            ]		
        #    }

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


    def compute_confusion_matrix_old(self, model_name, num_classes, threshold, iou_threshold, metrics_folder):

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

            # logging_info(f' inferenced_image: {inferenced_image}')

            # getting target and predictions bounding boxes for evaluation             
            targets = inferenced_image["targets_list"]
            preds = inferenced_image["preds_list"]
            # logging_info(f'targets: {targets}')
            # logging_info(f'preds: {preds}')
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
                            status='Undetected object'
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
                                    status = 'Target detected'
                                    number_of_bounding_boxes_predicted_with_target += 1                           
                                    if p_label == t_label:
                                        # True Positive 
                                        self.full_confusion_matrix[t_label, p_label] += 1
                                    else:                        
                                        # False Positive 
                                        self.full_confusion_matrix[t_label, p_label] += 1
                                else:
                                    # Counting ghost predictions   
                                    number_of_ghost_predictions += 1                         
                                    self.full_confusion_matrix[t_label, ghost_predictions_index] += 1
                                    status = 'Ghost prediction'

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
            model_name + '_images_bounding_boxes.xlsx'
        )
        self.save_inferenced_images(path_and_filename)

        # getting just confusion matrix whithout the background, ghost predictions and undetected objects
        self.confusion_matrix = np.copy(self.full_confusion_matrix[1:-1,1:-1])

        # normalizing values summarizing by rows
        self.confusion_matrix_normalized = np.copy(self.confusion_matrix)
        sum_columns_aux_1 = np.sum(self.confusion_matrix_normalized,axis=1)
        row, col = self.confusion_matrix_normalized.shape
        for i in range(row):
            if sum_columns_aux_1[i] > 0:
                self.confusion_matrix_normalized[i] = self.confusion_matrix_normalized[i] / sum_columns_aux_1[i]

        # normalizing values summarizing by rows
        self.full_confusion_matrix_normalized = np.copy(self.full_confusion_matrix)
        sum_columns_aux_2 = np.sum(self.full_confusion_matrix_normalized,axis=1)
        row, col = self.full_confusion_matrix_normalized.shape
        for i in range(row):
            if sum_columns_aux_2[i] > 0:
                self.full_confusion_matrix_normalized[i] = self.full_confusion_matrix_normalized[i] / sum_columns_aux_2[i]

        # summary of confusion matrix        
        self.confusion_matrix_summary["number_of_images"] = len(self.inferenced_images)
        self.confusion_matrix_summary["number_of_bounding_boxes_target"] = number_of_bounding_boxes_target         
        self.confusion_matrix_summary["number_of_bounding_boxes_predicted"] = number_of_bounding_boxes_predicted
        self.confusion_matrix_summary["number_of_bounding_boxes_predicted_with_target"] = number_of_bounding_boxes_predicted_with_target
        self.confusion_matrix_summary["number_of_ghost_predictions"] = number_of_ghost_predictions
        self.confusion_matrix_summary["number_of_undetected_objects"] = number_of_undetected_objects

        # computing metrics from confuson matrix 
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
    def compute_metrics_from_confusion_matrix_deactivated(self):
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

        logging_info(f'confusion: {LINE_FEED}{self.full_confusion_matrix}')

        self.tp_per_classes = []
        self.fp_per_classes = []
        self.fn_per_classes = []
        self.tn_per_classes = []
        self.tp_model = 0
        self.fp_model = 0
        self.fn_model = 0
        self.tn_model = 0

        cm_fp = self.full_confusion_matrix[1:-1, 1:]
        self.tp_per_classes = cm_fp.diagonal()
        self.fp_per_classes = cm_fp.sum(1) - self.tp_per_classes
        cm_fn = self.full_confusion_matrix[1:, 1:-1]
        self.fn_per_classes = cm_fn.sum(0) - self.tp_per_classes

        self.tp_model = self.tp_per_classes.sum()
        self.fp_model = self.fp_per_classes.sum()
        self.fn_model = self.fn_per_classes.sum()
        self.tn_model = 0 

        logging_info(f'TP / FN / FP / TN from confunsion matrix: ')
        # for count in self.counts_per_class:
        #     logging_info(f'count {count}')
        
        logging_info(f'self.tp_per_classes:{self.tp_per_classes}')
        logging_info(f'self.tp_model:{self.tp_model}')
        logging_info(f'self.fp_per_classes:{self.fp_per_classes}')
        logging_info(f'self.fp_model:{self.fp_model}')
        logging_info(f'self.fn_per_classes:{self.fn_per_classes}')
        logging_info(f'self.fn_model:{self.fn_model}')
        logging_info(f'self.tn_per_classes:{self.tn_per_classes}')
        logging_info(f'self.tn_model:{self.tn_model}')

    def get_value_metric(self, metric):
        value = 0
        for count in self.counts_per_class:
            value += count[metric]
        return value

    # https://docs.kolena.io/metrics/accuracy/
    def get_model_accuracy(self):
        accuracy = (self.tp_model + self.tn_model) /  \
                   (self.tp_model + self.tn_model + self.fp_model + self.fn_model)
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