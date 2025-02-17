import numpy as np
import cv2
import torch
import glob as glob
import os
import time
import argparse

# from model import create_model

# from config import (
#     NUM_CLASSES, DEVICE, CLASSES
# )

# Importing python modules
from common.manage_log import *
from common.utils import * 
from common.entity.ImageAnnotation import ImageAnnotation
from common.metrics import *
from common.metricsTorchmetrics import *
import torchvision

# ###########################################
# Constants
# ###########################################
LINE_FEED = '\n'

# setting seed 
np.random.seed(42)

# ###############################################
# The method below was added by Rubens
# ###############################################

def inference_neural_network_model(parameters, device, model):
    '''
    Inference test images dataset in the trained mode
    '''    

    logging_info(f'')
    logging_info(f'>> Processing the inference on the test images dataset')
    logging_info(f'')

    # creating predictions dictionary of all images 
    all_predictions = {}

    # creating working lists
    y_pred = []
    y_true = []

    # creating metric object 
    # metricsTorchmetrics
    inference_metric = Metrics(
        model=parameters['model_ssd']['neural_network_model']['model_name'],
        number_of_classes=parameters['model_ssd']['neural_network_model']['number_of_classes'],
    )
    metric_torchmetrics = MetricsTorchmetrics(model=parameters['model_ssd']['neural_network_model']['model_name'], number_of_classes=5)

    # loading weights
    path_and_weights_filename = os.path.join(
        parameters['model_ssd']['input']['inference']['weights_folder'],
        parameters['model_ssd']['input']['inference']['weights_filename'],
    )
    checkpoint = torch.load(path_and_weights_filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()

    # getting list of test images for inference 
    input_image_size = str(parameters['model_ssd']['input']['input_dataset']['input_image_size'])
    test_image_dataset_folder = os.path.join(
        parameters['processing']['research_root_folder'],
        parameters['model_ssd']['input']['input_dataset']['input_dataset_path'],
        parameters['model_ssd']['input']['input_dataset']['annotation_format'],
        input_image_size + 'x' + input_image_size,
        'test'
    )

    # classes 
    classes = parameters['neural_network_model']['classes']

    # COLORS 
    colors = [[0, 0, 0],        [255, 0, 0],        [0, 255, 0],    [0, 0, 255], 
              [238, 130, 238],  [106, 90, 205],     [188, 0, 239]]

    # DIR_TEST = args['input']
    test_images = glob.glob(f"{test_image_dataset_folder}/*.jpg")
    # print(f"Tests list: {test_images}")

    frame_count = 0 # To count total frames.
    total_fps = 0 # To get the final frames per second.

    for i in range(len(test_images)):
        
        # extracting parts of path and image filename 
        path, filename_with_extension, filename, extension = Utils.get_filename(test_images[i])

        # logging image name 
        logging_info(f'Test image #{i+1} - {filename_with_extension}')

        # setting the annotation filename 
        path_and_filename_xml_annotation = os.path.join(path, filename + '.xml')

        # getting xml annotation of the image 
        image_annotation = ImageAnnotation()
        image_annotation.get_annotation_file_in_voc_pascal_format(path_and_filename_xml_annotation)
        # logging_info(f'inference - image_annotation: {image_annotation.to_string()} ')
        
        # getting target bounding boxes 
        target = image_annotation.get_tensor_target(classes)
        # logging_info(f'target: {target}')

        # print(f'target 2 : {target}')

        # Get the image file name for saving output later on.
        image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
        image = cv2.imread(test_images[i])
        orig_image = image.copy()
        # if args['imgsz'] is not None:
        #     image = cv2.resize(image, (args['imgsz'], args['imgsz']))
        # logging_info(f'image.shape: {image.shape}')

        # BGR to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # Make the pixel range between 0 and 1.
        image /= 255.0
        
        # Bring color channels to front (H, W, C) => (C, H, W).
        image_input = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        # Convert to tensor.
        image_input = torch.tensor(image_input, dtype=torch.float).cuda()
        
        # Add batch dimension.
        image_input = torch.unsqueeze(image_input, 0)
        start_time = time.time()
        
        # Predictions
        with torch.no_grad():
            outputs = model(image_input.to(device))
        end_time = time.time()

        # logging_info(f'outputs: {outputs}')

        # NOTE: The bounding boxes selected in the "outputs" are ordered decreasing by its scores

        # Get the current fps.
        fps = 1 / (end_time - start_time)
        
        # Total FPS till current frame.
        total_fps += fps
        frame_count += 1

        # Load all detection to CPU for further operations.
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

        nms_prediction = apply_nms(outputs[0], iou_thresh=parameters['model_ssd']['neural_network_model']['non_maximum_suppression'])

        outputs[0] = nms_prediction

        # metricsTorchmetrics
        # adding bounding boxes to MetricTorchmetrics object
        targets_dict = dict()
        preds_dict = dict()
        targets_dict['boxes'] = torch.Tensor(target[0]['boxes'])
        targets_dict['labels'] = torch.Tensor(target[0]['labels'])
        metric_torchmetrics.add_targets(targets_dict)

        # Carry further only if there are detected boxes.
        if len(outputs[0]['boxes']) == 0:
            # metricsTorchmetrics
            preds_dict['boxes'] = torch.Tensor([])
            preds_dict['scores'] = torch.Tensor([])
            preds_dict['labels'] = torch.Tensor([])
            metric_torchmetrics.add_preds(preds_dict)

        else:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            labels = outputs[0]['labels'].data.numpy().astype(int)

            # Filter out boxes according to `detection_threshold`.
            # boxes = boxes[scores >= args['threshold']].astype(np.int32)
            threshold = parameters['model_ssd']['neural_network_model']['threshold']
            boxes = boxes[scores >= threshold].astype(np.int32)
            draw_boxes = boxes.copy()

            boxes_thresholded = []
            scores_thresholded = []
            labels_thresholded = []
            for i in range(len(boxes)):
                if scores[i] >= threshold:
                    boxes_thresholded.append(boxes[i].astype(np.int32))
                    scores_thresholded.append(scores[i])
                    labels_thresholded.append(labels[i])

            # logging_info(f'draw_boxes: {draw_boxes}')
            draw_boxes2 = boxes_thresholded.copy()

            # Get all the predicited class names.
            pred_classes = [classes[label] for label in outputs[0]['labels'].cpu().numpy()]
            
            # getting predicted bounding boxes 
            predicteds = inference_metric.get_predicted_bounding_boxes(
                                        boxes_thresholded,
                                        scores_thresholded,
                                        labels_thresholded)

            # setting target and predicteds bounding boxes for metrics            
            inference_metric.set_details_of_inferenced_image(
                filename_with_extension, target, predicteds)

            key = filename_with_extension
            value = (target, predicteds)
            all_predictions[key] = value 

            # metricsTorchmetrics
            # adding predicted bounding boxes to metric torchmetric object  
            if len(boxes_thresholded) > 0:
                preds_dict['boxes'] = torch.Tensor(boxes_thresholded)
                preds_dict['scores'] = torch.Tensor(scores_thresholded)
                preds_dict['labels'] = torch.Tensor(labels_thresholded).int()
            else: 
                preds_dict['boxes'] = None 
                preds_dict['scores'] = None 
                preds_dict['labels'] = None 
                
            metric_torchmetrics.add_preds(preds_dict)


            # Draw the bounding boxes and write the class name on top of it.
            for j, box in enumerate(draw_boxes):
                # logging_info(f'Inside for j,box - j:{j}  box:{box}')
                # logging_info(f'pred_classes:{pred_classes}')

                class_name = pred_classes[j]
                bbox_score = scores_thresholded[j]
                bbox_label_text = class_name + ' ('+ '%.2f' % bbox_score + ')'

                color = colors[classes.index(class_name)]
                # Recale boxes.
                xmin = int((box[0] / image.shape[1]) * orig_image.shape[1])
                ymin = int((box[1] / image.shape[0]) * orig_image.shape[0])
                xmax = int((box[2] / image.shape[1]) * orig_image.shape[1])
                ymax = int((box[3] / image.shape[0]) * orig_image.shape[0])
                cv2.rectangle(orig_image,
                            (xmin, ymin),
                            (xmax, ymax),
                            color[::-1],
                            3)
                cv2.putText(orig_image,
                            bbox_label_text,
                            (xmin, ymin-5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color[::-1],
                            2,
                            lineType=cv2.LINE_AA)

            # cv2.imshow('Prediction', orig_image)
            # cv2.waitKey(1)
            inferenced_image_folder = os.path.join(
                parameters['test_results']['inferenced_image_folder']
            )
            cv2.imwrite(f"{inferenced_image_folder}/{image_name}_predicted.jpg", orig_image)

        # logging_info(f"Image {i+1} done...")
        # logging_info(f'-'*50)
        
        # logging_info('TEST PREDICTIONS COMPLETE')
    # cv2.destroyAllWindows()

    # ----------------------------------------------------------
    # Computing metrics using TorchMetrics 
    # metricsTorchmetrics
    # metric_torchmetrics.compute_confusion_matrix()
    # metric_torchmetrics.compute_mean_average_precision()
    # metric_torchmetrics.compute_precision()  
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # Computing metrics using Sklearn 
    # 
    # inference_metric.compute_metrics_sklearn()
    # exit()
    
    # ----------------------------------------------------------

    # Computing Confusion Matrix 
    model_name = parameters['model_ssd']['neural_network_model']['model_name']
    num_classes = parameters['model_ssd']['neural_network_model']['number_of_classes'] + 1
    threshold = parameters['model_ssd']['neural_network_model']['threshold']
    iou_threshold = parameters['model_ssd']['neural_network_model']['iou_threshold']
    metrics_folder = parameters['test_results']['metrics_folder']
    running_id_text = parameters['processing']['running_id_text']
    tested_folder = parameters['test_results']['inferenced_image_folder']
    inference_metric.compute_confusion_matrix(model_name, num_classes, threshold, iou_threshold, 
                                              metrics_folder, running_id_text, tested_folder)
    inference_metric.confusion_matrix_to_string()

    # saving confusion matrix plots
    title =  'Full Confusion Matrix' + \
             ' - Model: ' + parameters['model_ssd']['neural_network_model']['model_name'] + \
             '   # images:' + str(inference_metric.confusion_matrix_summary['number_of_images'])
    title += LINE_FEED + \
             'Confidence threshold: ' + str(parameters['model_ssd']['neural_network_model']['threshold']) + \
             '   IoU threshold: ' + str(parameters['model_ssd']['neural_network_model']['iou_threshold']) + \
             '   Non-maximum Supression: ' + str(parameters['model_ssd']['neural_network_model']['non_maximum_suppression'])
    # title += LINE_FEED + '  # bounding box -' + \
    #          ' predicted with target: ' + str(inference_metric.confusion_matrix_summary['number_of_bounding_boxes_predicted_with_target']) + \
    #          '   ghost predictions: ' + str(inference_metric.confusion_matrix_summary['number_of_ghost_predictions']) + \
    #          '   undetected objects: ' + str(inference_metric.confusion_matrix_summary['number_of_undetected_objects'])
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'], 
        parameters['model_ssd']['neural_network_model']['model_name'] + \
        '_' + parameters['processing']['running_id_text'] + '_confusion_matrix_full.png'
    )
    number_of_classes = parameters['model_ssd']['neural_network_model']['number_of_classes']
    cm_classes = classes[0:(number_of_classes+1)]
    x_labels_names = cm_classes.copy()
    y_labels_names = cm_classes.copy()
    x_labels_names.append('Incorrect predictions')    
    y_labels_names.append('Undetected objects')
    format='.0f'
    Utils.save_plot_confusion_matrix(inference_metric.full_confusion_matrix, 
                                     path_and_filename, title, format,
                                     x_labels_names, y_labels_names)
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'], 
        parameters['model_ssd']['neural_network_model']['model_name'] + \
        '_' + parameters['processing']['running_id_text'] + '_confusion_matrix_full.xlsx'
    )
    Utils.save_confusion_matrix_excel(inference_metric.full_confusion_matrix,
                                      path_and_filename, 
                                      x_labels_names, y_labels_names, 
                                      inference_metric.tp_per_class,
                                      inference_metric.fp_per_class,
                                      inference_metric.fn_per_class,
                                      inference_metric.tn_per_class
    )
        
    # title = 'Confusion Matrix'
    # path_and_filename = os.path.join(
    #     parameters['test_results']['metrics_folder'], 
    #     parameters['model_ssd']['neural_network_model']['model_name'] + '_confusion_matrix.png'
    # )
    # cm_classes = classes[1:5]
    # x_labels_names = cm_classes.copy()
    # y_labels_names = cm_classes.copy()
    # format='.0f'
    # Utils.save_plot_confusion_matrix(inference_metric.confusion_matrix, 
    #                                  path_and_filename, title, format,
    #                                  x_labels_names, y_labels_names)
    # path_and_filename = os.path.join(
    #     parameters['test_results']['metrics_folder'], 
    #     parameters['model_ssd']['neural_network_model']['model_name'] + '_confusion_matrix.xlsx'
    # )
    # Utils.save_confusion_matrix_excel(inference_metric.confusion_matrix,
    #                                   path_and_filename,
    #                                   x_labels_names, y_labels_names,
    #                                   inference_metric.tp_per_class,
    #                                   inference_metric.fp_per_class,
    #                                   inference_metric.fn_per_class,
    #                                   inference_metric.tn_per_class
    #                                   )                      

    title =  'Full Confusion Matrix Normalized' + \
             ' - Model: ' + parameters['model_ssd']['neural_network_model']['model_name'] + \
             '   # images:' + str(inference_metric.confusion_matrix_summary['number_of_images'])
    title += LINE_FEED + \
             'Confidence threshold: ' + str(parameters['model_ssd']['neural_network_model']['threshold']) + \
             '   IoU threshold: ' + str(parameters['model_ssd']['neural_network_model']['iou_threshold']) + \
             '   Non-maximum Supression: ' + str(parameters['model_ssd']['neural_network_model']['non_maximum_suppression'])
    # title += LINE_FEED + '  # bounding box -' + \
    #          ' predicted with target: ' + str(inference_metric.confusion_matrix_summary['number_of_bounding_boxes_predicted_with_target']) + \
    #          '   ghost predictions: ' + str(inference_metric.confusion_matrix_summary['number_of_ghost_predictions']) + \
    #          '   undetected objects: ' + str(inference_metric.confusion_matrix_summary['number_of_undetected_objects'])
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'],
        parameters['model_ssd']['neural_network_model']['model_name'] + \
        '_' + parameters['processing']['running_id_text'] + '_confusion_matrix_full_normalized.png'
    )
    cm_classes = classes[0:(number_of_classes+1)]
    x_labels_names = cm_classes.copy()
    y_labels_names = cm_classes.copy()
    x_labels_names.append('Incorrect predictions')    
    y_labels_names.append('Undetected objects')
    format='.2f'
    Utils.save_plot_confusion_matrix(inference_metric.full_confusion_matrix_normalized, 
                                     path_and_filename, title, format,
                                     x_labels_names, y_labels_names)
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'], 
        parameters['model_ssd']['neural_network_model']['model_name'] + \
        '_' + parameters['processing']['running_id_text'] + '_confusion_matrix_full_normalized.xlsx'
    )
    Utils.save_confusion_matrix_excel(inference_metric.full_confusion_matrix_normalized,
                                      path_and_filename,
                                      x_labels_names, y_labels_names, 
                                      inference_metric.tp_per_class,
                                      inference_metric.fp_per_class,
                                      inference_metric.fn_per_class,
                                      inference_metric.tn_per_class
                                      )                      

    # title =  'Confusion Matrix Normalized' + \
    #          ' - Model: ' + parameters['model_ssd']['neural_network_model']['model_name'] + \
    #          '   # images:' + str(inference_metric.confusion_matrix_summary['number_of_images'])
    # title += LINE_FEED + '  # bounding box -' + \
    #          ' predicted with target: ' + str(inference_metric.confusion_matrix_summary['number_of_bounding_boxes_predicted_with_target']) + \
    #          '   ghost predictions: ' + str(inference_metric.confusion_matrix_summary['number_of_ghost_predictions']) + \
    #          '   undetected objects: ' + str(inference_metric.confusion_matrix_summary['number_of_undetected_objects'])
    # path_and_filename = os.path.join(
    #     parameters['test_results']['metrics_folder'], 
    #     parameters['model_ssd']['neural_network_model']['model_name'] + '_confusion_matrix_normalized.png'
    # )
    # cm_classes = classes[1:5]
    # x_labels_names = cm_classes.copy()
    # y_labels_names = cm_classes.copy()
    # format='.2f'
    # Utils.save_plot_confusion_matrix(inference_metric.confusion_matrix_normalized, 
    #                                  path_and_filename, title, format,
    #                                  x_labels_names, y_labels_names)
    # path_and_filename = os.path.join(
    #     parameters['test_results']['metrics_folder'], 
    #     parameters['model_ssd']['neural_network_model']['model_name'] + '_confusion_matrix_normalized.xlsx'
    # )
    # Utils.save_confusion_matrix_excel(inference_metric.confusion_matrix_normalized,
    #                                   path_and_filename,
    #                                   x_labels_names, y_labels_names, 
    #                                   inference_metric.tp_per_class,
    #                                   inference_metric.fp_per_class,
    #                                   inference_metric.fn_per_class,
    #                                   inference_metric.tn_per_class
    #                                   )
                                      
    # saving metrics from confusion matrix
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'],
        parameters['model_ssd']['neural_network_model']['model_name'] + \
        '_' + parameters['processing']['running_id_text'] + '_confusion_matrix_metrics.xlsx'
    )
    
    sheet_name='metrics_summary'
    sheet_list = []
    sheet_list.append(['Metrics Results calculated by application', ''])
    sheet_list.append(['', ''])
    sheet_list.append(['Model', f'{ parameters["neural_network_model"]["model_name"]}'])
    sheet_list.append(['', ''])
    sheet_list.append(['Threshold',  f"{parameters['model_ssd']['neural_network_model']['threshold']:.2f}"])
    sheet_list.append(['IoU Threshold',  f"{parameters['model_ssd']['neural_network_model']['iou_threshold']:.2f}"])
    sheet_list.append(['Non-Maximum Supression',  f"{parameters['model_ssd']['neural_network_model']['non_maximum_suppression']:.2f}"])
    sheet_list.append(['', ''])

    sheet_list.append(['TP / FP / FN / TN per Class', ''])
    cm_classes = classes[1:(number_of_classes+1)]

    # setting values of TP, FP, and FN per class
    sheet_list.append(['Class', 'TP', 'FP', 'FN', 'TN'])
    for i, class_name in enumerate(classes[1:(number_of_classes+1)]):
        row = [class_name, 
               f'{inference_metric.tp_per_class[i]:.0f}',
               f'{inference_metric.fp_per_class[i]:.0f}',
               f'{inference_metric.fn_per_class[i]:.0f}',
               f'{inference_metric.tn_per_class[i]:.0f}',
              ]
        sheet_list.append(row)

    i += 1
    row = ['Total',
           f'{inference_metric.tp_model:.0f}',
           f'{inference_metric.fp_model:.0f}',
           f'{inference_metric.fn_model:.0f}',
           f'{inference_metric.tn_model:.0f}',
          ]
    sheet_list.append(row)    
    sheet_list.append(['', ''])

    # setting values of metrics precision, recall, f1-score and dice per class
    sheet_list.append(['Class', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Dice'])
    for i, class_name in enumerate(classes[1:number_of_classes+1]):
        row = [class_name, 
               f'{inference_metric.accuracy_per_class[i]:.8f}',
               f'{inference_metric.precision_per_class[i]:.8f}',
               f'{inference_metric.recall_per_class[i]:.8f}',
               f'{inference_metric.f1_score_per_class[i]:.8f}',
               f'{inference_metric.dice_per_class[i]:.8f}',
              ]
        sheet_list.append(row)

    i += 1
    row = ['Model Metrics',
               f'{inference_metric.get_model_accuracy():.8f}',
               f'{inference_metric.get_model_precision():.8f}',
               f'{inference_metric.get_model_recall():.8f}',
               f'{inference_metric.get_model_f1_score():.8f}',
               f'{inference_metric.get_model_dice():.8f}',
          ]
    sheet_list.append(row)
    sheet_list.append(['', ''])

    # metric measures 
    sheet_list.append(['Metric measures', ''])
    sheet_list.append(['number_of_images', f'{inference_metric.confusion_matrix_summary["number_of_images"]:.0f}'])
    sheet_list.append(['number_of_bounding_boxes_target', f'{inference_metric.confusion_matrix_summary["number_of_bounding_boxes_target"]:.0f}'])
    sheet_list.append(['number_of_bounding_boxes_predicted', f'{inference_metric.confusion_matrix_summary["number_of_bounding_boxes_predicted"]:.0f}'])
    sheet_list.append(['number_of_bounding_boxes_predicted_with_target', f'{inference_metric.confusion_matrix_summary["number_of_bounding_boxes_predicted_with_target"]:.0f}'])
    sheet_list.append(['number_of_incorrect_predictions', f'{inference_metric.confusion_matrix_summary["number_of_ghost_predictions"]:.0f}'])
    sheet_list.append(['number_of_undetected_objects', f'{inference_metric.confusion_matrix_summary["number_of_undetected_objects"]:.0f}'])

    # saving metrics sheet
    Utils.save_metrics_excel(path_and_filename, sheet_name, sheet_list)
    logging_sheet(sheet_list)

    # Calculate and print the average FPS.
    avg_fps = total_fps / frame_count
    # logging_info(f"Average FPS: {avg_fps:.3f}")

    # returning predictions 
    return all_predictions
    
# This source code was extracted from 'inference.py' module of the 'wm-model-faster-rcnn' project
# the function takes the original prediction and the iou threshold.
def apply_nms(orig_prediction, iou_thresh=0.3):

    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    # logging_info(f'keep: {keep}')
    # logging_info(f'len(keep): {len(keep)}')

    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction
