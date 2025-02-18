# This module aims to implement the model ensembling method for the previous models 
# explored in our research as SSD, Faster RCNN, YOLO series, DETR, and TransUNEt models.
#  
# The code below was generated by CHatGPT with the following prompt:
#   "Give me an pytorch example using model ensembling for object detection with the pretrained models SSD, Faster RCNN, YOLOv8, YOLOv9, YOLOv10, 
#   DETR, and TRansUNet (adapated for obejct detection)"
# 

import math 
import torch 
import torchvision
from torchvision.ops import * 

from common.manage_log import *
from common.metrics import *
from common.utils  import *
from common.entity.ImageAnnotation import ImageAnnotation
from model_ensemble.image_utils import ImageUtils


def run_ensemble_test_images(parameters, all_predictions):
    
    print(f'')
    print(f'run_ensemble_test_images')
    print(f'')
    print(f'all_predictions: {all_predictions}')

    # getting all predictions keys as test image filename for all inferenced models 
    test_image_filenames = get_all_test_image_filenames(all_predictions)            
    print(f'len(test_image_filenames): {len(test_image_filenames)}')
    print(f'test_image_filenames: {test_image_filenames}')

    # getting IoU threshold 
    # iou_threshold_for_grouping = parameters['neural_network_model']['iou_threshold_for_grouping']
    # iou_threshold_for_inference = parameters['neural_network_model']['iou_threshold_for_inference']
    non_maximum_suppression = parameters['neural_network_model']['non_maximum_suppression']

    # creating inference metrics for ensemble strategies
    affirmative_prediction_type = "Affirmative"    
    affirmative_inference_metric = Metrics(
        model=parameters['neural_network_model']['model_name'] + ' - ' + affirmative_prediction_type,
        number_of_classes=parameters['neural_network_model']['number_of_classes'],
    )
    consensus_prediction_type = "Consensus"    
    consensus_inference_metric = Metrics(
        model=parameters['neural_network_model']['model_name'] + ' - ' + consensus_prediction_type,
        number_of_classes=parameters['neural_network_model']['number_of_classes'],
    )
    unanimous_prediction_type = "Unanimous"    
    unanimous_inference_metric = Metrics(
        model=parameters['neural_network_model']['model_name'] + ' - ' + unanimous_prediction_type,
        number_of_classes=parameters['neural_network_model']['number_of_classes'],
    )
        
    # ensembling the inference results for each test image 
    image_number = 0
    for test_image_filename in test_image_filenames:

        image_number += 1
        print(f'')
        print(f'----------------------------------------------------')
        print(f'test_image_filename #{image_number}: {test_image_filename}')


        # flattening list of predictions
        print(f'')
        print(f'get_flattened_predictons')
        flattened_predictions = get_flattened_predictons(all_predictions, test_image_filename)
        all_predictions_size = len(flattened_predictions)

        print(f'len(flattened_predictions): {len(flattened_predictions)}')
        print(f'flattened_predictions: {flattened_predictions}')

        # grouping detections based on overlapping of their bounding boxes and classes
        # by using the IoU metric
        print(f'')
        print(f'get_grouped_predictions')
        grouped_predictions = get_grouped_predictions(
            flattened_predictions, parameters['neural_network_model']['iou_threshold_for_grouping']
        )
        print(f'len(grouped_predictions): {len(grouped_predictions)}')
        i = 1
        for grouped_predicton in grouped_predictions:
            print(f'grouped_predicton: {i}) {len(grouped_predicton)} - {grouped_predicton}')
            i += 1

        # applying voting strategies in the grouped predictions: affirmative, consensus, and unanimous
        affirmative_predictions, consensus_predictions, unanimous_predictions = \
            apply_voting_strategies(grouped_predictions, all_predictions_size)
        print(f'')
        print(f'Voting Strategies Results:')
        print(f'')
        print(f'affirmative_predictions: {len(affirmative_predictions)} - {affirmative_predictions}')
        print(f'consensus_predictions: {len(consensus_predictions)} - {consensus_predictions}')
        print(f'unanimous_predictions: {len(unanimous_predictions)} - {unanimous_predictions}')
        print(f'')

        # applying non-maximumk supress (nms) in the predictions 
        affirmative_nms_predictions = apply_nms_into_predictions(affirmative_predictions, non_maximum_suppression)
        print(f'affirmative_predictions after nms: {len(affirmative_nms_predictions)} - {affirmative_nms_predictions}')
        consensus_nms_predictions = apply_nms_into_predictions(consensus_predictions, non_maximum_suppression)
        print(f'consensus_predictions after nms: {len(consensus_nms_predictions)} - {consensus_nms_predictions}')
        unanimous_nms_predictions = apply_nms_into_predictions(unanimous_predictions, non_maximum_suppression)
        print(f'unanimous_predictions after nms: {len(unanimous_nms_predictions)} - {unanimous_nms_predictions}')
        print(f'')

        # getting target annotations of the test image 
        targets = get_target_of_test_image(parameters, test_image_filename)
        
        # adding ensembled predictions to the performance metrics
        print(f'add_ensembled_predictions_to_performance_metrics - affirmative_inference_metric')
        add_ensembled_predictions_to_performance_metrics(
            affirmative_inference_metric, test_image_filename, targets, affirmative_nms_predictions)
        print(f'add_ensembled_predictions_to_performance_metrics - consensus_inference_metric')
        add_ensembled_predictions_to_performance_metrics(
            consensus_inference_metric, test_image_filename, targets, consensus_nms_predictions)
        print(f'add_ensembled_predictions_to_performance_metrics - unanimous_inference_metric')
        add_ensembled_predictions_to_performance_metrics(
            unanimous_inference_metric, test_image_filename, targets, unanimous_nms_predictions)
        print(f'')

        # saving predicted image with bounding boxes
        save_predicted_image(parameters, test_image_filename, affirmative_nms_predictions, affirmative_prediction_type)
        save_predicted_image(parameters, test_image_filename, consensus_nms_predictions, consensus_prediction_type)
        save_predicted_image(parameters, test_image_filename, unanimous_nms_predictions, unanimous_prediction_type)


    # computing the performance metrics 
    print(f'')
    print(f'affirmative_inference_metric: {len(affirmative_inference_metric.inferenced_images)}')
    print(f'affirmative_inference_metric: {affirmative_inference_metric.inferenced_images}')
    computing_performance_metrics(parameters, affirmative_inference_metric, affirmative_prediction_type)

    print(f'')
    print(f'consensus_inference_metric: {len(consensus_inference_metric.inferenced_images)}')
    print(f'consensus_inference_metric: {consensus_inference_metric.inferenced_images}')
    computing_performance_metrics(parameters, consensus_inference_metric, consensus_prediction_type)

    print(f'')   
    print(f'unanimous_inference_metric: {len(unanimous_inference_metric.inferenced_images)}')
    print(f'unanimous_inference_metric: {unanimous_inference_metric.inferenced_images}')   
    computing_performance_metrics(parameters, unanimous_inference_metric, unanimous_prediction_type)
    print(f'')


def get_all_test_image_filenames(predictions):

    # getting all test image filenames with no duplicates
    test_image_filenames = {}
    for model in list(predictions.keys()):
        for image_filename in list(predictions[model].keys()):
            test_image_filenames[image_filename] = image_filename
                
    # returning test image filenames 
    return test_image_filenames


def get_flattened_predictons(predictions, test_image_filename):

    # creating flattened predictions of one image from all models 
    flattened_predictions = []

    for model in list(predictions.keys()):
        if len(predictions[model]) > 0:
            test_image_predictions = predictions[model][test_image_filename]

            # get all valid predictions of the model 
            image_predictions = test_image_predictions[1]
            if len(image_predictions) == 0: 
                continue 

            # flattening image predictions 
            for box, score, label in zip(image_predictions[0]['boxes'], 
                                             image_predictions[0]['scores'], 
                                             image_predictions[0]['labels']):
                # setting data of the one detection 
                detection = {}
                detection['box'] = box 
                detection['score'] = score
                detection['label'] = label
                detection['model'] = model
                
                # adding detection to flattened predictions 
                flattened_predictions.append(detection)

    # returning flattened predictions
    return flattened_predictions

# grouping detections based on overlapping of their bounding boxes and classes
# by using the IoU metric
def get_grouped_predictions(flattened_predictions, iou_threshold_for_grouping):

    # setting flattened predictions as NOT removed
    for prediction in flattened_predictions:
        prediction['removed'] = False        

    # creating grouped predictions of one image from flattened predictions 
    grouped_predictions = []
    one_grouped_prediction = []

    # calculating the IoU between all predictions of the same class for grouping 
    for i in range(len(flattened_predictions)):
        # getting bounding box reference to compare with the others
        bounding_box_reference = flattened_predictions[i]
        if bounding_box_reference['removed']:
            continue

        # initialize IoU of the reference bounding box
        bounding_box_reference['iou'] = 0
        bounding_box_reference['iou_threshold_for_grouping'] = iou_threshold_for_grouping

        # setting one grouped prediction with first prediction (reference)
        one_grouped_prediction.append(bounding_box_reference)

        # removing bounding box reference from the flatted predictions 
        bounding_box_reference['removed'] = True

        print(f'')
        
        # comparing with others bounding boxes 
        for j in range(i+1, len(flattened_predictions)):
            # getting bounding box for comparision 
            bounding_box_next = flattened_predictions[j]
            if bounding_box_next['removed']:
                continue

            # calculating IoU of the two bounding boxes: reference and next
            # Both sets of boxes are expected to be in (x1, y1, x2, y2)
            box_reference = bounding_box_reference['box']
            box_next = bounding_box_next['box']
            iou = box_iou(box_reference.unsqueeze(0), box_next.unsqueeze(0))
            print(f'computing iou {box_reference} and {box_next}: {iou}')

            # setting IoU value 
            bounding_box_next['iou'] = iou
            bounding_box_next['iou_threshold_for_grouping'] = iou_threshold_for_grouping

            # evaluating overlapping of bounding boxes reference and next
            if (bounding_box_reference['label'] ==  bounding_box_next['label']) and \
               (iou >= iou_threshold_for_grouping):

                # removing bounding box reference from the flatted predictions 
                bounding_box_next['removed'] = True

                # adding bounding box to group
                one_grouped_prediction.append(bounding_box_next)

        # adding one new grouped prediction
        grouped_predictions.append(one_grouped_prediction)

        # initializing one new grouped prediction
        one_grouped_prediction = []

    # returning grouped predictions         
    return grouped_predictions

# applying voting strategies in the grouped predictions that can be of three strategies:
# 1) affirmative: all grouped predictions 
# 2) consensus: the group size must be greater than m/2, where m is the size of flattened predeictions
# 3) unanimous: the group size must be equal to m size
def apply_voting_strategies(grouped_predictions, all_predictions_size):

    # creating results list for each strategy
    affirmative_predictions = []
    consensus_predictions = []
    unanimous_predictions = []

    # processing all grouped predictions
    for grouped_prediction in grouped_predictions:

        # affirmative strategy
        for prediction in grouped_prediction:
            affirmative_predictions.append(prediction)

        # consensus strategy
        if len(grouped_prediction) >= math.ceil(all_predictions_size / 2.0):
            for prediction in grouped_prediction:
                consensus_predictions.append(prediction)

        # unanimous strategy 
        if len(grouped_prediction) == all_predictions_size:
            for prediction in grouped_prediction:
                unanimous_predictions.append(prediction)

    # returning results of ensembling 
    return affirmative_predictions, consensus_predictions, unanimous_predictions

# apply non-maximum supressoion in the predicitons list to remove overlapping bounding boxes
def apply_nms_into_predictions(predictions, iou_threshold):
    
    # initializing kept prediction 
    kept_prediction = {}

    # evaluating predictions size
    if len(predictions) == 0:
        print(f'predictions list for nms is empty: {predictions}')
        return kept_prediction
    
    # preparing boxes and scores to apply nms
    bounding_boxes = []
    scores = []
    for prediction in predictions:
        bounding_boxes.append(prediction['box'].numpy())
        scores.append(prediction['score']) 

    bounding_boxes = torch.Tensor(bounding_boxes)        
    scores = torch.Tensor(scores)
    print(f'')
    print(f'apply nms - bounding_boxes: {bounding_boxes}')
    print(f'apply nms - scores: {scores}')
    print(f'apply nms - iou_threshold: {iou_threshold}')
    print(f'')

    # applying nms in the predictions
    keep_indexes = nms(bounding_boxes, scores, iou_threshold)
    print(f'apply nms - keep_indexes: {keep_indexes}')

    # preparing kept predictions 
    keep_predictions = []
    for keep_index in keep_indexes:
        keep_predictions.append(predictions[keep_index])

    # for i in range(len(predictions)):
    #     if i in keep_index:
    #         kept_predictions.append(predictions[i])

    print(f'apply nms - keep_predictions: {keep_predictions}')
    return keep_predictions

# getting target annotations of the test image 
def get_target_of_test_image(parameters, test_image_filename):

    # setting workiing folders 
    path = parameters['processing']['image_dataset_folder_test']
    classes = parameters['neural_network_model']['classes']

    # setting the annotation filename 
    path_and_filename_xml_annotation = os.path.join(path, test_image_filename.replace('.jpg', '.xml'))

    # getting xml annotation of the image 
    image_annotation = ImageAnnotation()
    image_annotation.get_annotation_file_in_voc_pascal_format(path_and_filename_xml_annotation)

    # getting target bounding boxes 
    target = image_annotation.get_tensor_target(classes)

    # returning target of test image 
    return target 

# add ensembled predictions to the performance metrics
def add_ensembled_predictions_to_performance_metrics(inference_metric, image_name, targets, predictions):

    # creating adjusted predictions for inference metrics 
    adjusted_predictions = []
    
    # if len(predictions) == 0:
    #     print(f'predictions empty')
    #     return 

    # adjusting key names of the predictions for performance metric 
    for prediction in predictions:
        item = {}
        item['boxes'] = torch.unsqueeze(prediction['box'], 0)
        item['scores'] = torch.unsqueeze(prediction['score'], 0)
        item['labels'] = torch.unsqueeze(prediction['label'], 0)
        item['model'] = prediction['model']
        adjusted_predictions.append(item)
    
    print(f'add_ensembled_predictions_to_performance_metrics:')
    print(f'targets: {len(targets)} - {targets}')
    print(f'adjusted_predictions: {len(adjusted_predictions)} - {adjusted_predictions}')
    print(f'')
    
    # adding targets and predictions
    inference_metric.set_details_of_inferenced_image(image_name, targets, adjusted_predictions)


def save_predicted_image(parameters, image_name, predictions, prediction_type):

    # get test image folder
    path = parameters['processing']['image_dataset_folder_test']

    # classes 
    classes = parameters['neural_network_model']['classes_short_name']

    # COLORS 
    colors = [[0, 0, 0],        [255, 0, 0],        [0, 255, 0],    [0, 0, 255], 
              [238, 130, 238],  [106, 90, 205],     [188, 0, 239]]

    # read test image 
    image = ImageUtils.read_image(image_name,path)
    original_image = image.copy()
    
    # drawing all bounding boxes in the image
    for prediction in predictions:
        box = prediction['box']
        score = prediction['score']
        label = prediction['label']
        linP1 = int(box[1])
        colP1 = int(box[0])
        linP2 = int(box[3])
        colP2 = int(box[2])
        background_box_color = colors[label]
        thickness = 1
        label = classes[label] + ' (' + '%.2f' % score + ')'
        image = ImageUtils.draw_bounding_box(
                    image, linP1, colP1, linP2, colP2,
                    background_box_color, thickness, label
                )

    # saving predicted image
    if prediction_type == "Affirmative":
        path_and_filename = parameters['test_results']['affirmative_strategy_folder']
    elif prediction_type == "Consensus":
        path_and_filename = parameters['test_results']['consensus_strategy_folder']        
    elif prediction_type == "Unanimous":
        path_and_filename = parameters['test_results']['unanimous_strategy_folder']
    else:
        path_and_filename = parameters['test_results']['inferenced_image_folder']

    # save predicted image
    path_and_filename_predicted = os.path.join(path_and_filename, image_name.replace('.jpg', '_predicted.jpg'))
    ImageUtils.save_image(path_and_filename_predicted, image)

    path_and_filename_original = os.path.join(path_and_filename, image_name)
    ImageUtils.save_image(path_and_filename_original, original_image)

# calculate performance metrics of the ensembled predictions 
def computing_performance_metrics(parameters, inference_metric, voting_strategy):

    print(f'Processing model ensemble for performance metrics ')
    
    # classes 
    classes = parameters['neural_network_model']['classes']

    # Computing Confusion Matrix 
    model_name = parameters['neural_network_model']['model_name'] + '-' + voting_strategy
    num_classes = parameters['neural_network_model']['number_of_classes'] + 1
    threshold = parameters['neural_network_model']['threshold']
    iou_threshold_for_inference = parameters['neural_network_model']['iou_threshold_for_inference']
    metrics_folder = parameters['test_results']['metrics_folder']
    running_id_text = parameters['processing']['running_id_text']
    if voting_strategy == 'Affirmative':
        tested_folder = parameters['test_results']['affirmative_strategy_folder']
    elif voting_strategy == 'Consensus':
        tested_folder = parameters['test_results']['consensus_strategy_folder']
    elif voting_strategy == 'Unanimous':
        tested_folder = parameters['test_results']['unanimous_strategy_folder']
    else:
        tested_folder = parameters['test_results']['inferenced_image_folder']
    inference_metric.compute_confusion_matrix(model_name, num_classes, threshold, iou_threshold_for_inference, 
                                              metrics_folder, running_id_text, tested_folder)
    inference_metric.confusion_matrix_to_string()

    # saving confusion matrix plots
    title =  'Full Confusion Matrix' + \
             ' - Model: ' + parameters['neural_network_model']['model_name'] + \
             ' - Voting strategy ' + voting_strategy + \
             '   # images:' + str(inference_metric.confusion_matrix_summary['number_of_images'])
    title += LINE_FEED + \
             'Confidence threshold: ' + str(parameters['neural_network_model']['threshold']) + \
             '   IoU grouping: ' + str(parameters['neural_network_model']['iou_threshold_for_grouping']) + \
             '   IoU inference: ' + str(parameters['neural_network_model']['iou_threshold_for_inference']) + \
             '   Non-maximum Supression: ' + str(parameters['neural_network_model']['non_maximum_suppression'])
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'], 
        parameters['neural_network_model']['model_name'] + \
        '_' + parameters['processing']['running_id_text'] + '_' + voting_strategy + '_confusion_matrix_full.png'
    )
    number_of_classes = parameters['neural_network_model']['number_of_classes']
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
        parameters['neural_network_model']['model_name'] + \
        '_' + parameters['processing']['running_id_text'] + '_' + voting_strategy + '_confusion_matrix_full.xlsx'
    )
    Utils.save_confusion_matrix_excel(inference_metric.full_confusion_matrix,
                                      path_and_filename, 
                                      x_labels_names, y_labels_names, 
                                      inference_metric.tp_per_class,
                                      inference_metric.fp_per_class,
                                      inference_metric.fn_per_class,
                                      inference_metric.tn_per_class
    )

    title =  'Full Confusion Matrix Normalized' + \
             ' - Model: ' + parameters['neural_network_model']['model_name'] + \
             ' - Voting strategy ' + voting_strategy + \
             '   # images:' + str(inference_metric.confusion_matrix_summary['number_of_images'])
    title += LINE_FEED + \
             'Confidence threshold: ' + str(parameters['neural_network_model']['threshold']) + \
             '   IoU grouping: ' + str(parameters['neural_network_model']['iou_threshold_for_grouping']) + \
             '   IoU inference: ' + str(parameters['neural_network_model']['iou_threshold_for_inference']) + \
             '   Non-maximum Supression: ' + str(parameters['neural_network_model']['non_maximum_suppression'])
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'],
        parameters['neural_network_model']['model_name'] + \
        '_' + parameters['processing']['running_id_text'] + '_' + voting_strategy + '_confusion_matrix_full_normalized.png'
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
        parameters['neural_network_model']['model_name'] + \
        '_' + parameters['processing']['running_id_text'] + '_' + voting_strategy + '_confusion_matrix_full_normalized.xlsx'
    )
    Utils.save_confusion_matrix_excel(inference_metric.full_confusion_matrix_normalized,
                                      path_and_filename,
                                      x_labels_names, y_labels_names, 
                                      inference_metric.tp_per_class,
                                      inference_metric.fp_per_class,
                                      inference_metric.fn_per_class,
                                      inference_metric.tn_per_class
                                      )
                                      
    # saving metrics from confusion matrix
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'],
        parameters['neural_network_model']['model_name'] + \
        '_' + parameters['processing']['running_id_text'] + '_' + voting_strategy + '_confusion_matrix_metrics.xlsx'
    )
    
    sheet_name='metrics_summary'
    sheet_list = []
    sheet_list.append(['Metrics Results calculated by application', ''])
    sheet_list.append(['', ''])
    sheet_list.append(['Model', f'{ parameters["neural_network_model"]["model_name"]}'])
    sheet_list.append(['', ''])
    sheet_list.append(['Threshold',  f"{parameters['neural_network_model']['threshold']:.2f}"])
    sheet_list.append(['IoU Threshold Grouping',  f"{parameters['neural_network_model']['iou_threshold_for_grouping']:.2f}"])
    sheet_list.append(['IoU Threshold Inference',  f"{parameters['neural_network_model']['iou_threshold_for_inference']:.2f}"])
    sheet_list.append(['Non-Maximum Supression',  f"{parameters['neural_network_model']['non_maximum_suppression']:.2f}"])
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
