import torch 
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as torchtrans

# Importing python modules
from model_faster_rcnn.visualization import * 

from common.manage_log import *
from common.metrics import *

# the function takes the original prediction and the iou threshold.
def apply_nms(orig_prediction, iou_thresh=0.3):

    # logging_info(f'before apply nms')
    # logging_info(f'orig_prediction: {orig_prediction}')

    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    # logging_info(f'after apply nms')
    # logging_info(f'keep: {keep}')

    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction

# function to convert a torchtensor back to PIL image
def torch_to_pil(img):
    return torchtrans.ToPILImage()(img).convert('RGB')


def inference_faster_rcnn_model_with_dataset_test(parameters, model, device, dataset_test):

    # processing test images

    # detection_threshold = 0.5 DO NOT USE

    logging_info(f'Test dataset size #4: {len(dataset_test)}')

    # creating predictions dictionary of all images 
    all_predictions = {}

    # logging_info(f'Threshold confidence: {WHITE_MOLD_THRESHOLD}')
    threshold = parameters['model_faster_rcnn']['neural_network_model']['threshold']

    # creating metric object 
    inference_metric = Metrics(
        model=parameters['model_faster_rcnn']['neural_network_model']['model_name'],
        number_of_classes=parameters['model_faster_rcnn']['neural_network_model']['number_of_classes'],
    )

    for i in range(len(dataset_test)):

        # logging_info(f'dataset_test[i]: {dataset_test[i]}')
        # logging_info(f'dataset_test[i].img: {dataset_test[i].imgs}')

        # pick one image from the test set
        img, target = dataset_test[i]
        image_name = dataset_test.imgs[i]

        # logging_info(f'')
        # logging_info(f'-'*100)
        logging_info(f'Test image #{i+1} - {image_name}')

        # put the model in evaluation mode
        model.eval()
        with torch.no_grad():
            prediction = model([img.to(device)])[0]

        # Move all values from GPU to CPU
        for key, value in prediction.items():
            prediction[key] = value.cpu()      

        # setting results folder
        local_results_faster_rcnn_predict_path = os.path.join(
            parameters['test_results']['inferenced_image_folder']
        )
        # logging_info(f'')
        # logging_info('EXPECTED OUTPUT\n')
        path_and_filename_expected_image = os.path.join(local_results_faster_rcnn_predict_path, image_name + '_expected.jpg')
        plot_img_bbox(torch_to_pil(img), target, path_and_filename_image=path_and_filename_expected_image)

        # logging_info('FASTER R-CNN MODEL OUTPUT\n')
        logging_info(f'prediction: {prediction}')
        # nms_prediction = apply_nms(prediction, iou_thresh=0.01)
        nms_prediction = apply_nms(prediction, iou_thresh=parameters['model_faster_rcnn']['neural_network_model']['non_maximum_suppression'])
        
        # path_and_filename_predicted_image = os.path.join(local_results_faster_rcnn_predict_path, image_name + '_predicted.jpg')        
        filename, extension = Utils.get_filename_and_extension(image_name)
        image_filename = filename + '_predicted.jpg'
        path_and_filename_predicted_image = os.path.join(local_results_faster_rcnn_predict_path, image_filename)
        plot_img_bbox(
            torch_to_pil(img), 
            nms_prediction, 
            threshold=parameters['model_faster_rcnn']['neural_network_model']['threshold'], 
            path_and_filename_image=path_and_filename_predicted_image,
            classes=parameters['neural_network_model']['classes']
        )

        # logging_info(f'nms_prediction {nms_prediction}')

        # select just the predictions with score greater than confidence threshold
        logging_info(f'nms_prediction: {nms_prediction}')
        valid_prediction = {
            'boxes': [],
            'labels': [],
            'scores': [],
        }
        for i, score in enumerate(nms_prediction['scores']):
            logging_info(f'i: {i}   score: {score}   threshold: {threshold}')
            if score >= threshold:
                valid_prediction['boxes'].append((nms_prediction['boxes'][i]).numpy())
                valid_prediction['labels'].append((nms_prediction['labels'][i]).squeeze())
                valid_prediction['scores'].append((nms_prediction['scores'][i]).squeeze())

        logging_info(f'prediction: {valid_prediction}')
        logging_info(f"prediction['boxes']: {valid_prediction['boxes']}")
        logging_info(f"prediction['labels']: {valid_prediction['labels']}")
        logging_info(f"prediction['scores']: {valid_prediction['scores']}")
        valid_prediction['boxes'] = torch.tensor(valid_prediction['boxes'])
        valid_prediction['labels'] = torch.tensor(valid_prediction['labels'])
        valid_prediction['scores'] = torch.tensor(valid_prediction['scores'])
        logging_info(f'prediction converted to tensor: {valid_prediction}')


        # setting target and predicted bounding boxes for metrics 
        new_targets = []
        item_target = {
            "boxes": target['boxes'],
            "labels": target['labels']
            }
        new_targets.append(item_target)

        new_predicteds = []
        item_predicted = {
            "boxes": valid_prediction['boxes'],
            "scores": valid_prediction['scores'],
            "labels": valid_prediction['labels'],
            }
        new_predicteds.append(item_predicted)
        # print(f'--------------------------------------------------')
        # print(f'    target: {target}')
        # print(f'new_targets: {new_targets}')
        # print(f'new_predicteds: {new_predicteds}')
        # print(f'--------------------------------------------------')

        # setting target and predicted bounding boxes for metrics
        inference_metric.set_details_of_inferenced_image(
            image_name, new_targets, new_predicteds) 
        # inference_metric.target.extend(new_targets)
        # inference_metric.preds.extend(new_predicteds)
        # print(f'inference_metric.to_string: {inference_metric.to_string()}')
        # print(f'--------------------------------------------------')

        key = image_name
        value = (new_targets, new_predicteds)
        all_predictions[key] = value 


    # classes 
    classes = parameters['neural_network_model']['classes']

    # Computing Confusion Matrix 
    model_name = parameters['model_faster_rcnn']['neural_network_model']['model_name']
    num_classes = parameters['model_faster_rcnn']['neural_network_model']['number_of_classes'] + 1
    threshold = parameters['model_faster_rcnn']['neural_network_model']['threshold']
    iou_threshold = parameters['model_faster_rcnn']['neural_network_model']['iou_threshold']
    metrics_folder = parameters['test_results']['metrics_folder']
    running_id_text = parameters['processing']['running_id_text']    
    tested_folder = parameters['test_results']['inferenced_image_folder']
    inference_metric.compute_confusion_matrix(model_name, num_classes, threshold, iou_threshold, 
                                              metrics_folder, running_id_text, tested_folder)
    inference_metric.confusion_matrix_to_string()

    # saving confusion matrix plots
    title =  'Full Confusion Matrix' + \
             ' - Model: ' + parameters['model_faster_rcnn']['neural_network_model']['model_name'] + \
             '   # images:' + str(inference_metric.confusion_matrix_summary['number_of_images'])
    title += LINE_FEED + \
             'Confidence threshold: ' + str(parameters['model_faster_rcnn']['neural_network_model']['threshold']) + \
             '   IoU threshold: ' + str(parameters['model_faster_rcnn']['neural_network_model']['iou_threshold']) + \
             '   Non-maximum Supression: ' + str(parameters['model_faster_rcnn']['neural_network_model']['non_maximum_suppression'])
    # title += LINE_FEED + '  # bounding box -' + \
    #          ' predicted with target: ' + str(inference_metric.confusion_matrix_summary['number_of_bounding_boxes_predicted_with_target']) + \
    #          '   ghost predictions: ' + str(inference_metric.confusion_matrix_summary['number_of_ghost_predictions']) + \
    #          '   undetected objects: ' + str(inference_metric.confusion_matrix_summary['number_of_undetected_objects'])
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'], 
        parameters['model_faster_rcnn']['neural_network_model']['model_name'] + \
        '_' + parameters['processing']['running_id_text'] + '_confusion_matrix_full.png'
    )
    number_of_classes = parameters['model_faster_rcnn']['neural_network_model']['number_of_classes']
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
        parameters['model_faster_rcnn']['neural_network_model']['model_name'] + \
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
    #     parameters['model_faster_rcnn']['neural_network_model']['model_name'] + '_confusion_matrix.png'
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
    #     parameters['model_faster_rcnn']['neural_network_model']['model_name'] + '_confusion_matrix.xlsx'
    # )
    # Utils.save_confusion_matrix_excel(inference_metric.confusion_matrix,
    #                                   path_and_filename,
    #                                   x_labels_names, y_labels_names)                      

    title =  'Full Confusion Matrix Normalized' + \
             ' - Model: ' + parameters['model_faster_rcnn']['neural_network_model']['model_name'] + \
             '   # images:' + str(inference_metric.confusion_matrix_summary['number_of_images'])
    title += LINE_FEED + \
             'Confidence threshold: ' + str(parameters['model_faster_rcnn']['neural_network_model']['threshold']) + \
             '   IoU threshold: ' + str(parameters['model_faster_rcnn']['neural_network_model']['iou_threshold']) + \
             '   Non-maximum Supression: ' + str(parameters['model_faster_rcnn']['neural_network_model']['non_maximum_suppression'])
    # title += LINE_FEED + '  # bounding box -' + \
    #          ' predicted with target: ' + str(inference_metric.confusion_matrix_summary['number_of_bounding_boxes_predicted_with_target']) + \
    #          '   ghost predictions: ' + str(inference_metric.confusion_matrix_summary['number_of_ghost_predictions']) + \
    #          '   undetected objects: ' + str(inference_metric.confusion_matrix_summary['number_of_undetected_objects'])
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'],
        parameters['model_faster_rcnn']['neural_network_model']['model_name'] + \
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
        parameters['model_faster_rcnn']['neural_network_model']['model_name'] + \
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
    #          ' - Model: ' + parameters['model_faster_rcnn']['neural_network_model']['model_name'] + \
    #          '   # images:' + str(inference_metric.confusion_matrix_summary['number_of_images'])
    # title += LINE_FEED + '  # bounding box -' + \
    #          ' predicted with target: ' + str(inference_metric.confusion_matrix_summary['number_of_bounding_boxes_predicted_with_target']) + \
    #          '   ghost predictions: ' + str(inference_metric.confusion_matrix_summary['number_of_ghost_predictions']) + \
    #          '   undetected objects: ' + str(inference_metric.confusion_matrix_summary['number_of_undetected_objects'])
    # path_and_filename = os.path.join(
    #     parameters['test_results']['metrics_folder'], 
    #     parameters['model_faster_rcnn']['neural_network_model']['model_name'] + '_confusion_matrix_normalized.png'
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
    #     parameters['model_faster_rcnn']['neural_network_model']['model_name'] + '_confusion_matrix_normalized.xlsx'
    # )
    # Utils.save_confusion_matrix_excel(inference_metric.confusion_matrix_normalized,
    #                                   path_and_filename,
    #                                   x_labels_names, y_labels_names)

    # saving metrics from confusion matrix
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'],
        parameters['model_faster_rcnn']['neural_network_model']['model_name'] + \
        '_' + parameters['processing']['running_id_text'] + '_confusion_matrix_metrics.xlsx'
    )
    
    sheet_name='metrics_summary'
    sheet_list = []
    sheet_list.append(['Metrics Results calculated by application', ''])
    sheet_list.append(['', ''])
    sheet_list.append(['Model', f'{ parameters['model_faster_rcnn']["neural_network_model"]["model_name"]}'])
    sheet_list.append(['', ''])
    sheet_list.append(['Threshold',  f"{parameters['model_faster_rcnn']['neural_network_model']['threshold']:.2f}"])
    sheet_list.append(['IoU Threshold',  f"{parameters['model_faster_rcnn']['neural_network_model']['iou_threshold']:.2f}"])
    sheet_list.append(['Non-Maximum Supression',  f"{parameters['model_faster_rcnn']['neural_network_model']['non_maximum_suppression']:.2f}"])
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
    for i, class_name in enumerate(classes[1:(number_of_classes+1)]):
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

    # returning predictions 
    return all_predictions
