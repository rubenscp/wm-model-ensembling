import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

from coco_eval import CocoEvaluator
from common.manage_log import *
from common.metrics import *
from common.image_utils import *
from common.utils  import *
from dataset import *

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def plot_results(classes, pil_img, height, width, scores, labels, boxes, 
                 path_and_image_filename_with_bbox):

    # no action with no boxes detected
    if len(boxes) == 0:
        return               

    # plt.figure(figsize=(16,10))
    dpi = 96 
    width_in = width / dpi
    height_in = height / dpi
    fig = plt.figure(figsize=(width_in, height_in), dpi=dpi)
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100

    logging_info(f'classes: {classes}')

    for score, label, (xmin, ymin, xmax, ymax), c in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=2))
        logging_info(f'score: {score} label: {label}, c: {c}')
        # text = f'{model.config.id2label[label]}: {score:0.2f}'
        text = f'{classes[label-1]}: {score:0.3f}'
        ax.text(xmin, ymin, text, fontsize=12,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

    # saving figure 
    plt.savefig(path_and_image_filename_with_bbox)

    # close figure plot 
    plt.close(fig)

def save_original_image_with_bbox(classes, image, height, width, scores, labels, boxes, 
                 path_and_image_filename_with_bbox):

    # no action with no boxes detected
    # if len(boxes) == 0:
    #     return               
 
    # setting colors for bounding boxes
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], 
              [0, 255, 255], [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0]]

    logging_info(f'save_original_image_with_bbox - classes: {classes}')

    for score, label, (xmin, ymin, xmax, ymax), c in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):

        logging_info(f'score: {score} label: {label}, c: {c}')
        logging_info(f'xmin: {xmin} ymin: {ymin} xmax: {xmax} ymax: {ymax}')

        # creating new image to check the new coordinates of bounding box
        bgrBoxColor = colors[label-1]
        thickness = 1
        text_label = f'{classes[label]}: {score:0.3f}'
        image = ImageUtils.draw_bounding_box(
            image, ymin, xmin, ymax, xmax, bgrBoxColor, thickness, text_label
            )            

    # saving figure 
    ImageUtils.save_image(path_and_image_filename_with_bbox, image)



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

def inference_detr_model_with_dataset_test(parameters, device, model, processor, 
    dataset_test, dataset_test_original_boxes):

    # logging_info(f'model: {model}')
    # logging_info(f'model.config: {model.config}')
    # logging_info(f'model.config.id2label: {model.config.id2label}')
    # logging_info(f'model.config.label2id: {model.config.label2id}')


    # getting inference parameters
    threshold = parameters['neural_network_model']['threshold']
    classes = parameters['neural_network_model']['classes']
    classes_short_name = parameters['neural_network_model']['classes_short_name']

    # creating metric object 
    inference_metric = Metrics(
        model=parameters['neural_network_model']['model_name'],
        # number of classes doesn't consider the background class
        number_of_classes=parameters['neural_network_model']['number_of_classes']  - 1,
    )

    # Iterate over the test dataset
    i = 0
    for pixel_values, target in dataset_test:
        
        pixel_values = pixel_values.unsqueeze(0).to(device)
        i += 1
        logging_info(f'')
        logging_info(f'Imagem nr: {i}')
        logging_info(f'pixel_values.shape: {pixel_values.shape}')
        logging_info(f'target: {target}')
        
        with torch.no_grad():
            # forward pass to get class logits and bounding boxes
            outputs = model(pixel_values=pixel_values, pixel_mask=None)
            # logging_info(f'outputs {i}: {outputs.keys()}')
            # logging_info(f'outputs["logits"].shape {i}: {outputs["logits"].shape}')
            # logging_info(f'outputs["pred_boxes"].shape {i}: {outputs["pred_boxes"].shape}')        
            # logging_info(f'outputs {i}: {outputs}')        
            # logging_info(f'outputs["logits"] {i}: {outputs["logits"]}')
            # logging_info(f'outputs["pred_boxes"] {i}: {outputs["pred_boxes"]}')            
            # logging_info(f'rubens outputs: {outputs}')

        # load image based on ID
        image_id = target['image_id'].item()
        image = dataset_test.coco.loadImgs(image_id)[0]
                
        logging_info(f"image['file_name']: {image['file_name']}")
        # image = Image.open(image['file_name'])
        path_and_image_filename = image['file_name']
        image = cv2.imread(path_and_image_filename)

        # logging_info(f'image: {image}')
        # logging_info(f'image.shape: {image.shape}')

        # postprocess model outputs
        height, width, channels = image.shape 
        postprocessed_outputs = processor.post_process_object_detection(
            outputs,
            target_sizes=[(height, width)],
            threshold=threshold
        )
        results = postprocessed_outputs[0]
        logging_info(f'rubens results: {results}')
        results_row, results_col = results['boxes'].shape

        # results: {
        # 'scores': tensor([0.8210], device='cuda:0'), 
        # 'labels': tensor([5], device='cuda:0'), 
        # 'boxes': tensor([[108.4646,  74.6296, 189.4505, 222.6129]], device='cuda:0')}

        # apply nom maximum supression on bounding boxes to keep only the predictions 
        # with threshold above confidence
        nms_prediction = apply_nms(results, iou_thresh=parameters['neural_network_model']['non_maximum_suppression'])
        nms_prediction_row, nms_prediction_col = nms_prediction['boxes'].shape
        # logging_info(f'after non maximum supression')
        # logging_info(f'nms_prediction_row: {nms_prediction_row}')
        # logging_info(f'nms_prediction_col: {nms_prediction_col}')
        # logging_info(f'results_row: {results_row}')
        # logging_info(f'results_col: {results_col}')
        if nms_prediction_row != results_row:
            results = nms_prediction
            # logging_info(f'non maximum supression was applied')
            # logging_info(f'results: {results}')

        # logging_info(f'showing results post processed object detection')
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            logging_info(
                f"Detected label {label.item()} - {classes[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location bbox {box}"
            )

        # getting image filename 
        input_path, filename_with_extension, filename, extension = Utils.get_filename(
            path_and_image_filename)

        # copying original image into tested image folder 
        output_path = parameters['test_results']['inferenced_image_folder']
        Utils.copy_file_same_name(filename_with_extension, input_path, output_path)

        # saving image with bounding boxes
        # path_and_image_filename_with_bbox = os.path.join(
        #     parameters['test_results']['inferenced_image_folder'],
        #     filename + '_predicted.png'
        # )
        # plot_results(classes_short_name, 
        #              image,
        #              height, width, 
        #              results['scores'], 
        #              results['labels'], 
        #              results['boxes'],
        #              path_and_image_filename_with_bbox)
        
        # logging_info(f'results: {results.keys()}')
        logging_info(f'results: {results}')

        path_and_image_filename_with_bbox = os.path.join(
            parameters['test_results']['inferenced_image_folder'],
            filename + '_predicted.jpg'
        )
        save_original_image_with_bbox(classes_short_name, 
                                      image,
                                      height, width, 
                                      results['scores'], 
                                      results['labels'], 
                                      results['boxes'],
                                      path_and_image_filename_with_bbox)

       
        # setting target and predicted bounding boxes for metrics 
        new_targets = []
        original_target_boxes, original_target_labels = get_original_boxes(dataset_test_original_boxes, image_id)
        original_target_boxes = torch.tensor(original_target_boxes)
        original_target_labels = torch.tensor(original_target_labels)
        item_target = {
            "boxes": original_target_boxes,
            "labels": original_target_labels
            }
        new_targets.append(item_target)

        new_predicteds = []
        item_predicted = {
            "boxes": results['boxes'].cpu(),
            "scores": results['scores'].cpu(),
            "labels": results['labels'].cpu(),
            }
        new_predicteds.append(item_predicted)

        # logging_info(f'labels adjusted')
        # logging_info(f'new_targets: {new_targets}')
        # logging_info(f'new_predicteds: {new_predicteds}')
       

        # setting target and predicted bounding boxes for metrics
        inference_metric.set_details_of_inferenced_image(
            filename_with_extension, new_targets, new_predicteds)


    # logging_info(f'before adjusting labels of targets and predictions to be used in the metrics class')
    # logging_info(f'new_targets: {new_targets}')
    # logging_info(f'new_predicteds: {new_predicteds}')

    # adjusting labels of targets and predictions to be used in the metrics class    
    # by subtracting 1 from target and prediction labels
    # for target in new_targets:
    #     target["labels"] -= 1

    # for predicted in new_predicteds:
    #     predicted["labels"] -= 1

    # logging_info(f'after adjust labels of targets and predictions to be used in the metrics class')
    # logging_info(f'new_targets: {new_targets}')
    # logging_info(f'new_predicteds: {new_predicteds}')

    # remove path name from the image filename 
    inference_metric.remove_path_from_image_filename()

    # Computing Confusion Matrix 
    model_name = parameters['neural_network_model']['model_name']
    num_classes = parameters['neural_network_model']['number_of_classes']
    threshold = parameters['neural_network_model']['threshold']
    iou_threshold = parameters['neural_network_model']['iou_threshold']
    metrics_folder = parameters['test_results']['metrics_folder']
    running_id_text = parameters['processing']['running_id_text']    
    tested_folder = parameters['test_results']['inferenced_image_folder']
    inference_metric.compute_confusion_matrix(model_name, num_classes, threshold, iou_threshold, 
                                              metrics_folder, running_id_text, tested_folder)
    inference_metric.confusion_matrix_to_string()

    logging_info(f'inference_metric.to_string(): {inference_metric.to_string()}')
  
    # saving confusion matrix plots
    title =  'Full Confusion Matrix' + \
             ' - Model: ' + parameters['neural_network_model']['model_name'] + \
             '   # images:' + str(inference_metric.confusion_matrix_summary['number_of_images'])
    title += LINE_FEED + \
             'Confidence threshold: ' + str(parameters['neural_network_model']['threshold']) + \
             '   IoU threshold: ' + str(parameters['neural_network_model']['iou_threshold']) + \
             '   Non-maximum Supression: ' + str(parameters['neural_network_model']['non_maximum_suppression'])
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'], 
        parameters['neural_network_model']['model_name'] + \
        '_' + parameters['processing']['running_id_text'] + '_confusion_matrix_full.png'
    )
    number_of_classes = parameters['neural_network_model']['number_of_classes']
    cm_classes = classes[0:(number_of_classes+1)]
    x_labels_names = cm_classes.copy()
    y_labels_names = cm_classes.copy()
    x_labels_names.append('Incorrect predictions')    
    y_labels_names.append('Undetected objects')
    format='.0f'
    
    logging_info(f'inference_metric.full_confusion_matrix: {inference_metric.full_confusion_matrix}')
    logging_info(f'path_and_filename: {path_and_filename}')
    logging_info(f'title: {title}')
    logging_info(f'format: {format}')
    logging_info(f'x_labels_names: {x_labels_names}')
    logging_info(f'y_labels_names: {y_labels_names}')
    Utils.save_plot_confusion_matrix(inference_metric.full_confusion_matrix, 
                                     path_and_filename, title, format,
                                     x_labels_names, y_labels_names)
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'],
        parameters['neural_network_model']['model_name'] + \
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

    title =  'Full Confusion Matrix Normalized' + \
             ' - Model: ' + parameters['neural_network_model']['model_name'] + \
             '   # images:' + str(inference_metric.confusion_matrix_summary['number_of_images'])
    title += LINE_FEED + \
             'Confidence threshold: ' + str(parameters['neural_network_model']['threshold']) + \
             '   IoU threshold: ' + str(parameters['neural_network_model']['iou_threshold']) + \
             '   Non-maximum Supression: ' + str(parameters['neural_network_model']['non_maximum_suppression'])
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'],
        parameters['neural_network_model']['model_name'] + \
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
        parameters['neural_network_model']['model_name'] + \
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
    
    # saving metrics from confusion matrix
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'],
        parameters['neural_network_model']['model_name'] + \
        '_' + parameters['processing']['running_id_text'] + '_confusion_matrix_metrics.xlsx'
    )
    
    sheet_name='metrics_summary'
    sheet_list = []
    sheet_list.append(['Metrics Results calculated by application', ''])
    sheet_list.append(['', ''])
    sheet_list.append(['Model', f'{ parameters["neural_network_model"]["model_name"]}'])
    sheet_list.append(['', ''])
    sheet_list.append(['Threshold',  f"{parameters['neural_network_model']['threshold']:.2f}"])
    sheet_list.append(['IoU Threshold',  f"{parameters['neural_network_model']['iou_threshold']:.2f}"])
    sheet_list.append(['Non-Maximum Supression',  f"{parameters['neural_network_model']['non_maximum_suppression']:.2f}"])
    sheet_list.append(['', ''])

    sheet_list.append(['TP / FP / FN / TN per Class', ''])
    cm_classes = classes[1:(number_of_classes+1)]

    # setting values of TP, FP, and FN per class
    sheet_list.append(['Class', 'TP', 'FP', 'FN', 'TN'])
    for i, class_name in enumerate(classes[1:(number_of_classes+1)]):
    # for i, class_name in enumerate(classes):
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
    # for i, class_name in enumerate(classes):
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

def get_original_boxes(dataset_test_original_boxes, image_id):
    for image_with_original_boxes in dataset_test_original_boxes:
        if image_with_original_boxes['image_id'] == image_id:
            return image_with_original_boxes['image_boxes'], image_with_original_boxes['image_labels']
            
    return None
