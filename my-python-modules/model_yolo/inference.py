# Inference YOLO model 

from common.metrics import *
from common.entity.ImageAnnotation import ImageAnnotation

from model_yolo.yolo_utils import *

def inference_yolo_model(parameters, device, model, yolo_model_name):

    # setting label of the yolo model name 
    yolo_model_name_key = 'model_' + yolo_model_name

    # getting classes 
    classes =  parameters['neural_network_model']['classes']
    print(f'{yolo_model_name_key} - classes: {classes}')

    # creating predictions dictionary of all images 
    all_predictions = {}

    # Run batched inference on a list of images   
    image_dataset_folder_test_images = os.path.join(
        parameters['processing']['image_dataset_folder_test'],
        'images',
    )
    image_dataset_folder_test_labels = os.path.join(
        parameters['processing']['image_dataset_folder_test'],
        'labels',
    )
    logging_info(f'Test image dataset folder: {image_dataset_folder_test_images}')
    logging_info(f'')

    # get list of all test images for inference 
    test_images = Utils.get_files_with_extensions(image_dataset_folder_test_images, '.jpg')
    test_images_with_path = []
    for test_image in test_images:
        test_image = os.path.join(
            image_dataset_folder_test_images,
            test_image
        )
        test_images_with_path.append(test_image)
    
    data_file_yaml = os.path.join(
        parameters['processing']['research_root_folder'],
        parameters['processing']['project_name_folder'],
        parameters[yolo_model_name_key]['processing']['yolo_yaml_filename_test']
    )

    # creating metric object     
    inference_metric = Metrics(
        model=parameters[yolo_model_name_key]['neural_network_model']['model_name'],
        number_of_classes=parameters[yolo_model_name_key]['neural_network_model']['number_of_classes'],
    )

    # logging_info(f'len test_image #{len(test_images_with_path)}')

    count = 0    
    for test_image in test_images_with_path:
        count += 1
        results = model.predict(
            data=data_file_yaml, 
            source=test_image, 
            imgsz=parameters[yolo_model_name_key]['input']['input_dataset']['input_image_size'],
            project=parameters['test_results']['results_folder'],
            conf=parameters[yolo_model_name_key]['neural_network_model']['threshold'],
            iou=parameters[yolo_model_name_key]['neural_network_model']['iou_threshold'],
            device=device,
            verbose=True,
            show=False,
            save=True,
            save_conf=True,
            plots=True,
        )

        # extracting parts of path and image filename 
        path, filename_with_extension, filename, extension = Utils.get_filename(test_image)

        # setting the annotation filename 
        path_and_filename_yolo_annotation = os.path.join(
            image_dataset_folder_test_labels, 
            filename + '.txt'
            )

        # logging_info(f'-'*70)
        logging_info(f'Test image #{count} {filename_with_extension}')
        # logging_info(f'test_label #{count} {path_and_filename_yolo_annotation}')

        # getting all annotations of the image 
        image_annotation = ImageAnnotation()
        height = width = parameters[yolo_model_name_key]['input']['input_dataset']['input_image_size']
        image_annotation.get_annotation_file_in_yolo_v5_format(
            path_and_filename_yolo_annotation, classes, height, width
        )
        # logging_info(f'image_annotation: {image_annotation.to_string()}')

        # getting target bounding boxes 
        targets = image_annotation.get_tensor_target(classes)
        # logging_info(f'target annotated: {targets}')

        # setting target and predicted bounding boxes for metrics 
        # new_targets = []
        # item_target = {
        #     "boxes": target['boxes'],
        #     "labels": target['labels']
        #     }
        # new_targets.append(item_target)

        new_predicteds = []
        for result in results:
            result = result.to('cpu')
            for box in result.boxes:    
                # logging_info(f'boxes predicted: {box.xyxy}')
                item_predicted = {
                    "boxes": box.xyxy,
                    "scores": box.conf,
                    "labels": torch.tensor(box.cls, dtype=torch.int),
                    }
                new_predicteds.append(item_predicted)

        # logging_info(f'targets: {targets}')
        # logging_info(f'new_predicteds: {new_predicteds}')

        # setting target and predicted bounding boxes for metrics
        inference_metric.set_details_of_inferenced_image(
            filename_with_extension, targets, new_predicteds) 
        # inference_metric.target.extend(target)
        # inference_metric.preds.extend(new_predicteds)
        # logging_info(f'inference_metric.to_string: {inference_metric.to_string()}')
        # logging_info(f'--------------------------------------------------')

        # adding predictions 
        key = filename_with_extension
        value = (targets, new_predicteds)
        all_predictions[key] = value 

    # merging all image rsults to just one folder
    merge_image_results(parameters)

    # Computing Confusion Matrix 
    model_name = parameters[yolo_model_name_key]['neural_network_model']['model_name']
    num_classes = parameters[yolo_model_name_key]['neural_network_model']['number_of_classes'] + 1
    threshold = parameters[yolo_model_name_key]['neural_network_model']['threshold']
    iou_threshold = parameters[yolo_model_name_key]['neural_network_model']['iou_threshold']
    metrics_folder = parameters['test_results']['metrics_folder']
    running_id_text = parameters['processing']['running_id_text']
    tested_folder = parameters['test_results']['inferenced_image_folder']
    inference_metric.compute_confusion_matrix(model_name, num_classes, threshold, iou_threshold, 
                                              metrics_folder, running_id_text, tested_folder)
    inference_metric.confusion_matrix_to_string()

    print(f'{yolo_model_name_key} - classes: {classes}')

    # saving confusion matrix plots 
    title =  'Full Confusion Matrix' + \
             ' - Model: ' + parameters[yolo_model_name_key]['neural_network_model']['model_name'] + \
             '   # images:' + str(inference_metric.confusion_matrix_summary['number_of_images'])
    title += LINE_FEED + \
             'Confidence threshold: ' + str(parameters[yolo_model_name_key]['neural_network_model']['threshold']) + \
             '   IoU threshold: ' + str(parameters[yolo_model_name_key]['neural_network_model']['iou_threshold']) + \
             '   Non-maximum Supression: ' + str(parameters[yolo_model_name_key]['neural_network_model']['non_maximum_suppression'])

    path_and_filename = os.path.join(parameters['test_results']['metrics_folder'], 
        parameters[yolo_model_name_key]['neural_network_model']['model_name'] + \
        '_' + parameters['processing']['running_id_text'] + '_confusion_matrix_full.png'
    )
    number_of_classes = parameters[yolo_model_name_key]['neural_network_model']['number_of_classes']
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
        parameters[yolo_model_name_key]['neural_network_model']['model_name'] + \
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
             ' - Model: ' + parameters[yolo_model_name_key]['neural_network_model']['model_name'] + \
             '   # images:' + str(inference_metric.confusion_matrix_summary['number_of_images'])
    title += LINE_FEED + \
             'Confidence threshold: ' + str(parameters[yolo_model_name_key]['neural_network_model']['threshold']) + \
             '   IoU threshold: ' + str(parameters[yolo_model_name_key]['neural_network_model']['iou_threshold']) + \
             '   Non-maximum Supression: ' + str(parameters[yolo_model_name_key]['neural_network_model']['non_maximum_suppression'])
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'], 
        parameters[yolo_model_name_key]['neural_network_model']['model_name'] + \
        '_' + parameters['processing']['running_id_text'] + '_confusion_matrix_full_normalized.png'
    )
    cm_classes = classes[0:(number_of_classes+1)]
    x_labels_names = cm_classes.copy()
    y_labels_names = cm_classes.copy()
    x_labels_names.append('Incorrect prediction')    
    y_labels_names.append('Undetected objects')
    format='.2f'
    Utils.save_plot_confusion_matrix(inference_metric.full_confusion_matrix_normalized, 
                                     path_and_filename, title, format,
                                     x_labels_names, y_labels_names)
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'], 
        parameters[yolo_model_name_key]['neural_network_model']['model_name'] + \
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
        parameters[yolo_model_name_key]['neural_network_model']['model_name'] + \
        '_' + parameters['processing']['running_id_text'] + '_confusion_matrix_metrics.xlsx'
    )
    
    sheet_name='metrics_summary'
    sheet_list = []
    sheet_list.append(['Metrics Results calculated by application', ''])
    sheet_list.append(['', ''])
    sheet_list.append(['Model', f'{ parameters[yolo_model_name_key]["neural_network_model"]["model_name"]}'])
    sheet_list.append(['', ''])
    sheet_list.append(['Threshold',  f"{parameters[yolo_model_name_key]['neural_network_model']['threshold']:.2f}"])
    sheet_list.append(['IoU Threshold prediction',  f"{parameters[yolo_model_name_key]['neural_network_model']['iou_threshold']:.2f}"])
    # sheet_list.append(['IoU Threshold validation',  f"{parameters[yolo_model_name_key]['neural_network_model']['iou_threshold_for_validation']:.2f}"])
    sheet_list.append(['Non-Maximum Supression',  f"{parameters[yolo_model_name_key]['neural_network_model']['non_maximum_suppression']:.2f}"])
    sheet_list.append(['', ''])

    sheet_list.append(['TP / FP / FN per Class', ''])
    cm_classes = classes[1:(number_of_classes+1)]

    # setting values of TP, FP, FN, and TN per class
    sheet_list.append(['Class', 'TP', 'FP', 'FN', 'TN'])
    sheet_list.append(['Class', 'TP', 'FP', 'FN'])
    # for i, class_name in enumerate(classes[1:6]):
    for i, class_name in enumerate(classes[1:number_of_classes+1]):
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

    # returning predictions 
    return all_predictions


# def inference_yolo_model_ultralytics(parameters, device, model, yolo_model_name):
  
#     logging_info(f'')
#     logging_info(f'Testing images using model val from Ultralytics')
#     logging_info(f'')

#     # getting classes 
#     classes =  parameters['neural_network_model']['classes']

#     # setting label of the yolo model name 
#     yolo_model_name_key = 'model_' + yolo_model_name

#     # creating predictions dictionary of all images 
#     all_predictions = {}

#     # Run batched inference on a list of images   
#     image_dataset_folder_test_images = os.path.join(
#         parameters['processing']['image_dataset_folder_test'],
#         'images',
#     )
#     image_dataset_folder_test_labels = os.path.join(
#         parameters['processing']['image_dataset_folder_test'],
#         'labels',
#     )
#     logging_info(f'Test image dataset folder: {image_dataset_folder_test_images}')
#     logging_info(f'')

#     # get list of all test images for inference 
#     test_images = Utils.get_files_with_extensions(image_dataset_folder_test_images, '.jpg')
#     test_images_with_path = []
#     for test_image in test_images:
#         test_image = os.path.join(
#             image_dataset_folder_test_images,
#             test_image
#         )
#         test_images_with_path.append(test_image)
    
#     data_file_yaml = os.path.join(
#         parameters['processing']['research_root_folder'],
#         parameters['processing']['project_name_folder'],
#         parameters[yolo_model_name_key]['processing']['yolo_yaml_filename_test']
#     )

#     # running test in test image dataset by 'model.val' method
#     metric_results = model.val(
#         data=data_file_yaml, 
#         imgsz=parameters[yolo_model_name_key]['input']['input_dataset']['input_image_size'],
#         project=parameters['test_results']['results_folder'],
#         conf=parameters[yolo_model_name_key]['neural_network_model']['threshold'],
#         iou=parameters[yolo_model_name_key]['neural_network_model']['iou_threshold'],
#         max_det=300,
#         nms=True,
#         device=device,
#         verbose=True,
#         show=False,
#         save=True,
#         save_conf=True,
#         plots=True,    
#         save_json=True,
#         save_txt=True,
#         save_crop=True,
#         )

#     # save_hybrid=True,
#     # save_frames=True,
#     # save_crop=False,        
    
#     logging_info(f'metric_results.box: {metric_results.box}')

#     # setting class names
#     number_of_classes = parameters['neural_network_model']['number_of_classes']
#     cm_classes = classes[0:(number_of_classes+1)]
#     x_labels_names = cm_classes.copy()
#     y_labels_names = cm_classes.copy()
#     x_labels_names.append('??background??')    
#     y_labels_names.append('??background??')

#     # saving confusion matrix 
#     path_and_filename = os.path.join(
#         parameters['test_results']['metrics_folder'], 
#         parameters['neural_network_model']['model_name'] + '_' + \
#         parameters['processing']['running_id_text'] + '_val_ultralytics_confusion_matrix_full.xlsx'
#     )
#     Utils.save_confusion_matrix_excel(metric_results.confusion_matrix.matrix,
#                                       path_and_filename, 
#                                       x_labels_names, y_labels_names, 
#                                       [], [], [], []
#     )
                                      

#     # logging_info(f'metric_results: {metric_results}')
#     # logging_info(f'--------------------')
#     # logging_info(f'metric_results.box: {metric_results.box}')
#     # logging_info(f'--------------------')

#     # saving metrics from confusion matrix
#     path_and_filename = os.path.join(
#         parameters['test_results']['metrics_folder'],        
#         parameters['neural_network_model']['model_name'] + '_' + \
#         parameters['processing']['running_id_text'] + '_val_ultralytics_confusion_matrix_metrics.xlsx'
#     )
#     sheet_name='summary_metrics'
#     sheet_list = []
#     sheet_list.append(['Metrics Results calculated by Ultralytics', ''])
#     sheet_list.append(['', ''])
#     sheet_list.append(['Model', f'{ parameters["neural_network_model"]["model_name"]}'])
#     sheet_list.append(['', ''])

#     # computing TP, FP from confusion matrix 
#     logging_info(f'metric_results.confusion_matrix.tp_fp: {metric_results.confusion_matrix.tp_fp()}')
#     tp_fp = metric_results.confusion_matrix.tp_fp()
#     tp = tp_fp[0]
#     tp_total = tp.sum()
#     fp = tp_fp[1]
#     fp_total = fp.sum()
#     sheet_list.append(['TP_FP', tp_fp])
#     sheet_list.append(['TP', tp])
#     sheet_list.append(['FP', fp])
#     sheet_list.append(['TP', f'{tp_total:.0f}'])
#     sheet_list.append(['FP', f'{fp_total:.0f}'])
#     sheet_list.append(['FN', f'{0:.0f}'])
#     sheet_list.append(['TN', f'{0:.0f}'])
#     sheet_list.append(['', ''])

#     # computing f1-score 
#     f1_score = np.mean(metric_results.box.f1)
#     # logging_info(f'f1: {metric_results.box.f1}')
#     # f1_score_computed = 2 * (metric_results.box.mp * metric_results.box.mr) / (metric_results.box.mp + metric_results.box.mr) 
#     # logging_info(f'f1_score: {f1_score}')
#     # logging_info(f'f1_score_computed: {f1_score_computed}')

#     logging_info(f'metric_results.box: {metric_results.box}')

#     # metric measures 
#     sheet_list.append(['Metric measures', ''])
#     sheet_list.append(['Accuracy', f'{0:.8f}'])
#     sheet_list.append(['Precision', f'{ metric_results.box.mp:.8f}'])
#     sheet_list.append(['Recall', f'{ metric_results.box.mr:.8f}'])
#     sheet_list.append(['F1-score', f'{f1_score:.8f}'])
#     sheet_list.append(['Dice', f'{0:.8f}'])
#     sheet_list.append(['map', f'{metric_results.box.map:.8f}'])
#     sheet_list.append(['map50', f'{metric_results.box.map50:.8f}'])
#     sheet_list.append(['map75', f'{metric_results.box.map75:.8f}']) 
#     sheet_list.append(['', ''])

#     sheet_list.append(['Metric measures per class', ''])
#     sheet_list.append(['', ''])
#     sheet_list.append(['Class', 'Precision', 'Recall (Revocação)', 'F1-Score'])
#     for i, class_index in enumerate(metric_results.box.ap_class_index):
#         logging_info(f'rubens i: {i}  class_index: {class_index}')
#         class_name = classes[class_index]
#         sheet_list.append(
#             [class_name, 
#              f'{metric_results.box.p[i]:.8f}',
#              f'{metric_results.box.r[i]:.8f}',
#              f'{metric_results.box.f1[i]:.8f}', 
#             ]
#         )

#     sheet_list.append(
#         ['Model', 
#          f'{metric_results.box.mp:.8f}',
#          f'{metric_results.box.mr:.8f}',
#          f'{f1_score:.8f}', 
#         ]
#     )

#     # saving metrics sheet
#     Utils.save_metrics_excel(path_and_filename, sheet_name, sheet_list)
#     logging_sheet(sheet_list)
       
def merge_image_results(parameters):

    # setting parameters 
    results_folder = os.path.join(parameters['test_results']['results_folder'])
    folder_prefix = 'predict'
    test_image_folder = os.path.join(parameters['test_results']['inferenced_image_folder'])
    test_image_sufix = '_predicted'

    # copy all image files from results to one specific folder
    YoloUtils.merge_image_results_to_one_folder(results_folder, folder_prefix, test_image_folder, test_image_sufix)
