"""
Institution: Institute of Computing - University of Campinas (IC/Unicamp)
Project: White Mold 
Description: Implements the Model Ensembling for step of inference to previous evaluated models.
Author: Rubens de Castro Pereira
Advisors: 
    Prof. Dr. Hélio Pedrini - advisor at IC-Unicamp
    Prof. Dr. Díbio Leandro Borges - coadvisor at CIC-UnB
    Prof. Dr. Murillo Lobo Jr. - coadvisor at Embrapa Rice and Beans
Date: 17/11/2024
Version: 1.0
This implementation is based on this notebook: 
"""

# Basic python and ML Libraries
import os
from datetime import datetime
import shutil

# torchvision libraries
import torch
from transformers import DetrImageProcessor


# these are the helper libraries imported.

# Importing python modules
from common.manage_log import *
from common.tasks import Tasks
from common.entity.ImageAnnotation import ImageAnnotation
from common.entity.AnnotationsStatistic import AnnotationsStatistic

# from model import *
# from dataset import *
# from inference import * 

# ###########################################
# Constants
# ###########################################
LINE_FEED = '\n'
NEW_FILE = True

# ###########################################
# Application Methods
# ###########################################

# ###########################################
# Methods of Level 1
# ###########################################

def main():
    """
    Main method that perform inference of the neural network model.

    All values of the parameters used here are defined in the external file "wm_model_faster_rcnn_parameters.json".
    
    """
 
    # creating Tasks object 
    processing_tasks = Tasks()

    # setting dictionary initial parameters for processing
    full_path_project = '/home/lovelace/proj/proj939/rubenscp/research/white-mold-applications/wm-model-detr'

    # getting application parameters 
    processing_tasks.start_task('Getting application parameters')
    parameters_filename = 'wm_model_detr_parameters.json'
    parameters = get_parameters(full_path_project, parameters_filename)
    processing_tasks.finish_task('Getting application parameters')

    # setting new values of parameters according of initial parameters
    processing_tasks.start_task('Setting input image folders')
    set_input_image_folders(parameters)
    processing_tasks.finish_task('Setting input image folders')

    # getting last running id
    processing_tasks.start_task('Getting running id')
    running_id = get_running_id(parameters)
    processing_tasks.finish_task('Getting running id')

    # setting output folder results
    processing_tasks.start_task('Setting result folders')
    set_result_folders(parameters)
    processing_tasks.finish_task('Setting result folders')
    
    # creating log file 
    processing_tasks.start_task('Creating log file')
    logging_create_log(
        parameters['test_results']['log_folder'], 
        parameters['test_results']['log_filename']
    )
    processing_tasks.finish_task('Creating log file')

    logging_info('White Mold Research')
    logging_info('Inference of the model DETR' + LINE_FEED)

    logging_info(f'')
    logging_info(f'>> Set input image folders')
    logging_info(f'')
    logging_info(f'>> Get running id')
    logging_info(f'running id: {str(running_id)}')   
    logging_info(f'')
    logging_info(f'>> Set result folders')

    # creating yaml file with parameters used by Ultralytics
    # processing_tasks.start_task('Creating yaml file for ultralytics')
    # create_yaml_file_for_ultralytics(parameters)
    # processing_tasks.finish_task('Creating yaml file for ultralytics')

    # getting device CUDA
    processing_tasks.start_task('Getting device CUDA')
    device = get_device(parameters)
    processing_tasks.finish_task('Getting device CUDA')
    
    # creating new instance of parameters file related to current running
    processing_tasks.start_task('Saving processing parameters')
    save_processing_parameters(parameters_filename, parameters)
    processing_tasks.finish_task('Saving processing parameters')
   
    # copying pre-trained files 
    processing_tasks.start_task('Copying pre-trained files')
    copy_pretrained_model_files(parameters)
    processing_tasks.finish_task('Copying pre-trained files')     

    # copying weights file produced by training step 
    processing_tasks.start_task('Copying weights file used in inference')
    copy_weights_file(parameters)
    processing_tasks.finish_task('Copying weights file used in inference')

    # loading datasets and dataloaders of image dataset for processing
    processing_tasks.start_task('Loading test dataset of image dataset')
    dataset_type = 'test'
    dataset_test, dataset_test_original_boxes, processor = get_dataset(parameters, device, dataset_type)
    processing_tasks.finish_task('Loading test dataset of image dataset')

    # creating neural network model 
    processing_tasks.start_task('Creating neural network model')
    model = get_neural_network_model(parameters, device)
    processing_tasks.finish_task('Creating neural network model')

    # getting statistics of input dataset 
    if parameters['processing']['show_statistics_of_input_dataset']:
        processing_tasks.start_task('Getting statistics of input dataset')
        annotation_statistics = get_input_dataset_statistics(parameters)
        show_input_dataset_statistics(parameters, annotation_statistics)
        processing_tasks.finish_task('Getting statistics of input dataset')    

    # inference the neural netowrk model
    processing_tasks.start_task('Running prediction on the test images dataset')
    inference_detr_model(parameters, device, model, processor, dataset_test, dataset_test_original_boxes)
    processing_tasks.finish_task('Running prediction on the test images dataset')

    # showing input dataset statistics
    # if parameters['processing']['show_statistics_of_input_dataset']:
    #     show_input_dataset_statistics(parameters, annotation_statistics)

    # merging all image rsults to just one folder
    # merge_image_results(parameters)

    # finishing model training 
    logging_info('')
    logging_info('Finished the test of the model DETR' + LINE_FEED)

    # printing tasks summary 
    processing_tasks.finish_processing()
    logging_info(processing_tasks.to_string())

    # copying processing files to log folder 
    # copy_processing_files_to_log(parameters)


# ###########################################
# Methods of Level 2
# ###########################################

def get_parameters(full_path_project, parameters_filename):
    '''
    Get dictionary parameters for processing
    '''    
    # getting parameters 
    path_and_parameters_filename = os.path.join(full_path_project, parameters_filename)
    parameters = Utils.read_json_parameters(path_and_parameters_filename)

    # returning parameters 
    return parameters

def set_input_image_folders(parameters):
    '''
    Set folder name of input images dataset
    '''    
    
    # getting image dataset folder according processing parameters 
    input_image_size = str(parameters['input']['input_dataset']['input_image_size'])
    image_dataset_folder = os.path.join(
        parameters['processing']['research_root_folder'],
        parameters['input']['input_dataset']['input_dataset_path'],
        parameters['input']['input_dataset']['annotation_format'],
        input_image_size + 'x' + input_image_size,
    )

    # setting image dataset folder in processing parameters 
    parameters['processing']['image_dataset_folder'] = image_dataset_folder
    parameters['processing']['image_dataset_folder_train'] = \
        os.path.join(image_dataset_folder, 'train')
    parameters['processing']['image_dataset_folder_valid'] = \
        os.path.join(image_dataset_folder, 'valid')
    parameters['processing']['image_dataset_folder_test'] = \
        os.path.join(image_dataset_folder, 'test')

def get_running_id(parameters):
    '''
    Get last running id to calculate the current id
    '''    

    # setting control filename 
    running_control_filename = os.path.join(
        parameters['processing']['research_root_folder'],
        parameters['processing']['project_name_folder'],
        parameters['processing']['running_control_filename'],
    )

    # getting control info 
    running_control = Utils.read_json_parameters(running_control_filename)

    # calculating the current running id 
    running_control['last_running_id'] = int(running_control['last_running_id']) + 1

    # updating running control file 
    running_id = int(running_control['last_running_id'])

    # saving file 
    Utils.save_text_file(running_control_filename, \
                         Utils.get_pretty_json(running_control), 
                         NEW_FILE)

    # updating running id in the processing parameters 
    parameters['processing']['running_id'] = running_id
    parameters['processing']['running_id_text'] = 'running-' + f'{running_id:04}'

    # returning the current running id
    return running_id

def set_result_folders(parameters):
    '''
    Set folder name of output results
    '''

    # resetting training results 
    parameters['training_results'] = {}

    # creating results folders 
    main_folder = os.path.join(
        parameters['processing']['research_root_folder'],     
        parameters['test_results']['main_folder']
    )
    parameters['test_results']['main_folder'] = main_folder
    Utils.create_directory(main_folder)

    # setting and creating model folder 
    parameters['test_results']['model_folder'] = parameters['neural_network_model']['model_name']
    model_folder = os.path.join(
        main_folder,
        parameters['test_results']['model_folder']
    )
    parameters['test_results']['model_folder'] = model_folder
    Utils.create_directory(model_folder)

    # setting and creating experiment folder
    experiment_folder = os.path.join(
        model_folder,
        parameters['input']['experiment']['id']
    )
    parameters['test_results']['experiment_folder'] = experiment_folder
    Utils.create_directory(experiment_folder)

    # setting and creating action folder of training
    action_folder = os.path.join(
        experiment_folder,
        parameters['test_results']['action_folder']
    )
    parameters['test_results']['action_folder'] = action_folder
    Utils.create_directory(action_folder)

    # setting and creating running folder 
    running_id = parameters['processing']['running_id']
    running_id_text = 'running-' + f'{running_id:04}'
    input_image_size = str(parameters['input']['input_dataset']['input_image_size'])
    parameters['test_results']['running_folder'] = running_id_text + "-" + input_image_size + 'x' + input_image_size   
    running_folder = os.path.join(
        action_folder,
        parameters['test_results']['running_folder']
    )
    parameters['test_results']['running_folder'] = running_folder
    Utils.create_directory(running_folder)

    # setting and creating others specific folders
    processing_parameters_folder = os.path.join(
        running_folder,
        parameters['test_results']['processing_parameters_folder']
    )
    parameters['test_results']['processing_parameters_folder'] = processing_parameters_folder
    Utils.create_directory(processing_parameters_folder)

    pretrained_model_folder = os.path.join(
        running_folder,
        parameters['test_results']['pretrained_model_folder']
    )
    parameters['test_results']['pretrained_model_folder'] = pretrained_model_folder
    Utils.create_directory(pretrained_model_folder)   

    weights_folder = os.path.join(
        running_folder,
        parameters['test_results']['weights_folder']
    )
    parameters['test_results']['weights_folder'] = weights_folder
    Utils.create_directory(weights_folder)

    metrics_folder = os.path.join(
        running_folder,
        parameters['test_results']['metrics_folder']
    )
    parameters['test_results']['metrics_folder'] = metrics_folder
    Utils.create_directory(metrics_folder)

    inferenced_image_folder = os.path.join(
        running_folder,
        parameters['test_results']['inferenced_image_folder']
    )
    parameters['test_results']['inferenced_image_folder'] = inferenced_image_folder
    Utils.create_directory(inferenced_image_folder)

    log_folder = os.path.join(
        running_folder,
        parameters['test_results']['log_folder']
    )
    parameters['test_results']['log_folder'] = log_folder
    Utils.create_directory(log_folder)

    results_folder = os.path.join(
        running_folder,
        parameters['test_results']['results_folder']
    )
    parameters['test_results']['results_folder'] = results_folder
    Utils.create_directory(results_folder)

# def create_yaml_file_for_ultralytics(parameters):

#     # preparing parameters 
#     yolo_v8_yaml_filename = parameters['processing']['yolo_v8_yaml_filename_test']
#     path_and_filename_white_mold_yaml = os.path.join(
#         parameters['processing']['research_root_folder'],
#         parameters['processing']['project_name_folder'],
#         yolo_v8_yaml_filename
#     )
#     image_dataset_folder = parameters['processing']['image_dataset_folder']
#     number_of_classes = parameters['neural_network_model']['number_of_classes']
#     classes = (parameters['neural_network_model']['classes'])[:(number_of_classes+1)]
   
#     # creating yaml file 
#     create_project_yaml_file_for_test(
#         path_and_filename_white_mold_yaml,
#         image_dataset_folder,
#         classes,    
#     )

def get_device(parameters):
    '''
    Get device CUDA to train models
    '''    

    logging_info(f'')
    logging_info(f'>> Get device')

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    parameters['processing']['device'] = f'{device}'
    
    logging_info(f'Device: {device}')
    
    # returning current device 
    return device 

def save_processing_parameters(parameters_filename, parameters):
    '''
    Update parameters file of the processing
    '''

    logging_info(f'')
    logging_info(f'>> Save processing parameters of this running')

    # setting full path and log folder  to write parameters file 
    path_and_parameters_filename = os.path.join(
        parameters['test_results']['processing_parameters_folder'], 
        parameters_filename)

    # saving current processing parameters in the log folder 
    Utils.save_text_file(path_and_parameters_filename, \
                        Utils.get_pretty_json(parameters), 
                        NEW_FILE)

def copy_pretrained_model_files(parameters):
    '''
    Copy pretrained model files for this running
    '''    

    input_path = parameters['neural_network_model']['pretrained_model_path']
    output_path = parameters['test_results']['pretrained_model_folder']

    filename = 'config.json'
    Utils.copy_file_same_name(filename, input_path, output_path)
    filename = 'model.safetensors'
    Utils.copy_file_same_name(filename, input_path, output_path)
    filename = 'preprocessor_config.json'
    Utils.copy_file_same_name(filename, input_path, output_path)
    filename = 'pytorch_model.bin'
    Utils.copy_file_same_name(filename, input_path, output_path)

def copy_weights_file(parameters):
    '''
    Copying weights file to inference step
    '''

    logging_info(f'')
    logging_info(f'>> Copy weights file of the model for inference')
    logging_info(f"Folder name: {parameters['input']['inference']['weights_folder']}")
    logging_info(f"Filename   : {parameters['input']['inference']['weights_filename']}")

    Utils.copy_file_same_name(
        parameters['input']['inference']['weights_filename'],
        parameters['input']['inference']['weights_folder'],
        parameters['test_results']['weights_folder']
    )

def get_dataset(parameters, device, dataset_type):
    '''
    Get datasets and dataloaders of testing from image dataset 
    '''

    logging_info(f'Get dataset type: {dataset_type}')

    # getting image dataset folders
    if dataset_type == 'train':
        image_dataset_folder = parameters['processing']['image_dataset_folder_train']
    elif dataset_type == 'valid':
        image_dataset_folder = parameters['processing']['image_dataset_folder_valid']
    elif dataset_type == 'test':
        image_dataset_folder = parameters['processing']['image_dataset_folder_test']

    # setting parameters for training and validation datasets
    pretrained_model_name_or_path = parameters['neural_network_model']['pretrained_model_path']
    cache_dir = parameters['neural_network_model']['model_cache_dir']

    # getting image processor
    processor = DetrImageProcessor.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            cache_dir=cache_dir,
            revision="no_timm",
            local_files_only=True,            
        )
    logging.info(f'processor: {processor}')

    # getting datasets for training and validation
    dataset_test  = CocoDetection(img_folder=image_dataset_folder, 
                                  processor=processor,
                                  dataset_type=dataset_type)

    logging.info(f'Getting datasets')
    logging.info(f'Number of testing images   : {len(dataset_test)}')
    logging_info(f'')

    # getting annotations of the dataset
    dataset_test_original_boxes = get_original_boxes_of_dataset(parameters, dataset_type)
    logging_info(f'Number of original boxes of testing images: {len(dataset_test_original_boxes)}')
    logging_info(f'dataset_test_original_boxes: {dataset_test_original_boxes}')

    # returning dataset for processing 
    return dataset_test, dataset_test_original_boxes, processor

def get_neural_network_model(parameters, device):
    '''
    Get neural network model
    '''

    logging_info(f'')
    logging_info(f'>> Get neural network model')
    
    model_name = parameters['neural_network_model']['model_name']

    logging_info(f'Model used: {model_name}')

    learning_rate = parameters['neural_network_model']['learning_rate_initial']
    learning_rate_backbone = parameters['neural_network_model']['learning_rate_backbone']
    weight_decay = parameters['neural_network_model']['weight_decay']
    num_labels = parameters['neural_network_model']['number_of_classes']
    pretrained_model_name_or_path = parameters['neural_network_model']['pretrained_model_path']
    cache_dir = parameters['neural_network_model']['model_cache_dir']   
    model = Detr(lr=learning_rate, 
                 lr_backbone=learning_rate_backbone, 
                 weight_decay=weight_decay,
                 pretrained_model_name_or_path=pretrained_model_name_or_path,
                 cache_dir=cache_dir,
                 num_labels=num_labels,
                 train_dataloader=None,
                 val_dataloader=None)

    # loading weights of the model trained in before step
    path_and_weigths_filename = os.path.join(
        parameters['input']['inference']['weights_folder'],
        parameters['input']['inference']['weights_filename'],
    )      
   
    logging_info(f'')
    logging_info(f'Loading weights file: {path_and_weigths_filename}')
    logging_info(f'')
    model.load_state_dict(torch.load(path_and_weigths_filename))
    # model.load_state_dict(torch.load(MODEL_DIR+'objectdetection.pth'))

    # moving model into GPU 
    model = model.to(device)

    number_of_parameters = count_parameters(model)
    logging.info(f'Number of model parameters: {number_of_parameters}')    

    num_layers = compute_num_layers(model)
    logging_info(f'Number of layers: {num_layers}')

    logging.info(f'{model}')

    # returning neural network model
    return model

# getting statistics of input dataset 
def get_input_dataset_statistics(parameters):
    
    annotation_statistics = AnnotationsStatistic()
    # steps = ['train', 'valid', 'test'] 
    steps = ['test'] 
    annotation_statistics.processing_statistics(parameters, steps)
    return annotation_statistics

def show_input_dataset_statistics(parameters, annotation_statistics):

    logging_info(f'Input dataset statistic')
    logging_info(annotation_statistics.to_string())
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'],
        parameters['neural_network_model']['model_name'] + \
        '_' + parameters['processing']['running_id_text'] + '_annotations_statistics.xlsx',
    )
    annotation_format = parameters['input']['input_dataset']['annotation_format']
    input_image_size = parameters['input']['input_dataset']['input_image_size']
    classes = (parameters['neural_network_model']['classes'])[1:5]
    annotation_statistics.save_annotations_statistics(
        path_and_filename,
        annotation_format,
        input_image_size,
        classes
    )

def inference_detr_model(parameters, device, model, processor, dataset_test, dataset_test_original_boxes):
    '''
    Execute inference of the neural network model
    '''
    inference_detr_model_with_dataset_test(parameters, device, model, processor, dataset_test, dataset_test_original_boxes)


# def inference_yolo_model(parameters, device, model, dataset_test):
#     '''
#     Execute inference of the neural network model
#     Confusion matrix of YOLOv8:
#     - https://medium.com/@a0922/confusion-matrix-of-yolov8-97fd7ff0074e
#     '''

#     logging_info(f'')
#     logging_info(f'Testing images using my class Metrics')
#     logging_info(f'')

#     # getting classes 
#     classes =  parameters['neural_network_model']['classes']

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
#         parameters['processing']['yolo_v8_yaml_filename_test']
#     )

#     # creating metric object     
#     inference_metric = Metrics(
#         model=parameters['neural_network_model']['model_name'],
#         number_of_classes=parameters['neural_network_model']['number_of_classes'],
#     )

#     # logging_info(f'len test_image #{len(test_images_with_path)}')

#     count = 0    
#     for test_image in test_images_with_path:
#         count += 1
#         results = model.predict(
#             data=data_file_yaml, 
#             source=test_image, 
#             imgsz=parameters['input']['input_dataset']['input_image_size'],
#             project=parameters['test_results']['results_folder'],
#             conf=parameters['neural_network_model']['threshold'],
#             iou=parameters['neural_network_model']['iou_threshold_for_prediction'],
#             device=device,
#             verbose=True,
#             show=False,
#             save=True,
#             save_conf=True,
#             plots=True,
#         )

#         # extracting parts of path and image filename 
#         path, filename_with_extension, filename, extension = Utils.get_filename(test_image)

#         # setting the annotation filename 
#         path_and_filename_yolo_annotation = os.path.join(
#             image_dataset_folder_test_labels, 
#             filename + '.txt'
#             )

#         # logging_info(f'-'*70)
#         logging_info(f'Test image #{count} {filename_with_extension}')
#         # logging_info(f'test_label #{count} {path_and_filename_yolo_annotation}')

#         # getting all annotations of the image 
#         image_annotation = ImageAnnotation()
#         height = width = parameters['input']['input_dataset']['input_image_size']
#         image_annotation.get_annotation_file_in_yolo_v5_format(
#             path_and_filename_yolo_annotation, classes, height, width
#         )
#         # logging_info(f'image_annotation: {image_annotation.to_string()}')

#         # getting target bounding boxes 
#         targets = image_annotation.get_tensor_target(classes)
#         # logging_info(f'target annotated: {targets}')

#         # setting target and predicted bounding boxes for metrics 
#         # new_targets = []
#         # item_target = {
#         #     "boxes": target['boxes'],
#         #     "labels": target['labels']
#         #     }
#         # new_targets.append(item_target)

#         new_predicteds = []
#         for result in results:
#             result = result.to('cpu')
#             for box in result.boxes:    
#                 # logging_info(f'boxes predicted: {box.xyxy}')
#                 item_predicted = {
#                     "boxes": box.xyxy,
#                     "scores": box.conf,
#                     "labels": torch.tensor(box.cls, dtype=torch.int),
#                     }
#                 new_predicteds.append(item_predicted)

#         # logging_info(f'targets: {targets}')
#         # logging_info(f'new_predicteds: {new_predicteds}')

#         # setting target and predicted bounding boxes for metrics
#         inference_metric.set_details_of_inferenced_image(
#             filename_with_extension, targets, new_predicteds) 
#         # inference_metric.target.extend(target)
#         # inference_metric.preds.extend(new_predicteds)
#         # logging_info(f'inference_metric.to_string: {inference_metric.to_string()}')
#         # logging_info(f'--------------------------------------------------')

#         # # Process results list
#         # for result in results:
#         #     logging_info(f'-'*50)
#         #     logging_info(f'result: {result}')
#         #     logging_info(f'result.boxes: {result.boxes}')
#         #     logging_info(f'result.masks: {result.masks}')
#         #     logging_info(f'result.probs: {result.probs}')
#         #     boxes = result.boxes  # Boxes object for bounding box outputs
#         #     masks = result.masks  # Masks object for segmentation masks outputs
#         #     keypoints = result.keypoints  # Keypoints object for pose outputs
#         #     probs = result.probs  # Probs object for classification outputs
#         #     result.show()  # display to screen
#         #     result.save(filename='result.jpg')  # save to disk


#     # merging all image rsults to just one folder
#     merge_image_results(parameters)

#     # Computing Confusion Matrix 
#     model_name = parameters['neural_network_model']['model_name']
#     num_classes = parameters['neural_network_model']['number_of_classes'] + 1
#     threshold = parameters['neural_network_model']['threshold']
#     iou_threshold = parameters['neural_network_model']['iou_threshold_for_prediction']
#     metrics_folder = parameters['test_results']['metrics_folder']
#     running_id_text = parameters['processing']['running_id_text']
#     tested_folder = parameters['test_results']['inferenced_image_folder']
#     inference_metric.compute_confusion_matrix(model_name, num_classes, threshold, iou_threshold, 
#                                               metrics_folder, running_id_text, tested_folder)
#     inference_metric.confusion_matrix_to_string()

#     # saving confusion matrix plots 
#     title =  'Full Confusion Matrix' + \
#              ' - Model: ' + parameters['neural_network_model']['model_name'] + \
#              '   # images:' + str(inference_metric.confusion_matrix_summary['number_of_images'])
#     title += LINE_FEED + \
#              'Confidence threshold: ' + str(parameters['neural_network_model']['threshold']) + \
#              '   IoU threshold: ' + str(parameters['neural_network_model']['iou_threshold_for_prediction']) + \
#              '   Non-maximum Supression: ' + str(parameters['neural_network_model']['non_maximum_suppression'])
#     # title += LINE_FEED + '  # bounding box -' + \
#     #          ' predicted with target: ' + str(inference_metric.confusion_matrix_summary['number_of_bounding_boxes_predicted_with_target']) + \
#     #          '   ghost predictions: ' + str(inference_metric.confusion_matrix_summary['number_of_ghost_predictions']) + \
#     #          '   undetected objects: ' + str(inference_metric.confusion_matrix_summary['number_of_undetected_objects'])

#     # logging_info(f'Bounding boxes target                : ' + \
#     #              f'{self.confusion_matrix_summary["Numsey$2023number_of_bounding_boxes_target"]}')
#     # logging_info(f'Bounding boxes predicted             : ' + \
#     #              f'{self.confusion_matrix_summary["number_of_bounding_boxes_predicted"]}')
#     # logging_info(f'Bounding boxes predicted with target : ' + \
#     #              f'{self.confusion_matrix_summary["number_of_bounding_boxes_predicted_with_target"]}')
#     # logging_info(f'Number of ghost preditions           : ' + \
#     #              f'{self.confusion_matrix_summary["number_of_ghost_predictions"]}')
#     # logging_info(f'Number of undetected objects         : ' + \
#     #              f'{self.confusion_matrix_summary["number_of_undetected_objects"]}')

#     path_and_filename = os.path.join(parameters['test_results']['metrics_folder'], 
#         parameters['neural_network_model']['model_name'] + \
#         '_' + parameters['processing']['running_id_text'] + '_confusion_matrix_full.png'
#     )
#     number_of_classes = parameters['neural_network_model']['number_of_classes']
#     cm_classes = classes[0:(number_of_classes+1)]
#     x_labels_names = cm_classes.copy()
#     y_labels_names = cm_classes.copy()
#     x_labels_names.append('Incorrect predictions')    
#     y_labels_names.append('Undetected objects')
#     format='.0f'
#     Utils.save_plot_confusion_matrix(inference_metric.full_confusion_matrix, 
#                                      path_and_filename, title, format,
#                                      x_labels_names, y_labels_names)
#     path_and_filename = os.path.join(
#         parameters['test_results']['metrics_folder'], 
#         parameters['neural_network_model']['model_name'] + \
#         '_' + parameters['processing']['running_id_text'] + '_confusion_matrix_full.xlsx'
#     )
#     Utils.save_confusion_matrix_excel(inference_metric.full_confusion_matrix,
#                                       path_and_filename, 
#                                       x_labels_names, y_labels_names, 
#                                       inference_metric.tp_per_class,
#                                       inference_metric.fp_per_class,
#                                       inference_metric.fn_per_class,
#                                       inference_metric.tn_per_class
#     )
        
#     # title = 'Confusion Matrix'
#     # path_and_filename = os.path.join(parameters['test_results']['metrics_folder'],
#     #     parameters['neural_network_model']['model_name'] + \
#     #     '_' + parameters['processing']['running_id_text'] + '_confusion_matrix.png'
#     # )
#     # cm_classes = classes[1:5]
#     # x_labels_names = cm_classes.copy()
#     # y_labels_names = cm_classes.copy()
#     # format='.0f'
#     # Utils.save_plot_confusion_matrix(inference_metric.confusion_matrix, 
#     #                                  path_and_filename, title, format,
#     #                                  x_labels_names, y_labels_names)
#     # path_and_filename = os.path.join(
#     #     parameters['test_results']['metrics_folder'], 
#     #     parameters['neural_network_model']['model_name'] + \
#     #     '_' + parameters['processing']['running_id_text'] + '_confusion_matrix.xlsx'
#     # )
#     # Utils.save_confusion_matrix_excel(inference_metric.confusion_matrix,
#     #                                   path_and_filename,
#     #                                   x_labels_names, y_labels_names, 
#     #                                   inference_metric.tp_per_class,
#     #                                   inference_metric.fp_per_class,
#     #                                   inference_metric.fn_per_class,
#     #                                   inference_metric.tn_per_class
#     # )

#     title =  'Full Confusion Matrix Normalized' + \
#              ' - Model: ' + parameters['neural_network_model']['model_name'] + \
#              '   # images:' + str(inference_metric.confusion_matrix_summary['number_of_images'])
#     title += LINE_FEED + \
#              'Confidence threshold: ' + str(parameters['neural_network_model']['threshold']) + \
#              '   IoU threshold: ' + str(parameters['neural_network_model']['iou_threshold_for_prediction']) + \
#              '   Non-maximum Supression: ' + str(parameters['neural_network_model']['non_maximum_suppression'])
#     # title += LINE_FEED + '  # bounding box -' + \
#     #          ' predicted with target: ' + str(inference_metric.confusion_matrix_summary['number_of_bounding_boxes_predicted_with_target']) + \
#     #          '   ghost predictions: ' + str(inference_metric.confusion_matrix_summary['number_of_ghost_predictions']) + \
#     #          '   undetected objects: ' + str(inference_metric.confusion_matrix_summary['number_of_undetected_objects'])
#     path_and_filename = os.path.join(
#         parameters['test_results']['metrics_folder'], 
#         parameters['neural_network_model']['model_name'] + \
#         '_' + parameters['processing']['running_id_text'] + '_confusion_matrix_full_normalized.png'
#     )
#     cm_classes = classes[0:(number_of_classes+1)]
#     x_labels_names = cm_classes.copy()
#     y_labels_names = cm_classes.copy()
#     x_labels_names.append('Incorrect prediction')    
#     y_labels_names.append('Undetected objects')
#     format='.2f'
#     Utils.save_plot_confusion_matrix(inference_metric.full_confusion_matrix_normalized, 
#                                      path_and_filename, title, format,
#                                      x_labels_names, y_labels_names)
#     path_and_filename = os.path.join(
#         parameters['test_results']['metrics_folder'], 
#         parameters['neural_network_model']['model_name'] + \
#         '_' + parameters['processing']['running_id_text'] + '_confusion_matrix_full_normalized.xlsx'
#     )
#     Utils.save_confusion_matrix_excel(inference_metric.full_confusion_matrix_normalized,
#                                       path_and_filename,
#                                       x_labels_names, y_labels_names, 
#                                       inference_metric.tp_per_class,
#                                       inference_metric.fp_per_class,
#                                       inference_metric.fn_per_class,
#                                       inference_metric.tn_per_class
#     )

#     # title =  'Confusion Matrix Normalized' + \
#     #          ' - Model: ' + parameters['neural_network_model']['model_name'] + \
#     #          '   # images:' + str(inference_metric.confusion_matrix_summary['number_of_images'])
#     # title += LINE_FEED + '  # bounding box -' + \
#     #          ' predicted with target: ' + str(inference_metric.confusion_matrix_summary['number_of_bounding_boxes_predicted_with_target']) + \
#     #          '   ghost predictions: ' + str(inference_metric.confusion_matrix_summary['number_of_ghost_predictions']) + \
#     #          '   undetected objects: ' + str(inference_metric.confusion_matrix_summary['number_of_undetected_objects'])
#     # path_and_filename = os.path.join(
#     #     parameters['test_results']['metrics_folder'], 
#     #     parameters['neural_network_model']['model_name'] + \
#     #     '_' + parameters['processing']['running_id_text'] + '_confusion_matrix_normalized.png'
#     # )
#     # cm_classes = classes[1:5]
#     # x_labels_names = cm_classes.copy()
#     # y_labels_names = cm_classes.copy()
#     # format='.2f'
#     # Utils.save_plot_confusion_matrix(inference_metric.confusion_matrix_normalized, 
#     #                                  path_and_filename, title, format,
#     #                                  x_labels_names, y_labels_names)
#     # path_and_filename = os.path.join(
#     #     parameters['test_results']['metrics_folder'], 
#     #     parameters['neural_network_model']['model_name'] + \
#     #     '_' + parameters['processing']['running_id_text'] + '_confusion_matrix_normalized.xlsx'
#     # )
#     # Utils.save_confusion_matrix_excel(inference_metric.confusion_matrix_normalized,
#     #                                   path_and_filename,
#     #                                   x_labels_names, y_labels_names, 
#     #                                   inference_metric.tp_per_class,
#     #                                   inference_metric.fp_per_class,
#     #                                   inference_metric.fn_per_class,
#     #                                   inference_metric.tn_per_class
#     # )


#    # saving metrics from confusion matrix
#     path_and_filename = os.path.join(
#         parameters['test_results']['metrics_folder'],
#         parameters['neural_network_model']['model_name'] + \
#         '_' + parameters['processing']['running_id_text'] + '_confusion_matrix_metrics.xlsx'
#     )
    
#     sheet_name='metrics_summary'
#     sheet_list = []
#     sheet_list.append(['Metrics Results calculated by application', ''])
#     sheet_list.append(['', ''])
#     sheet_list.append(['Model', f'{ parameters["neural_network_model"]["model_name"]}'])
#     sheet_list.append(['', ''])
#     sheet_list.append(['Threshold',  f"{parameters['neural_network_model']['threshold']:.2f}"])
#     sheet_list.append(['IoU Threshold prediction',  f"{parameters['neural_network_model']['iou_threshold_for_prediction']:.2f}"])
#     sheet_list.append(['IoU Threshold validation',  f"{parameters['neural_network_model']['iou_threshold_for_validation']:.2f}"])
#     sheet_list.append(['Non-Maximum Supression',  f"{parameters['neural_network_model']['non_maximum_suppression']:.2f}"])
#     sheet_list.append(['', ''])

#     sheet_list.append(['TP / FP / FN per Class', ''])
#     cm_classes = classes[1:(number_of_classes+1)]

#     # setting values of TP, FP, FN, and TN per class
#     sheet_list.append(['Class', 'TP', 'FP', 'FN', 'TN'])
#     sheet_list.append(['Class', 'TP', 'FP', 'FN'])
#     for i, class_name in enumerate(classes[1:5]):
#         row = [class_name, 
#                f'{inference_metric.tp_per_class[i]:.0f}',
#                f'{inference_metric.fp_per_class[i]:.0f}',
#                f'{inference_metric.fn_per_class[i]:.0f}',
#                f'{inference_metric.tn_per_class[i]:.0f}',
#               ]
#         sheet_list.append(row)

#     i += 1
#     row = ['Total',
#            f'{inference_metric.tp_model:.0f}',
#            f'{inference_metric.fp_model:.0f}',
#            f'{inference_metric.fn_model:.0f}',
#            f'{inference_metric.tn_model:.0f}',
#           ]
#     sheet_list.append(row)    
#     sheet_list.append(['', ''])

#     # setting values of metrics precision, recall, f1-score and dice per class
#     sheet_list.append(['Class', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Dice'])
#     for i, class_name in enumerate(classes[1:5]):
#         row = [class_name, 
#                f'{inference_metric.accuracy_per_class[i]:.8f}',
#                f'{inference_metric.precision_per_class[i]:.8f}',
#                f'{inference_metric.recall_per_class[i]:.8f}',
#                f'{inference_metric.f1_score_per_class[i]:.8f}',
#                f'{inference_metric.dice_per_class[i]:.8f}',
#               ]
#         sheet_list.append(row)

#     i += 1
#     row = ['Model Metrics',
#                f'{inference_metric.get_model_accuracy():.8f}',
#                f'{inference_metric.get_model_precision():.8f}',
#                f'{inference_metric.get_model_recall():.8f}',
#                f'{inference_metric.get_model_f1_score():.8f}',
#                f'{inference_metric.get_model_dice():.8f}',
#           ]
#     sheet_list.append(row)
#     sheet_list.append(['', ''])

#     # metric measures 
#     sheet_list.append(['Metric measures', ''])
#     sheet_list.append(['number_of_images', f'{inference_metric.confusion_matrix_summary["number_of_images"]:.0f}'])
#     sheet_list.append(['number_of_bounding_boxes_target', f'{inference_metric.confusion_matrix_summary["number_of_bounding_boxes_target"]:.0f}'])
#     sheet_list.append(['number_of_bounding_boxes_predicted', f'{inference_metric.confusion_matrix_summary["number_of_bounding_boxes_predicted"]:.0f}'])
#     sheet_list.append(['number_of_bounding_boxes_predicted_with_target', f'{inference_metric.confusion_matrix_summary["number_of_bounding_boxes_predicted_with_target"]:.0f}'])
#     sheet_list.append(['number_of_incorrect_predictions', f'{inference_metric.confusion_matrix_summary["number_of_ghost_predictions"]:.0f}'])
#     sheet_list.append(['number_of_undetected_objects', f'{inference_metric.confusion_matrix_summary["number_of_undetected_objects"]:.0f}'])    

#     # saving metrics sheet
#     Utils.save_metrics_excel(path_and_filename, sheet_name, sheet_list)
#     logging_sheet(sheet_list)


# def test_neural_network_model_ultralytics(parameters, device, model):
#     '''
#     Execute inference of the neural network model
#     Confusion matrix of YOLOv8:
#     - https://medium.com/@a0922/confusion-matrix-of-yolov8-97fd7ff0074e
#     '''

#     logging_info(f'')
#     logging_info(f'Testing images using model val from Ultralytics')
#     logging_info(f'')

#     # getting classes 
#     classes =  parameters['neural_network_model']['classes']

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
#         parameters['processing']['yolo_v8_yaml_filename_test']
#     )

#     # running test in test image dataset by 'model.val' method
#     metric_results = model.val(
#         data=data_file_yaml, 
#         imgsz=parameters['input']['input_dataset']['input_image_size'],
#         project=parameters['test_results']['results_folder'],
#         conf=parameters['neural_network_model']['threshold'],
#         iou=parameters['neural_network_model']['iou_threshold_for_validation'],
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
       
#     # logging_info(f'metric_results.box: {metric_results.box}')
#     # logging_info(f'metric_results.confusion_matrix: {metric_results.confusion_matrix}')
#     # logging_info(f'metric_results.confusion_matrix.matrix: {metric_results.confusion_matrix.matrix}')
#     # logging_info(f'metric_results.confusion_matrix.tp_fp: {metric_results.confusion_matrix.tp_fp()}')
#     # logging_info(f'metric_results.results_dict: {metric_results.results_dict}')
#     # logging_info(f'=====================================================================')
#     # logging_info(f'metric_results.box.map: {metric_results.box.map}')
#     # logging_info(f'metric_results.box.mp: {metric_results.box.mp}')
#     # logging_info(f'metric_results.box.mr: {metric_results.box.mr}')
#     # logging_info(f'metric_results.box.map50: {metric_results.box.map50}')
#     # logging_info(f'metric_results.box.map75: {metric_results.box.map75}')
#     # logging_info(f'metric_results.box.maps: {metric_results.box.maps}')
#     # logging_info(f'=====================================================================')
#     # logging_info(f'metric_results: {metric_results}')

# def copy_processing_files_to_log(parameters):
#     input_path = os.path.join(
#         parameters['processing']['research_root_folder'],
#         parameters['processing']['project_name_folder'],
#     )
#     output_path = parameters['test_results']['log_folder']
#     input_filename = output_filename = 'yolo_v8_inference_errors'
#     Utils.copy_file(input_filename, input_path, output_filename, output_path)

#     input_filename = output_filename = 'yolo_v8_inference_output'
#     Utils.copy_file(input_filename, input_path, output_filename, output_path)


# def merge_image_results(parameters):

#     # setting parameters 
#     results_folder = os.path.join(parameters['test_results']['results_folder'])
#     folder_prefix = 'predict'
#     test_image_folder = os.path.join(parameters['test_results']['inferenced_image_folder'])
#     test_image_sufix = '_predicted'

#     # copy all image files from results to one specific folder
#     YoloUtils.merge_image_results_to_one_folder(results_folder, folder_prefix, test_image_folder, test_image_sufix)


# ###########################################
# Methods of Level 3
# ###########################################


def get_original_boxes_of_dataset(parameters, dataset_type):

    # setting the filename of the original annotations of the test dataset
    if dataset_type == 'train':
        path_and_filename = os.path.join(
            parameters['processing']['image_dataset_folder_train'],
            'custom_train.json'
        )
    elif dataset_type == 'valid':
        path_and_filename = os.path.join(
            parameters['processing']['image_dataset_folder_valid'],
            'custom_valid.json'
        )
    elif dataset_type == 'test':
        path_and_filename = os.path.join(
            parameters['processing']['image_dataset_folder_test'],
            'custom_test.json'
        )
    
    logging_info(f'path_and_filename: {path_and_filename}')

    # reading original annotations of the test dataset
    custom_test = {}
    with open(path_and_filename) as json_file:
        custom_test = json.load(json_file)

    # processing bounding boxes of the annotations for two points format creating a dict
    dataset_test_original_boxes = []
    for image in custom_test['images']:
        item = {}
        item['image_id'] = image['id']
        item['image_filename'] = image['file_name']
        item['image_width'] = image['width']
        item['image_height'] = image['height']
        item['image_boxes'] = [] 
        item['image_labels'] = [] 
        for annotation in custom_test['annotations']:
            if annotation['image_id'] == image['id']:
                bbox = annotation['bbox']
                x = bbox[0]
                y = bbox[1]
                w = bbox[2]
                h = bbox[3]
                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h
                item['image_boxes'].append([x1, y1, x2, y2])
                item['image_labels'].append(annotation['category_id'])

        # adding image annotations to the dataset
        dataset_test_original_boxes.append(item)

    # logging_info(f'--------------------------------')
    # logging_info(f'Number of original boxes of testing images: {len(dataset_test_original_boxes)}')
    # logging_info(f'dataset_test_original_boxes: {dataset_test_original_boxes}')
    # logging_info(f'--------------------------------')

    # returning original annotations of the test dataset
    return dataset_test_original_boxes

# ###########################################
# Main method
# ###########################################
if __name__ == '__main__':
    main()
