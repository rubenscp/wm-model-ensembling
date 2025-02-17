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
This implementation is based on:
- Paper: Ensemble methods for Object detection - https://ebooks.iospress.nl/volumearticle/55202 
- Github: https://github.com/ancasag/ensembleObjectDetection/tree/master 

Note:
- Non maximum suppression: iou_threshold (float) - discards all overlapping boxes with IoU > iou_threshold

"""

# Basic python and ML Libraries
import os
from datetime import datetime
import shutil
import copy

# Importing python modules
from common.manage_log import *
from common.tasks import Tasks
from common.entity.ImageAnnotation import ImageAnnotation
from common.entity.AnnotationsStatistic import AnnotationsStatistic

from manage_ssd_test import * 
from manage_faster_rcnn_test import * 
from manage_yolo_test import * 
from manage_detr_test import * 
from manage_transunet_test import *

from model_ensemble.ensemble import * 


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
    full_path_project = '/home/lovelace/proj/proj939/rubenscp/research/white-mold-applications/wm-model-ensembling'

    # getting application parameters 
    processing_tasks.start_task('Getting application parameters')
    parameters_filename = 'wm_model_ensembling_parameters.json'
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
    logging_info('Inference of the model Ensembling' + LINE_FEED)

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
   
    original_parameters = copy.deepcopy(parameters)

    # initializing predictions dictionaries 
    ssd_predictions = {}
    faster_rcnn_predictions = {}
    yolov8_predictions = {}
    yolov9_predictions = {}
    yolov10_predictions = {}
    detr_predictions = {}
    trans_unet_predictions = {}
    
    # inference all test image with trained SSD model
    if parameters['input']['model_to_ensemble']['ssd'][1]:
        model_to_ensemble = parameters['input']['model_to_ensemble']['ssd'][0]
        processing_tasks.start_task(f'Inferencing test image with trained {model_to_ensemble} model')
        ssd_predictions = model_ssd_inference(parameters, processing_tasks, device)
        print(f'ssd_predictions - len: {len(ssd_predictions)}')
        print(f'ssd_predictions: {ssd_predictions}')
        processing_tasks.finish_task(f'Inferencing test image with trained {model_to_ensemble} model')

    # inference all test image with trained Faster RCNN model 
    if parameters['input']['model_to_ensemble']['faster_rcnn'][1]:
        model_to_ensemble = parameters['input']['model_to_ensemble']['faster_rcnn'][0]
        processing_tasks.start_task(f'Inferencing test image with trained {model_to_ensemble} model')
        faster_rcnn_predictions = model_faster_rcnn_inference(parameters, processing_tasks, device)
        print(f'faster_rcnn_predictions - len: {len(faster_rcnn_predictions)}')
        print(f'faster_rcnn_predictions: {faster_rcnn_predictions}')
        processing_tasks.finish_task(f'Inferencing test image with trained {model_to_ensemble} model')

    # inference all test image with trained YOLO series model 
    if parameters['input']['model_to_ensemble']['yolov8'][1]:
        model_to_ensemble = parameters['input']['model_to_ensemble']['yolov8'][0]
        processing_tasks.start_task(f'Inferencing test image with trained {model_to_ensemble} model')
        yolov8_predictions = model_yolo_inference(parameters, processing_tasks, device, model_to_ensemble)
        print(f'yolov8_predictions - len: {len(yolov8_predictions)}')
        print(f'yolov8_predictions: {yolov8_predictions}')
        processing_tasks.finish_task(f'Inferencing test image with trained {model_to_ensemble} model')

    if parameters['input']['model_to_ensemble']['yolov9'][1]:
        model_to_ensemble = parameters['input']['model_to_ensemble']['yolov9'][0]
        processing_tasks.start_task(f'Inferencing test image with trained {model_to_ensemble} model')
        yolov9_predictions = model_yolo_inference(parameters, processing_tasks, device, model_to_ensemble)
        print(f'yolov9_predictions - len: {len(yolov9_predictions)}')
        print(f'yolov9_predictions: {yolov9_predictions}')
        processing_tasks.finish_task(f'Inferencing test image with trained {model_to_ensemble} model')

    if parameters['input']['model_to_ensemble']['yolov10'][1]:
        model_to_ensemble = parameters['input']['model_to_ensemble']['yolov10'][0]
        processing_tasks.start_task(f'Inferencing test image with trained {model_to_ensemble} model')
        yolov10_predictions = model_yolo_inference(parameters, processing_tasks, device, model_to_ensemble)
        print(f'yolov10_predictions - len: {len(yolov10_predictions)}')
        print(f'yolov10_predictions: {yolov10_predictions}')
        processing_tasks.finish_task(f'Inferencing test image with trained {model_to_ensemble} model')

    if parameters['input']['model_to_ensemble']['detr'][1]:
        model_to_ensemble = parameters['input']['model_to_ensemble']['detr'][0]
        processing_tasks.start_task(f'Inferencing test image with trained {model_to_ensemble} model')
        detr_predictions = model_detr_inference(parameters, processing_tasks, device)
        print(f'detr_predictions - len: {len(detr_predictions)}')
        print(f'detr_predictions: {detr_predictions}')
        processing_tasks.finish_task(f'Inferencing test image with trained {model_to_ensemble} model')

    if parameters['input']['model_to_ensemble']['trans_unet'][1]:
        model_to_ensemble = parameters['input']['model_to_ensemble']['trans_unet'][0]
        processing_tasks.start_task(f'Inferencing test image with trained {model_to_ensemble} model')
        trans_unet_predictions = model_trans_unet_inference(parameters, processing_tasks, device)
        print(f'trans_unet_predictions - len: {len(trans_unet_predictions)}')
        print(f'trans_unet_predictions: {trans_unet_predictions}')
        processing_tasks.finish_task(f'Inferencing test image with trained {model_to_ensemble} model')

    print(f'')
    print(f'======================================================')
    print(f'')
    print(f'ssd_predictions         - len: {len(ssd_predictions)}')
    print(f'faster_rcnn_predictions - len: {len(faster_rcnn_predictions)}')
    print(f'yolov8_predictions      - len: {len(yolov8_predictions)}')
    print(f'yolov9_predictions      - len: {len(yolov9_predictions)}')
    print(f'yolov10_predictions     - len: {len(yolov10_predictions)}')
    print(f'detr_predictions        - len: {len(detr_predictions)}')
    print(f'trans_unet_predictions  - len: {len(trans_unet_predictions)}')
    print(f'======================================================')

    logging_info(f'')
    logging_info(f'Summary of Previous Model Predictions')    
    logging_info(f'')
    logging_info(f'ssd_predictions         - len: {len(ssd_predictions)}')
    logging_info(f'faster_rcnn_predictions - len: {len(faster_rcnn_predictions)}')
    logging_info(f'yolov8_predictions      - len: {len(yolov8_predictions)}')
    logging_info(f'yolov9_predictions      - len: {len(yolov9_predictions)}')
    logging_info(f'yolov10_predictions     - len: {len(yolov10_predictions)}')
    logging_info(f'detr_predictions        - len: {len(detr_predictions)}')
    logging_info(f'trans_unet_predictions  - len: {len(trans_unet_predictions)}')
    logging_info(f'')

    # build all predictions for ensembling of the predictions
    all_predictions = {}
    all_predictions[parameters['input']['model_to_ensemble']['ssd'][0]] = ssd_predictions
    all_predictions[parameters['input']['model_to_ensemble']['faster_rcnn'][0]] = faster_rcnn_predictions
    all_predictions[parameters['input']['model_to_ensemble']['yolov8'][0]] = yolov8_predictions
    all_predictions[parameters['input']['model_to_ensemble']['yolov9'][0]] = yolov9_predictions
    all_predictions[parameters['input']['model_to_ensemble']['yolov10'][0]] = yolov10_predictions
    all_predictions[parameters['input']['model_to_ensemble']['detr'][0]] = detr_predictions
    all_predictions[parameters['input']['model_to_ensemble']['trans_unet'][0]] = trans_unet_predictions

    # run ensemble of the all predictions
    processing_tasks.start_task(f'Ensembling inference results of the previous models')
    run_ensemble_test_images(original_parameters, all_predictions)
    processing_tasks.finish_task(f'Ensembling inference results of the previous models')
    
    # # copying pre-trained files
    # processing_tasks.start_task('Copying pre-trained files')
    # copy_pretrained_model_files(parameters)
    # processing_tasks.finish_task('Copying pre-trained files')     

    # # copying weights file produced by training step 
    # # processing_tasks.start_task('Copying weights file used in inference')
    # # copy_weights_file(parameters)
    # # processing_tasks.finish_task('Copying weights file used in inference')

    # # loading datasets and dataloaders of image dataset for processing
    # processing_tasks.start_task('Loading test dataset of image dataset')
    # dataset_type = 'test'
    # dataset_test, dataset_test_original_boxes, processor = get_dataset(parameters, device, dataset_type)
    # processing_tasks.finish_task('Loading test dataset of image dataset')

    # # creating neural network model 
    # processing_tasks.start_task('Creating neural network model')
    # model = get_neural_network_model(parameters, device)
    # processing_tasks.finish_task('Creating neural network model')

    # # getting statistics of input dataset 
    # if parameters['processing']['show_statistics_of_input_dataset']:
    #     processing_tasks.start_task('Getting statistics of input dataset')
    #     annotation_statistics = get_input_dataset_statistics(parameters)
    #     show_input_dataset_statistics(parameters, annotation_statistics)
    #     processing_tasks.finish_task('Getting statistics of input dataset')    

    # # inference the neural netowrk model
    # processing_tasks.start_task('Running prediction on the test images dataset')
    # inference_detr_model(parameters, device, model, processor, dataset_test, dataset_test_original_boxes)
    # processing_tasks.finish_task('Running prediction on the test images dataset')

    # # showing input dataset statistics
    # # if parameters['processing']['show_statistics_of_input_dataset']:
    # #     show_input_dataset_statistics(parameters, annotation_statistics)

    # # merging all image rsults to just one folder
    # # merge_image_results(parameters)

    # finishing model training 
    logging_info('')
    logging_info('Finished the test of the model Ensembling' + LINE_FEED)
    print('Finished the test of the model Ensembling')

    # printing tasks summary 
    processing_tasks.finish_processing()
    logging_info(processing_tasks.to_string())


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

    affirmative_strategy_folder = os.path.join(
        inferenced_image_folder,
        parameters['test_results']['affirmative_strategy_folder']
    )
    parameters['test_results']['affirmative_strategy_folder'] = affirmative_strategy_folder
    Utils.create_directory(affirmative_strategy_folder)

    consensus_strategy_folder = os.path.join(
        inferenced_image_folder,
        parameters['test_results']['consensus_strategy_folder']
    )
    parameters['test_results']['consensus_strategy_folder'] = consensus_strategy_folder
    Utils.create_directory(consensus_strategy_folder)

    unanimous_strategy_folder = os.path.join(
        inferenced_image_folder,
        parameters['test_results']['unanimous_strategy_folder']
    )
    parameters['test_results']['unanimous_strategy_folder'] = unanimous_strategy_folder
    Utils.create_directory(unanimous_strategy_folder)

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
