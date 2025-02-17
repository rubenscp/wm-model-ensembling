"""
Implements the test images inference of the trained YOLO model.
"""

# Importing libraries 
from ultralytics import YOLO

# Importing python modules
from common.manage_log import *
from common.tasks import Tasks
from common.entity.ImageAnnotation import ImageAnnotation
from common.entity.AnnotationsStatistic import AnnotationsStatistic
from common.metrics import *

from model_yolo.yolo_utils import *
from model_yolo.create_yaml_file import * 
from model_yolo.inference import * 

# ###########################################
# Methods of Level 1
# ###########################################

def model_yolo_inference(parameters, processing_tasks, device, yolo_model_name):
    logging_info('')
    logging_info('-----------------------------------')
    logging_info(f'Inferencing {yolo_model_name} model')
    logging_info('-----------------------------------')
    logging_info('')    
    print(f'parameters: {parameters}')

    # setting new values of parameters according of initial parameters
    processing_tasks.start_task('Setting input image folders')
    set_input_image_folders(parameters, yolo_model_name)
    processing_tasks.finish_task('Setting input image folders')

    # creating yaml file with parameters used by Ultralytics
    processing_tasks.start_task('Creating yaml file for ultralytics')
    create_yaml_file_for_ultralytics(parameters, yolo_model_name)
    processing_tasks.finish_task('Creating yaml file for ultralytics')

    # copying weights file produced by training step 
    processing_tasks.start_task('Copying weights file used in inference')
    copy_weights_file(parameters, yolo_model_name)
    processing_tasks.finish_task('Copying weights file used in inference')

    # loading dataloaders of image dataset for processing
    # processing_tasks.start_task('Loading dataloaders of image dataset')    
    # dataset_test = get_dataset_test(parameters)
    # processing_tasks.finish_task('Loading dataloaders of image dataset')
    
    # creating neural network model 
    processing_tasks.start_task('Creating neural network model')
    model = get_neural_network_model(parameters, device, yolo_model_name)
    processing_tasks.finish_task('Creating neural network model')

    # adjusting tested-images folder for Faster RCNN model
    model_name = 'model_' + yolo_model_name
    parameters['test_results']['inferenced_image_folder'] = os.path.join(
        parameters['test_results']['running_folder'], 
        'tested-image', 
        parameters[model_name]['neural_network_model']['model_name'], 
    )
    Utils.create_directory(parameters['test_results']['inferenced_image_folder'])

    # inference the neural netowrk model
    processing_tasks.start_task('Running inference of the test images dataset')
    all_predictions = inference_yolo_model(parameters, device, model, yolo_model_name)
    processing_tasks.finish_task('Running inference of the test images dataset')

    # returning predictions 
    return all_predictions


# ###########################################
# Methods of Level 2
# ###########################################

def set_input_image_folders(parameters, yolo_model_name):
    '''
    Set folder name of input images dataset
    '''    
    # setting label of the yolo model name 
    yolo_model_name_key = 'model_' + yolo_model_name

    # getting image dataset folder according processing parameters 
    input_image_size = str(parameters[yolo_model_name_key]['input']['input_dataset']['input_image_size'])
    image_dataset_folder = os.path.join(
        parameters['processing']['research_root_folder'],
        parameters[yolo_model_name_key]['input']['input_dataset']['input_dataset_path'],
        parameters[yolo_model_name_key]['input']['input_dataset']['annotation_format'],
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

def create_yaml_file_for_ultralytics(parameters, yolo_model_name):

    # setting label of the yolo model name 
    yolo_model_name_key = 'model_' + yolo_model_name
    yolo_yaml_filename = parameters[yolo_model_name_key]['processing']['yolo_yaml_filename_test']

    path_and_filename_white_mold_yaml = os.path.join(
        parameters['processing']['research_root_folder'],
        parameters['processing']['project_name_folder'],
        yolo_yaml_filename
    )
    image_dataset_folder = parameters['processing']['image_dataset_folder']
    number_of_classes = parameters[yolo_model_name_key]['neural_network_model']['number_of_classes']
    classes = (parameters['neural_network_model']['classes'])[:(number_of_classes+1)]
   
    # creating yaml file 
    create_project_yaml_file_for_test(
        path_and_filename_white_mold_yaml,
        image_dataset_folder,
        classes,    
    )

def copy_weights_file(parameters, yolo_model_name):
    '''
    Copying weights file to inference step
    '''

    # setting the yolo name 
    model_name = 'model_' + yolo_model_name
    print(f'model_name: {model_name}')

    logging_info(f'')
    logging_info(f'>> Copy weights file of the model for inference')
    # weights_folder_name = parameters[model_name]['input']['inference']['weights_folder']
    # weights_filename = parameters[model_name]['input']['inference']['weights_filename']
    # print(f'weights_folder_name: {weights_folder_name}')
    # print(f'weights_filename: {weights_filename}')

    # logging_info(f"Folder name: {weights_folder_name}")
    # logging_info(f"Filename   : {weights_filename}")
    # Utils.copy_file_same_name(
    #     weights_folder_name,
    #     weights_filename,
    #     parameters['test_results']['weights_folder']
    # )
    print(f"{parameters[model_name]['input']['inference']['weights_folder']}")
    print(f"{parameters[model_name]['input']['inference']['weights_filename']}")
    print(f"{parameters['test_results']['weights_folder']}")

    logging_info(f"Folder name: {parameters[model_name]['input']['inference']['weights_folder']}")
    logging_info(f"Filename   : {parameters[model_name]['input']['inference']['weights_filename']}")
    logging_info(f"test_results : {parameters['test_results']['weights_folder']}")
    Utils.copy_file_same_name(
        parameters[model_name]['input']['inference']['weights_filename'],
        parameters[model_name]['input']['inference']['weights_folder'],
        parameters['test_results']['weights_folder']
    )


# def get_dataset_test(parameters):
#     '''
#     Get dataset of testing from image dataset 
#     '''

#     # getting dataloaders from faster rcnn dataset 
#     dataset_test, dataloader_test = get_test_datasets_and_dataloaders_faster_rcnn(parameters)

#     # returning dataloaders from datasets for processing 
#     return dataset_test

def get_neural_network_model(parameters, device, yolo_model_name):
    '''
    Get neural network model
    '''

    # setting the yolo name 
    model_name = 'model_' + yolo_model_name

    logging_info(f'')
    logging_info(f'>> Get neural network model')

    # Load a YOLO model with the pretrained weights
    path_and_yolo_model_filename_with_best_weights = os.path.join(
        parameters[model_name]['input']['inference']['weights_folder'],
        parameters[model_name]['input']['inference']['weights_filename'],
    )
    logging_info(f'Model used with the best weights of the training step: {path_and_yolo_model_filename_with_best_weights}')

    model = YOLO(path_and_yolo_model_filename_with_best_weights)

    logging.info(f'model: {model}')
    logging.info(f'model.info(): {yolo_model_name} - {model.info()}')

    # returning neural network model
    return model
