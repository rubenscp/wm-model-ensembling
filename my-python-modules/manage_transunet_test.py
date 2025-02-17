"""
Implements the test images inference of the trained TransUNet model.
"""

# torchvision libraries
import torch
# from transformers import DetrImageProcessor

# Importing python modules
from common.manage_log import *

# from model_transunet.model import *
# from model_transunet.dataset import *
# from model_transunet.inference import * 
from model_transunet.test import *


# ###########################################
# Methods of Level 1
# ###########################################

def model_trans_unet_inference(parameters, processing_tasks, device):    

    logging_info('')
    logging_info('-----------------------------------')
    logging_info(f'Inferencing TransUNet model')
    logging_info('-----------------------------------')
    logging_info('')    
    print(f'parameters: {parameters}')

    # setting new values of parameters according of initial parameters
    processing_tasks.start_task('Setting input image folders')
    set_input_image_folders(parameters)
    processing_tasks.finish_task('Setting input image folders')

    # copying weights file produced by training step 
    # processing_tasks.start_task('Copying weights file used in inference')
    # copy_weights_file(parameters)
    # processing_tasks.finish_task('Copying weights file used in inference')

    # loading datasets and dataloaders of image dataset for processing
    # processing_tasks.start_task('Loading test dataset of image dataset')
    # dataset_type = 'test'
    # dataset_test, dataset_test_original_boxes, processor = get_dataset(parameters, device, dataset_type)
    # processing_tasks.finish_task('Loading test dataset of image dataset')

    # creating neural network model 
    # processing_tasks.start_task('Creating neural network model')
    # model = get_neural_network_model(parameters, device)
    # processing_tasks.finish_task('Creating neural network model')
   
    # adjusting tested-images folder for SSD model
    parameters['test_results']['inferenced_image_folder'] = os.path.join(
        parameters['test_results']['inferenced_image_folder'], 'model_transunet'
    )
    parameters['test_results']['inferenced_image_folder'] = os.path.join(
        parameters['test_results']['running_folder'], 
        'tested-image', 
        parameters['model_transunet']['neural_network_model']['model_name'], 
    )
    Utils.create_directory(parameters['test_results']['inferenced_image_folder'])

    # inference the neural netowrk model
    processing_tasks.start_task('Running inference of test images dataset')
    all_predictions = inference_neural_network_model(parameters, device)
    processing_tasks.finish_task('Running inference of test images dataset')

    # returning predictions 
    return all_predictions


# ###########################################
# Methods of Level 2
# ###########################################

def set_input_image_folders(parameters):
    '''
    Set folder name of input images dataset
    '''    
    
    # getting image dataset folder according processing parameters 
    input_image_size = str(parameters['input']['input_dataset']['input_image_size'])
    image_dataset_folder = os.path.join(
        parameters['processing']['research_root_folder'],
        parameters['model_transunet']['input']['input_dataset']['input_dataset_path'],
        parameters['model_transunet']['input']['input_dataset']['annotation_format'],
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

def copy_weights_file(parameters):
    '''
    Copying weights file to inference step
    '''

    logging_info(f'')
    logging_info(f'>> Copy weights file of the model for inference')
    logging_info(f"Folder name: {parameters['model_transunet']['input']['inference']['weights_folder']}")
    logging_info(f"Filename   : {parameters['model_transunet']['input']['inference']['weights_filename']}")

    Utils.copy_file_same_name(
        parameters['model_transunet']['input']['inference']['weights_filename'],
        parameters['model_transunet']['input']['inference']['weights_folder'],
        parameters['test_results']['weights_folder']
    )

# def get_dataset(parameters, device, dataset_type):
#     '''
#     Get datasets and dataloaders of testing from image dataset 
#     '''

#     logging_info(f'Get dataset type: {dataset_type}')

#     # getting image dataset folders
#     if dataset_type == 'train':
#         image_dataset_folder = parameters['processing']['image_dataset_folder_train']
#     elif dataset_type == 'valid':
#         image_dataset_folder = parameters['processing']['image_dataset_folder_valid']
#     elif dataset_type == 'test':
#         image_dataset_folder = parameters['processing']['image_dataset_folder_test']

#     # setting parameters for training and validation datasets
#     pretrained_model_name_or_path = parameters['model_transunet']['neural_network_model']['pretrained_model_path']
#     cache_dir = parameters['model_transunet']['neural_network_model']['model_cache_dir']

#     # getting image processor
#     processor = DetrImageProcessor.from_pretrained(
#             pretrained_model_name_or_path=pretrained_model_name_or_path,
#             cache_dir=cache_dir,
#             revision="no_timm",
#             local_files_only=True,            
#         )
#     logging.info(f'processor: {processor}')

#     # getting datasets for training and validation
#     dataset_test  = CocoDetection(img_folder=image_dataset_folder, 
#                                   processor=processor,
#                                   dataset_type=dataset_type)

#     logging.info(f'Getting datasets')
#     logging.info(f'Number of testing images   : {len(dataset_test)}')
#     logging_info(f'')

#     # getting annotations of the dataset
#     dataset_test_original_boxes = get_original_boxes_of_dataset(parameters, dataset_type)
#     logging_info(f'Number of original boxes of testing images: {len(dataset_test_original_boxes)}')
#     logging_info(f'dataset_test_original_boxes: {dataset_test_original_boxes}')

#     # returning dataset for processing 
#     return dataset_test, dataset_test_original_boxes, processor

# def get_neural_network_model(parameters, device):
#     '''
#     Get neural network model
#     '''

#     logging_info(f'')
#     logging_info(f'>> Get neural network model')
    
#     model_name = parameters['model_transunet']['neural_network_model']['model_name']

#     logging_info(f'Model used: {model_name}')

#     learning_rate = parameters['model_transunet']['neural_network_model']['learning_rate_initial']
#     learning_rate_backbone = parameters['model_transunet']['neural_network_model']['learning_rate_backbone']
#     weight_decay = parameters['model_transunet']['neural_network_model']['weight_decay']
#     num_labels = parameters['model_transunet']['neural_network_model']['number_of_classes']
#     pretrained_model_name_or_path = parameters['model_transunet']['neural_network_model']['pretrained_model_path']
#     cache_dir = parameters['model_transunet']['neural_network_model']['model_cache_dir']
#     logging_info(f'detr - pretrained_model_name_or_path: {pretrained_model_name_or_path}')
#     logging_info(f'detr - cache_dir: {cache_dir}')
#     model = Detr(lr=learning_rate, 
#                  lr_backbone=learning_rate_backbone, 
#                  weight_decay=weight_decay,
#                  pretrained_model_name_or_path=pretrained_model_name_or_path,
#                  cache_dir=cache_dir,
#                  num_labels=num_labels,
#                  train_dataloader=None,
#                  val_dataloader=None)

#     logging_info(f'after get DETR model')

#     # loading weights of the model trained in before step
#     path_and_weigths_filename = os.path.join(
#         parameters['model_transunet']['input']['inference']['weights_folder'],
#         parameters['model_transunet']['input']['inference']['weights_filename'],
#     )      
   
#     logging_info(f'')
#     logging_info(f'Loading weights file: {path_and_weigths_filename}')
#     logging_info(f'')
#     model.load_state_dict(torch.load(path_and_weigths_filename))
#     # model.load_state_dict(torch.load(MODEL_DIR+'objectdetection.pth'))

#     # moving model into GPU 
#     model = model.to(device)

#     number_of_parameters = count_parameters(model)
#     logging.info(f'Number of model parameters: {number_of_parameters}')    

#     num_layers = compute_num_layers(model)
#     logging_info(f'Number of layers: {num_layers}')

#     logging.info(f'{model}')

#     # returning neural network model
#     return model


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
