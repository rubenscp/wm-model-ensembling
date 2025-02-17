"""
Implements the test images inference of the trained Faster RCNN model.
"""

# Importing python modules
from common.manage_log import *

# Import python code from debugger_cafe
from model_faster_rcnn.dataset import * 
from model_faster_rcnn.model import * 
from model_faster_rcnn.inference import * 

# ###########################################
# Methods of Level 1
# ###########################################

def model_faster_rcnn_inference(parameters, processing_tasks, device):
    logging_info('')
    logging_info('-----------------------------------')
    logging_info(f'Inferencing Faster-RCNN model')
    logging_info('-----------------------------------')
    logging_info('')    
    print(f'parameters: {parameters}')

    # setting new values of parameters according of initial parameters
    processing_tasks.start_task('Setting input image folders')
    set_input_image_folders(parameters)
    processing_tasks.finish_task('Setting input image folders')

    # copying weights file produced by training step 
    processing_tasks.start_task('Copying weights file used in inference')
    copy_weights_file(parameters)
    processing_tasks.finish_task('Copying weights file used in inference')

    # loading dataloaders of image dataset for processing
    processing_tasks.start_task('Loading dataloaders of image dataset')    
    dataset_test = get_dataset_test(parameters)
    processing_tasks.finish_task('Loading dataloaders of image dataset')
    
    # creating neural network model 
    processing_tasks.start_task('Creating neural network model')
    model = get_neural_network_model_with_custom_weights(parameters, device)
    processing_tasks.finish_task('Creating neural network model')

    # adjusting tested-images folder for Faster RCNN model
    parameters['test_results']['inferenced_image_folder'] = os.path.join(
        parameters['test_results']['running_folder'], 
        'tested-image', 
        parameters['model_faster_rcnn']['neural_network_model']['model_name'], 
    )

    Utils.create_directory(parameters['test_results']['inferenced_image_folder'])

    # inference the neural netowrk model
    processing_tasks.start_task('Running inference of test images dataset')
    all_predictions = inference_faster_rcnn_model_with_dataset_test(
        parameters, model, device, dataset_test)
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
    input_image_size = str(parameters['model_faster_rcnn']['input']['input_dataset']['input_image_size'])
    image_dataset_folder = os.path.join(
        parameters['processing']['research_root_folder'],
        parameters['model_faster_rcnn']['input']['input_dataset']['input_dataset_path'],
        parameters['model_faster_rcnn']['input']['input_dataset']['annotation_format'],
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
    logging_info(f"Folder name: {parameters['model_faster_rcnn']['input']['inference']['weights_folder']}")
    logging_info(f"Filename   : {parameters['model_faster_rcnn']['input']['inference']['weights_filename']}")

    Utils.copy_file_same_name(
        parameters['model_faster_rcnn']['input']['inference']['weights_filename'],
        parameters['model_faster_rcnn']['input']['inference']['weights_folder'],
        parameters['test_results']['weights_folder']
    )

def get_dataset_test(parameters):
    '''
    Get dataset of testing from image dataset 
    '''

    # getting dataloaders from faster rcnn dataset 
    dataset_test, dataloader_test = get_test_datasets_and_dataloaders_faster_rcnn(parameters)

    # returning dataloaders from datasets for processing 
    return dataset_test

def get_neural_network_model_with_custom_weights(parameters, device):
    '''
    Get neural network model
    '''      

    # getting model 
    model = get_object_detection_model(len(parameters['neural_network_model']['classes']))
       
    # loading weights into the model from training step
    load_weigths_into_model(parameters, model)

    number_of_parameters = count_parameters(model)
    logging.info(f'Number of model parameters: {number_of_parameters}')    

    num_layers = compute_num_layers(model)
    logging_info(f'Number of layers: {num_layers}')
  
    logging.info(f'Moving model to device: {device}')
    model = model.to(device)

    logging.info(f'Creating neural network model')
    logging.info(f'{model}')
    logging_info(f'')

    # returning neural network model
    return model











# def get_dataloaders(parameters):
#     '''
#     Get dataloaders of testing from image dataset 
#     '''

#     logging_info(f'')
#     logging_info(f'>> Get dataset and dataloaders of the images for processing')

#     # getting datasets 
#     test_dataset = create_test_dataset(
#         parameters['processing']['image_dataset_folder_test'], 
#         parameters['neural_network_model']['resize_of_input_image'], 
#         parameters['neural_network_model']['classes'], 
#     )
   
#     logging.info(f'Getting datasets')
#     logging.info(f'   Number of testing images: {len(test_dataset)}')
#     logging.info(f'   Total                   : {len(test_dataset)}')
#     logging_info(f'')

#     # getting dataloaders
#     test_dataloader = create_test_loader(
#         test_dataset, 
#         parameters['neural_network_model']['number_workers']
#     )

#     # returning dataloaders for processing 
#     return test_dataloader 

# def get_neural_network_model(parameters, device):
#     '''
#     Get neural network model
#     '''      

#     logging_info(f'')
#     logging_info(f'>> Get neural network model')

#     # Initialize the model and move to the computation device.
#     # model = create_model(num_classes=NUM_CLASSES, size=RESIZE_TO)
#     model = create_model_pytorchvision(
#         parameters['neural_network_model']['classes'], 
#         size=parameters['neural_network_model']['resize_of_input_image'],         
#         nms=parameters['model_faster_rcnn']['neural_network_model']['non_maximum_suppression'],
#         pretrained=Utils.to_boolean_value(
#             parameters['neural_network_model']['is_pre_trained_weights']
#         )
#     )
    
#     model = model.to(device)

#     number_of_parameters = count_parameters(model)
#     logging_info(f'')
#     logging.info(f'Number of model parameters: {number_of_parameters}')    

#     # num_layers = compute_num_layers(model)
#     # logging_info(f'Number of layers: {num_layers}')
#     # logging_info(f'')

#     # numer_of_flops, model_params = compute_flops(model, (3, 300, 300))
#     # logging_info(f'Number of FLOPS: {numer_of_flops} - params: {model_params}')

#     logging.info(f'{model}')

#     # returning neural network model
#     return model
