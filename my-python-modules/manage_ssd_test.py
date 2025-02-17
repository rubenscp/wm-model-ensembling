"""
Implements the test images inference of the trained SSD model.
"""

# Importing python modules
from common.manage_log import *

# Import python code from debugger_cafe
from model_ssd.debugger_cafe.datasets import * 
from model_ssd.debugger_cafe.model import * 
from model_ssd.debugger_cafe.inference import * 

# ###########################################
# Methods of Level 1
# ###########################################

def model_ssd_inference(parameters, processing_tasks, device):
    logging_info('')
    logging_info('-----------------------------------')
    logging_info(f'Inferencing SSD model')
    logging_info('-----------------------------------')
    logging_info('')    
    print(f'parameters: {parameters}')

    # copying weights file produced by training step 
    processing_tasks.start_task('Copying weights file used in inference')
    copy_weights_file(parameters)
    processing_tasks.finish_task('Copying weights file used in inference')

    # loading dataloaders of image dataset for processing
    processing_tasks.start_task('Loading dataloaders of image dataset')
    test_dataloader = get_dataloaders(parameters)
    processing_tasks.finish_task('Loading dataloaders of image dataset')
    
    # creating neural network model 
    processing_tasks.start_task('Creating neural network model')
    model = get_neural_network_model(parameters, device)
    processing_tasks.finish_task('Creating neural network model')
    
    # adjusting tested-images folder for SSD model
    parameters['test_results']['inferenced_image_folder'] = os.path.join(
        parameters['test_results']['inferenced_image_folder'], 'model_ssd'
    )
    parameters['test_results']['inferenced_image_folder'] = os.path.join(
        parameters['test_results']['running_folder'], 
        'tested-image', 
        parameters['model_ssd']['neural_network_model']['model_name'], 
    )
    Utils.create_directory(parameters['test_results']['inferenced_image_folder'])

    # inference the neural netowrk model
    processing_tasks.start_task('Running inference of test images dataset')
    all_predictions = inference_neural_network_model(parameters, device, model)
    processing_tasks.finish_task('Running inference of test images dataset')

    # returning predictions 
    return all_predictions


# ###########################################
# Methods of Level 2
# ###########################################

def copy_weights_file(parameters):
    '''
    Copying weights file to inference step
    '''

    logging_info(f'')
    logging_info(f'>> Copy weights file of the model for inference')
    logging_info(f"Folder name: {parameters['model_ssd']['input']['inference']['weights_folder']}")
    logging_info(f"Filename   : {parameters['model_ssd']['input']['inference']['weights_filename']}")

    Utils.copy_file_same_name(
        parameters['model_ssd']['input']['inference']['weights_filename'],
        parameters['model_ssd']['input']['inference']['weights_folder'],
        parameters['test_results']['weights_folder']
    )

def get_dataloaders(parameters):
    '''
    Get dataloaders of testing from image dataset 
    '''

    logging_info(f'')
    logging_info(f'>> Get dataset and dataloaders of the images for processing')

    # getting datasets 
    test_dataset = create_test_dataset(
        parameters['processing']['image_dataset_folder_test'], 
        parameters['neural_network_model']['resize_of_input_image'], 
        parameters['neural_network_model']['classes'], 
    )
   
    logging.info(f'Getting datasets')
    logging.info(f'   Number of testing images: {len(test_dataset)}')
    logging.info(f'   Total                   : {len(test_dataset)}')
    logging_info(f'')

    # getting dataloaders
    test_dataloader = create_test_loader(
        test_dataset, 
        parameters['neural_network_model']['number_workers']
    )

    # returning dataloaders for processing 
    return test_dataloader 

def get_neural_network_model(parameters, device):
    '''
    Get neural network model
    '''      

    logging_info(f'')
    logging_info(f'>> Get neural network model')

    # Initialize the model and move to the computation device.
    # model = create_model(num_classes=NUM_CLASSES, size=RESIZE_TO)
    model = create_model_pytorchvision(
        parameters['neural_network_model']['classes'], 
        size=parameters['neural_network_model']['resize_of_input_image'],         
        nms=parameters['model_ssd']['neural_network_model']['non_maximum_suppression'],
        pretrained=Utils.to_boolean_value(
            parameters['model_ssd']['neural_network_model']['is_pre_trained_weights']
        )
    )
    
    model = model.to(device)

    number_of_parameters = count_parameters(model)
    logging_info(f'')
    logging.info(f'Number of model parameters: {number_of_parameters}')    

    # num_layers = compute_num_layers(model)
    # logging_info(f'Number of layers: {num_layers}')
    # logging_info(f'')

    # numer_of_flops, model_params = compute_flops(model, (3, 300, 300))
    # logging_info(f'Number of FLOPS: {numer_of_flops} - params: {model_params}')

    logging.info(f'{model}')

    # returning neural network model
    return model
