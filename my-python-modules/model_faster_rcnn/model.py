import torch 
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Importing python modules
from common.manage_log import *

def get_object_detection_model(num_classes):

    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    input_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(input_features, num_classes)

    # returning the model object created
    return model

def save_model_and_weights(model, parameters): 

    # saving model after training in the hard drive of virtual machine
    # now = datetime.now()

    # formatted_now = str(formatted_now).replace('-','_').replace(' ','_').replace(':','_').replace('.','_')
    # formatted_now = now.strftime("%Y_%m_%d_%Hh_%Mm")

    path_and_full_model_base_filename = os.path.join(
        parameters['training_results']['weights_folder'],
        parameters['training_results']['full_model_base_filename'] + '.pth'
    )

    # model_filename = FASTER_RCNN_MODEL_FILENAME + '_' + formatted_now + FASTER_RCNN_MODEL_EXTENSION_FILENAME
    # local_faster_rcnn_entire_model_path_and_filename = os.path.join(local_results_faster_rcnn_path, model_filename)
    torch.save(model, path_and_full_model_base_filename)
    logging_info(f'Faster R-CNN entire model saved at: {path_and_full_model_base_filename}')
    logging_info(f'')

    path_and_weights_base_filename = os.path.join(
        parameters['training_results']['weights_folder'],
        parameters['training_results']['weights_base_filename'] + '.pth'
    )
    # weights_filename = FASTER_RCNN_MODEL_WEIGHTS_FILENAME + '_' + formatted_now + FASTER_RCNN_MODEL_EXTENSION_FILENAME
    # local_faster_rcnn_weigths_path_and_filename = os.path.join(local_results_faster_rcnn_path, weights_filename)
    torch.save(model.state_dict(), path_and_weights_base_filename)
    logging_info(f'Faster R-CNN weights (state_dict()saved at: {path_and_weights_base_filename}')

def load_weigths_into_model(parameters, model):

    # setting path and wieghts filename used to load into the model 
    path_and_weights_filename = os.path.join(
        parameters['model_faster_rcnn']['input']['inference']['weights_folder'],
        parameters['model_faster_rcnn']['input']['inference']['weights_filename'],
    )
    
    logging_info(f'Loading weights into the model from training step: {path_and_weights_filename}')

    # loading weights
    weights = torch.load(path_and_weights_filename)

    # setting weights into the model 
    model.load_state_dict(weights)


# Function to count the number of parameters
def count_parameters(model):
    number_of_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)    
    return number_of_parameters

def count_layers(module):
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
        return 1
    return 0

def compute_num_layers(model):
    num_layers = sum(count_layers(layer) for layer in model.modules())
    return num_layers
